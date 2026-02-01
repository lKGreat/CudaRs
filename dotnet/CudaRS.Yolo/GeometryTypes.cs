using System;
using System.Collections.Generic;
using System.Linq;

namespace CudaRS.Yolo;

/// <summary>
/// YOLO model version enumeration covering all major releases.
/// </summary>
public enum YoloVersion
{
    Auto = 0,
    V3 = 3,
    V4 = 4,
    V5 = 5,
    V6 = 6,
    V7 = 7,
    V8 = 8,
    V9 = 9,
    V10 = 10,
    V11 = 11,
}

/// <summary>
/// YOLO task type.
/// </summary>
public enum YoloTask
{
    Detect = 0,
    Segment = 1,
    Pose = 2,
    Classify = 3,
    Obb = 4,
}

/// <summary>
/// Inference backend selection.
/// </summary>
public enum InferenceBackend
{
    Auto = 0,
    OnnxRuntime = 1,
    TensorRT = 2,
    TorchScript = 3,
    OpenVino = 4,
}

/// <summary>
/// Bounding box format.
/// </summary>
public enum BoxFormat
{
    CenterWH = 0,
    Xyxy = 1,
    Xywh = 2,
}

/// <summary>
/// 2D point.
/// </summary>
public readonly struct Point2D : IEquatable<Point2D>
{
    public float X { get; }
    public float Y { get; }

    public Point2D(float x, float y) { X = x; Y = y; }

    public Point2D Scale(float sx, float sy) => new(X * sx, Y * sy);
    public float DistanceTo(Point2D other) => MathF.Sqrt(MathF.Pow(X - other.X, 2) + MathF.Pow(Y - other.Y, 2));

    public bool Equals(Point2D other) => X == other.X && Y == other.Y;
    public override bool Equals(object? obj) => obj is Point2D p && Equals(p);
    public override int GetHashCode() => HashCode.Combine(X, Y);
    public override string ToString() => $"({X:F1},{Y:F1})";

    public static bool operator ==(Point2D a, Point2D b) => a.Equals(b);
    public static bool operator !=(Point2D a, Point2D b) => !a.Equals(b);
}

/// <summary>
/// Axis-aligned bounding box.
/// </summary>
public readonly struct BoundingBox : IEquatable<BoundingBox>
{
    public float X { get; }
    public float Y { get; }
    public float Width { get; }
    public float Height { get; }

    public float Left => X;
    public float Top => Y;
    public float Right => X + Width;
    public float Bottom => Y + Height;
    public float CenterX => X + Width / 2;
    public float CenterY => Y + Height / 2;
    public Point2D Center => new(CenterX, CenterY);
    public float Area => Width * Height;

    public BoundingBox(float x, float y, float width, float height)
    {
        X = x; Y = y; Width = width; Height = height;
    }

    public static BoundingBox FromXyxy(float x1, float y1, float x2, float y2)
        => new(x1, y1, x2 - x1, y2 - y1);

    public static BoundingBox FromCenterWH(float cx, float cy, float w, float h)
        => new(cx - w / 2, cy - h / 2, w, h);

    public BoundingBox Scale(float sx, float sy)
        => new(X * sx, Y * sy, Width * sx, Height * sy);

    public BoundingBox Clamp(float maxWidth, float maxHeight)
    {
        var x = Math.Max(0, X);
        var y = Math.Max(0, Y);
        var r = Math.Min(maxWidth, Right);
        var b = Math.Min(maxHeight, Bottom);
        return new BoundingBox(x, y, Math.Max(0, r - x), Math.Max(0, b - y));
    }

    public float IoU(BoundingBox other)
    {
        var interLeft = Math.Max(Left, other.Left);
        var interTop = Math.Max(Top, other.Top);
        var interRight = Math.Min(Right, other.Right);
        var interBottom = Math.Min(Bottom, other.Bottom);

        if (interRight <= interLeft || interBottom <= interTop)
            return 0f;

        var interArea = (interRight - interLeft) * (interBottom - interTop);
        var unionArea = Area + other.Area - interArea;
        return unionArea > 0 ? interArea / unionArea : 0f;
    }

    public bool Contains(Point2D point)
        => point.X >= Left && point.X <= Right && point.Y >= Top && point.Y <= Bottom;

    public bool Equals(BoundingBox other)
        => X == other.X && Y == other.Y && Width == other.Width && Height == other.Height;

    public override bool Equals(object? obj) => obj is BoundingBox b && Equals(b);
    public override int GetHashCode() => HashCode.Combine(X, Y, Width, Height);
    public override string ToString() => $"[{X:F1},{Y:F1},{Width:F1},{Height:F1}]";

    public static bool operator ==(BoundingBox a, BoundingBox b) => a.Equals(b);
    public static bool operator !=(BoundingBox a, BoundingBox b) => !a.Equals(b);
}

/// <summary>
/// Rotated bounding box (OBB).
/// </summary>
public readonly struct RotatedBox : IEquatable<RotatedBox>
{
    public float CenterX { get; }
    public float CenterY { get; }
    public float Width { get; }
    public float Height { get; }
    public float AngleDegrees { get; }

    public Point2D Center => new(CenterX, CenterY);
    public float AngleRadians => AngleDegrees * MathF.PI / 180f;
    public float Area => Width * Height;

    public RotatedBox(float centerX, float centerY, float width, float height, float angleDegrees)
    {
        CenterX = centerX;
        CenterY = centerY;
        Width = width;
        Height = height;
        AngleDegrees = angleDegrees;
    }

    public Point2D[] GetCorners()
    {
        var cos = MathF.Cos(AngleRadians);
        var sin = MathF.Sin(AngleRadians);
        var hw = Width / 2;
        var hh = Height / 2;

        var corners = new[]
        {
            new Point2D(-hw, -hh),
            new Point2D(hw, -hh),
            new Point2D(hw, hh),
            new Point2D(-hw, hh),
        };

        var result = new Point2D[corners.Length];
        var cx = CenterX;
        var cy = CenterY;
        for (int i = 0; i < corners.Length; i++)
        {
            var c = corners[i];
            result[i] = new Point2D(
                cx + c.X * cos - c.Y * sin,
                cy + c.X * sin + c.Y * cos
            );
        }

        return result;
    }

    public BoundingBox GetAxisAlignedBox()
    {
        var corners = GetCorners();
        var minX = corners.Min(c => c.X);
        var minY = corners.Min(c => c.Y);
        var maxX = corners.Max(c => c.X);
        var maxY = corners.Max(c => c.Y);
        return new BoundingBox(minX, minY, maxX - minX, maxY - minY);
    }

    public RotatedBox Scale(float sx, float sy)
        => new(CenterX * sx, CenterY * sy, Width * sx, Height * sy, AngleDegrees);

    public bool Equals(RotatedBox other)
        => CenterX == other.CenterX && CenterY == other.CenterY &&
           Width == other.Width && Height == other.Height && AngleDegrees == other.AngleDegrees;

    public override bool Equals(object? obj) => obj is RotatedBox r && Equals(r);
    public override int GetHashCode() => HashCode.Combine(CenterX, CenterY, Width, Height, AngleDegrees);
    public override string ToString() => $"[{CenterX:F1},{CenterY:F1},{Width:F1},{Height:F1},θ={AngleDegrees:F1}°]";

    public static bool operator ==(RotatedBox a, RotatedBox b) => a.Equals(b);
    public static bool operator !=(RotatedBox a, RotatedBox b) => !a.Equals(b);
}
