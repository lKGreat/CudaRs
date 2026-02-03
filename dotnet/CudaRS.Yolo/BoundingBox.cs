using System;

namespace CudaRS.Yolo;

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
        X = x;
        Y = y;
        Width = width;
        Height = height;
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
