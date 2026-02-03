using System;
using System.Linq;

namespace CudaRS.Yolo;

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
    public override string ToString() => $"[{CenterX:F1},{CenterY:F1},{Width:F1},{Height:F1},deg={AngleDegrees:F1}]";

    public static bool operator ==(RotatedBox a, RotatedBox b) => a.Equals(b);
    public static bool operator !=(RotatedBox a, RotatedBox b) => !a.Equals(b);
}
