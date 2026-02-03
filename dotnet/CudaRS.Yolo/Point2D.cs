using System;

namespace CudaRS.Yolo;

/// <summary>
/// 2D point.
/// </summary>
public readonly struct Point2D : IEquatable<Point2D>
{
    public float X { get; }
    public float Y { get; }

    public Point2D(float x, float y)
    {
        X = x;
        Y = y;
    }

    public Point2D Scale(float sx, float sy) => new(X * sx, Y * sy);
    public float DistanceTo(Point2D other) => MathF.Sqrt(MathF.Pow(X - other.X, 2) + MathF.Pow(Y - other.Y, 2));

    public bool Equals(Point2D other) => X == other.X && Y == other.Y;
    public override bool Equals(object? obj) => obj is Point2D p && Equals(p);
    public override int GetHashCode() => HashCode.Combine(X, Y);
    public override string ToString() => $"({X:F1},{Y:F1})";

    public static bool operator ==(Point2D a, Point2D b) => a.Equals(b);
    public static bool operator !=(Point2D a, Point2D b) => !a.Equals(b);
}
