namespace CudaRS.Yolo;

/// <summary>
/// Single keypoint with position and confidence.
/// </summary>
public readonly struct Keypoint
{
    public KeypointType Type { get; }
    public float X { get; }
    public float Y { get; }
    public float Confidence { get; }

    public bool IsVisible => Confidence > 0;
    public Point2D Position => new(X, Y);

    public Keypoint(KeypointType type, float x, float y, float confidence)
    {
        Type = type;
        X = x;
        Y = y;
        Confidence = confidence;
    }

    public Keypoint Scale(float sx, float sy)
        => new(Type, X * sx, Y * sy, Confidence);

    public override string ToString()
        => $"{Type}: ({X:F1},{Y:F1}) conf={Confidence:F2}";
}
