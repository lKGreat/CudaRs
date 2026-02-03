namespace CudaRS.Yolo;

/// <summary>
/// Classification result.
/// </summary>
public sealed class Classification
{
    public int ClassId { get; init; }
    public string ClassName { get; init; } = string.Empty;
    public float Confidence { get; init; }

    public override string ToString()
        => $"{ClassName}({ClassId}): {Confidence:P1}";
}
