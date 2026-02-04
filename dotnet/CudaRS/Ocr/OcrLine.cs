namespace CudaRS.Ocr;

public sealed class OcrLine
{
    public float[] Points { get; init; } = new float[8];
    public float Score { get; init; }
    public int ClassLabel { get; init; }
    public float ClassScore { get; init; }
    public string Text { get; init; } = string.Empty;
}
