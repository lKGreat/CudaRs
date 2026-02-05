namespace CudaRS.Yolo;

public sealed class AnnotatedImageResult
{
    public byte[] ImageBytes { get; init; } = [];
    public ImageFormat Format { get; init; }
    public int Width { get; init; }
    public int Height { get; init; }
}

public sealed class CombinedResult
{
    public ModelInferenceResult Inference { get; init; } = null!;
    public AnnotatedImageResult? AnnotatedImage { get; init; }
}
