namespace CudaRS.Fluent;

/// <summary>
/// Backend/provider choice for a pipeline. Exactly one must be selected.
/// </summary>
public enum ProviderKind
{
    None = 0,
    TensorRt = 1,
    Onnx = 2,
    OpenVinoGpu = 3,
    OpenVinoCpu = 4,
    Paddle = 5,
}
