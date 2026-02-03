namespace CudaRS.Yolo;

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
