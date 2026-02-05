namespace CudaRS.OpenVino;

/// <summary>
/// Model precision hints for OpenVINO inference.
/// </summary>
public enum ModelPrecision
{
    /// <summary>
    /// Automatic precision selection (default).
    /// </summary>
    Auto,

    /// <summary>
    /// 32-bit floating point precision.
    /// </summary>
    FP32,

    /// <summary>
    /// 16-bit floating point precision.
    /// </summary>
    FP16,

    /// <summary>
    /// 8-bit integer precision (quantized).
    /// </summary>
    INT8,
}
