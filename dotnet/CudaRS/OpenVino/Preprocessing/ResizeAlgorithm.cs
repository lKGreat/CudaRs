namespace CudaRS.OpenVino.Preprocessing;

/// <summary>
/// OpenVINO resize algorithms for preprocessing.
/// </summary>
public enum ResizeAlgorithm
{
    /// <summary>
    /// Resize with linear interpolation.
    /// </summary>
    Linear = 0,

    /// <summary>
    /// Resize with cubic interpolation.
    /// </summary>
    Cubic = 1,

    /// <summary>
    /// Resize with nearest neighbor interpolation.
    /// </summary>
    Nearest = 2,
}
