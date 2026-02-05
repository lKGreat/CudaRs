using System;
using System.Linq;

namespace CudaRS.OpenVino;

/// <summary>
/// Model tensor information (input or output metadata).
/// </summary>
public sealed class ModelTensorInfo
{
    /// <summary>
    /// Gets the name of the tensor.
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets the shape of the tensor.
    /// </summary>
    public long[] Shape { get; set; } = Array.Empty<long>();

    /// <summary>
    /// Gets the element type of the tensor.
    /// </summary>
    public TensorElementType ElementType { get; set; }

    /// <summary>
    /// Gets a string representation of the tensor info.
    /// </summary>
    public override string ToString()
    {
        var shapeStr = string.Join(", ", Shape.Select(d => d.ToString()));
        return $"{Name}: [{shapeStr}] ({ElementType})";
    }
}

/// <summary>
/// OpenVINO tensor element types.
/// </summary>
public enum TensorElementType
{
    /// <summary>
    /// Undefined type.
    /// </summary>
    Undefined = 0,

    /// <summary>
    /// 16-bit floating point (FP16).
    /// </summary>
    F16 = 3,

    /// <summary>
    /// 32-bit floating point (FP32).
    /// </summary>
    F32 = 4,

    /// <summary>
    /// 32-bit integer.
    /// </summary>
    I32 = 9,

    /// <summary>
    /// 64-bit integer.
    /// </summary>
    I64 = 10,

    /// <summary>
    /// 8-bit unsigned integer.
    /// </summary>
    U8 = 16,
}
