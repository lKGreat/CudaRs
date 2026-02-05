using System;

namespace CudaRS.OpenVino;

/// <summary>
/// Represents a partial dimension that can be either static (fixed size) or dynamic.
/// </summary>
public sealed class PartialDimension
{
    /// <summary>
    /// Gets whether this dimension is static (has a fixed size).
    /// </summary>
    public bool IsStatic { get; }

    /// <summary>
    /// Gets the value of the dimension. Only valid if IsStatic is true.
    /// </summary>
    public long Value { get; }

    private PartialDimension(bool isStatic, long value)
    {
        IsStatic = isStatic;
        Value = value;
    }

    /// <summary>
    /// Creates a static dimension with a fixed size.
    /// </summary>
    /// <param name="value">The dimension size.</param>
    /// <returns>A static partial dimension.</returns>
    public static PartialDimension Static(long value)
    {
        if (value < 0)
            throw new ArgumentOutOfRangeException(nameof(value), "Static dimension value must be non-negative.");
        return new PartialDimension(true, value);
    }

    /// <summary>
    /// Creates a dynamic dimension (size determined at runtime).
    /// </summary>
    public static PartialDimension Dynamic { get; } = new PartialDimension(false, -1);

    /// <summary>
    /// Gets a string representation of the dimension.
    /// </summary>
    public override string ToString()
    {
        return IsStatic ? Value.ToString() : "?";
    }

    /// <summary>
    /// Implicitly converts a long to a static partial dimension.
    /// </summary>
    public static implicit operator PartialDimension(long value)
    {
        return Static(value);
    }
}
