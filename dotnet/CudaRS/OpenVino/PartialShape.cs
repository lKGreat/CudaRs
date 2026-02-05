using System;
using System.Linq;

namespace CudaRS.OpenVino;

/// <summary>
/// Represents a partial shape with static and/or dynamic dimensions.
/// </summary>
public sealed class PartialShape
{
    /// <summary>
    /// Gets the dimensions of the shape.
    /// </summary>
    public PartialDimension[] Dimensions { get; }

    private PartialShape(PartialDimension[] dimensions)
    {
        Dimensions = dimensions ?? throw new ArgumentNullException(nameof(dimensions));
    }

    /// <summary>
    /// Creates a partial shape from an array of dimensions.
    /// </summary>
    /// <param name="dimensions">The dimensions of the shape.</param>
    /// <returns>A partial shape.</returns>
    public static PartialShape Create(params PartialDimension[] dimensions)
    {
        if (dimensions == null || dimensions.Length == 0)
            throw new ArgumentException("Dimensions cannot be null or empty.", nameof(dimensions));
        return new PartialShape(dimensions);
    }

    /// <summary>
    /// Creates a partial shape from an array of long values (all static dimensions).
    /// </summary>
    /// <param name="shape">The shape dimensions.</param>
    /// <returns>A partial shape with all static dimensions.</returns>
    public static PartialShape FromStaticShape(params long[] shape)
    {
        if (shape == null || shape.Length == 0)
            throw new ArgumentException("Shape cannot be null or empty.", nameof(shape));
        return new PartialShape(shape.Select(PartialDimension.Static).ToArray());
    }

    /// <summary>
    /// Creates a partial shape from an array of long values, where -1 indicates a dynamic dimension.
    /// </summary>
    /// <param name="shape">The shape dimensions (-1 for dynamic).</param>
    /// <returns>A partial shape.</returns>
    public static PartialShape FromArray(params long[] shape)
    {
        if (shape == null || shape.Length == 0)
            throw new ArgumentException("Shape cannot be null or empty.", nameof(shape));
        
        var dimensions = shape.Select(d => d == -1 ? PartialDimension.Dynamic : PartialDimension.Static(d)).ToArray();
        return new PartialShape(dimensions);
    }

    /// <summary>
    /// Gets the rank (number of dimensions) of the shape.
    /// </summary>
    public int Rank => Dimensions.Length;

    /// <summary>
    /// Gets whether all dimensions are static.
    /// </summary>
    public bool IsStatic => Dimensions.All(d => d.IsStatic);

    /// <summary>
    /// Gets whether any dimension is dynamic.
    /// </summary>
    public bool IsDynamic => Dimensions.Any(d => !d.IsStatic);

    /// <summary>
    /// Gets a string representation of the shape.
    /// </summary>
    public override string ToString()
    {
        return $"[{string.Join(", ", Dimensions.Select(d => d.ToString()))}]";
    }
}
