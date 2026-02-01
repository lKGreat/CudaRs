using System;

namespace CudaRS;

/// <summary>
/// Entry point for building a fluent inference pipeline.
/// </summary>
public static class InferencePipeline
{
    /// <summary>
    /// Create a new pipeline builder with default settings.
    /// </summary>
    public static PipelineBuilder Create() => new PipelineBuilder();
}
