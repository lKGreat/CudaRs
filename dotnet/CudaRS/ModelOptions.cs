using System.Collections.Generic;

namespace CudaRS;

public sealed class ModelOptions
{
    public string ModelId { get; set; } = "default";
    public ModelKind Kind { get; set; } = ModelKind.Unknown;
    public string ConfigJson { get; set; } = "{}";
    public IList<PipelineOptions> Pipelines { get; } = new List<PipelineOptions>();
}
