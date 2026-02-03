namespace CudaRS.Yolo;

public sealed class YoloGpuThroughputOptions
{
    public int MaxConcurrency { get; set; } = 1;
    public YoloPipelineOptions PipelineOptions { get; set; } = new YoloPipelineOptions();
}
