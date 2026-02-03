namespace CudaRS.Yolo;

public sealed class YoloModelDefinition
{
    public string ModelId { get; init; } = string.Empty;
    public string ModelPath { get; init; } = string.Empty;
    public YoloConfig Config { get; init; } = new YoloConfig();
    public int DeviceId { get; init; }
}
