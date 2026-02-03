using System.Text.Json.Serialization;

namespace CudaRS.Yolo;

internal sealed class YoloModelConfigDto
{
    [JsonPropertyName("model_path")]
    public string ModelPath { get; set; } = string.Empty;

    [JsonPropertyName("device_id")]
    public int DeviceId { get; set; }

    [JsonPropertyName("input_width")]
    public int InputWidth { get; set; }

    [JsonPropertyName("input_height")]
    public int InputHeight { get; set; }

    [JsonPropertyName("input_channels")]
    public int InputChannels { get; set; }

    [JsonPropertyName("backend")]
    public string Backend { get; set; } = "tensorrt";
}
