using System.Text.Json;
using System.Text.Json.Serialization;

namespace CudaRS.OpenVino;

public sealed class OpenVinoPipelineConfig
{
    [JsonPropertyName("openvino_device")]
    public string OpenVinoDevice { get; set; } = "auto";

    [JsonPropertyName("openvino_config_json")]
    public string OpenVinoConfigJson { get; set; } = string.Empty;

    public string ToJson()
        => JsonSerializer.Serialize(this, new JsonSerializerOptions { DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull });
}
