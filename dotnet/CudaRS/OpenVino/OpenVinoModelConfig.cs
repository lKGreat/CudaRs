using System.Text.Json;
using System.Text.Json.Serialization;

namespace CudaRS.OpenVino;

public sealed class OpenVinoModelConfig
{
    [JsonPropertyName("model_path")]
    public string ModelPath { get; set; } = string.Empty;

    [JsonPropertyName("enable_profiling")]
    public bool EnableProfiling { get; set; } = false;

    [JsonPropertyName("precision")]
    public ModelPrecision Precision { get; set; } = ModelPrecision.Auto;

    public string ToJson()
        => JsonSerializer.Serialize(this, new JsonSerializerOptions { DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull });
}
