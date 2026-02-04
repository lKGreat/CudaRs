using System.Text.Json;
using System.Text.Json.Serialization;

namespace CudaRS.Ocr;

public sealed class OcrPipelineConfig
{
    [JsonPropertyName("worker_count")]
    public int WorkerCount { get; set; } = 1;

    [JsonPropertyName("enable_struct_json")]
    public bool EnableStructJson { get; set; }

    public string ToJson()
        => JsonSerializer.Serialize(this, new JsonSerializerOptions { DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull });
}
