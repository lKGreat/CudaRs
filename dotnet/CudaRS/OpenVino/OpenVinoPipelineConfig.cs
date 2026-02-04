using System.Text.Json;
using System.Text.Json.Serialization;

namespace CudaRS.OpenVino;

public sealed class OpenVinoPipelineConfig
{
    [JsonPropertyName("openvino_device")]
    public string OpenVinoDevice { get; set; } = "auto";

    [JsonPropertyName("openvino_config_json")]
    public string OpenVinoConfigJson { get; set; } = string.Empty;

    [JsonPropertyName("openvino_performance_mode")]
    public string OpenVinoPerformanceMode { get; set; } = string.Empty;

    [JsonPropertyName("openvino_num_requests")]
    public int OpenVinoNumRequests { get; set; }

    [JsonPropertyName("openvino_num_streams")]
    public int OpenVinoNumStreams { get; set; }

    [JsonPropertyName("openvino_enable_profiling")]
    public bool OpenVinoEnableProfiling { get; set; }

    [JsonPropertyName("openvino_cache_dir")]
    public string OpenVinoCacheDir { get; set; } = string.Empty;

    [JsonPropertyName("openvino_enable_mmap")]
    public bool? OpenVinoEnableMmap { get; set; }

    // Queue/backpressure (用于高吞吐，占位字段，后端选择性支持)
    [JsonPropertyName("queue_capacity")]
    public int? QueueCapacity { get; set; }

    [JsonPropertyName("queue_timeout_ms")]
    public int? QueueTimeoutMs { get; set; }

    [JsonPropertyName("queue_backpressure")]
    public bool? QueueBackpressure { get; set; }

    public string ToJson()
        => JsonSerializer.Serialize(this, new JsonSerializerOptions { DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull });
}
