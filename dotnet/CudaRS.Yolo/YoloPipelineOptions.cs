using System.Text.Json;
using System.Text.Json.Serialization;

namespace CudaRS.Yolo;

public sealed class YoloPipelineOptions
{
    [JsonIgnore]
    public InferenceDevice Device { get; set; } = InferenceDevice.Gpu;

    [JsonPropertyName("max_input_width")]
    public int MaxInputWidth { get; set; } = 4096;

    [JsonPropertyName("max_input_height")]
    public int MaxInputHeight { get; set; } = 4096;

    [JsonPropertyName("batch_size")]
    public int BatchSize { get; set; } = 1;

    [JsonPropertyName("max_batch_delay_ms")]
    public int MaxBatchDelayMs { get; set; } = 2;

    [JsonPropertyName("allow_partial_batch")]
    public bool AllowPartialBatch { get; set; } = true;

    [JsonPropertyName("worker_count")]
    public int WorkerCount { get; set; } = 1;

    [JsonPropertyName("cpu_threads")]
    public int CpuThreads { get; set; } = 1;

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

    [JsonPropertyName("queue_capacity")]
    public int QueueCapacity { get; set; } = 32;

    [JsonPropertyName("queue_timeout_ms")]
    public int QueueTimeoutMs { get; set; } = -1;

    [JsonPropertyName("queue_backpressure")]
    public bool QueueBackpressure { get; set; } = true;

    public string ToJson()
    {
        return JsonSerializer.Serialize(this);
    }
}
