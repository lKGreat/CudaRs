using System.Text.Json;
using System.Text.Json.Serialization;

namespace CudaRS.Yolo;

public sealed class YoloPipelineOptions
{
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

    public string ToJson()
    {
        return JsonSerializer.Serialize(this);
    }
}
