using System.Text.Json;
using System.Text.Json.Serialization;

namespace CudaRS.Ocr;

public sealed class OpenVinoOcrModelConfig
{
    [JsonPropertyName("det_model_path")]
    public string DetModelPath { get; set; } = string.Empty;

    [JsonPropertyName("rec_model_path")]
    public string RecModelPath { get; set; } = string.Empty;

    [JsonPropertyName("dict_path")]
    public string DictPath { get; set; } = string.Empty;

    [JsonPropertyName("device")]
    public string? Device { get; set; } = "cpu";

    [JsonPropertyName("det_resize_long")]
    public int? DetResizeLong { get; set; } = 960;

    [JsonPropertyName("det_stride")]
    public int? DetStride { get; set; } = 128;

    [JsonPropertyName("det_thresh")]
    public float? DetThresh { get; set; } = 0.3f;

    [JsonPropertyName("det_box_thresh")]
    public float? DetBoxThresh { get; set; } = 0.6f;

    [JsonPropertyName("det_unclip_ratio")]
    public float? DetUnclipRatio { get; set; } = 1.5f;

    [JsonPropertyName("det_max_candidates")]
    public int? DetMaxCandidates { get; set; } = 1000;

    [JsonPropertyName("det_min_area")]
    public int? DetMinArea { get; set; } = 10;

    [JsonPropertyName("det_box_padding")]
    public int? DetBoxPadding { get; set; } = 2;

    [JsonPropertyName("rec_input_h")]
    public int? RecInputH { get; set; } = 48;

    [JsonPropertyName("rec_input_w")]
    public int? RecInputW { get; set; } = 320;

    [JsonPropertyName("rec_batch_size")]
    public int? RecBatchSize { get; set; } = 8;

    [JsonPropertyName("rec_score_thresh")]
    public float? RecScoreThresh { get; set; } = 0.0f;

    public string ToJson()
        => JsonSerializer.Serialize(this, new JsonSerializerOptions { DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull });
}
