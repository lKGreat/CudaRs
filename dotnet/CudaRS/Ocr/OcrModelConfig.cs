using System.Text.Json;
using System.Text.Json.Serialization;

namespace CudaRS.Ocr;

public sealed class OcrModelConfig
{
    [JsonPropertyName("det_model_dir")]
    public string DetModelDir { get; set; } = string.Empty;

    [JsonPropertyName("rec_model_dir")]
    public string RecModelDir { get; set; } = string.Empty;

    [JsonPropertyName("doc_orientation_model_name")]
    public string? DocOrientationModelName { get; set; }

    [JsonPropertyName("doc_orientation_model_dir")]
    public string? DocOrientationModelDir { get; set; }

    [JsonPropertyName("doc_unwarping_model_name")]
    public string? DocUnwarpingModelName { get; set; }

    [JsonPropertyName("doc_unwarping_model_dir")]
    public string? DocUnwarpingModelDir { get; set; }

    [JsonPropertyName("text_detection_model_name")]
    public string? TextDetectionModelName { get; set; }

    [JsonPropertyName("text_recognition_model_name")]
    public string? TextRecognitionModelName { get; set; }

    [JsonPropertyName("textline_orientation_model_name")]
    public string? TextlineOrientationModelName { get; set; }

    [JsonPropertyName("textline_orientation_model_dir")]
    public string? TextlineOrientationModelDir { get; set; }

    [JsonPropertyName("textline_orientation_batch_size")]
    public int? TextlineOrientationBatchSize { get; set; }

    [JsonPropertyName("text_recognition_batch_size")]
    public int? TextRecognitionBatchSize { get; set; }

    [JsonPropertyName("use_doc_orientation_classify")]
    public bool? UseDocOrientationClassify { get; set; } = false;

    [JsonPropertyName("use_doc_unwarping")]
    public bool? UseDocUnwarping { get; set; } = false;

    [JsonPropertyName("use_textline_orientation")]
    public bool? UseTextlineOrientation { get; set; } = false;

    [JsonPropertyName("text_det_limit_side_len")]
    public int? TextDetLimitSideLen { get; set; }

    [JsonPropertyName("text_det_max_side_limit")]
    public int? TextDetMaxSideLimit { get; set; }

    [JsonPropertyName("text_det_limit_type")]
    public string? TextDetLimitType { get; set; }

    [JsonPropertyName("text_det_thresh")]
    public float? TextDetThresh { get; set; }

    [JsonPropertyName("text_det_box_thresh")]
    public float? TextDetBoxThresh { get; set; }

    [JsonPropertyName("text_det_unclip_ratio")]
    public float? TextDetUnclipRatio { get; set; }

    [JsonPropertyName("text_rec_score_thresh")]
    public float? TextRecScoreThresh { get; set; }

    [JsonPropertyName("text_det_input_shape")]
    public int[]? TextDetInputShape { get; set; }

    [JsonPropertyName("text_rec_input_shape")]
    public int[]? TextRecInputShape { get; set; }

    [JsonPropertyName("lang")]
    public string? Lang { get; set; }

    [JsonPropertyName("ocr_version")]
    public string? OcrVersion { get; set; }

    [JsonPropertyName("vis_font_dir")]
    public string? VisFontDir { get; set; }

    [JsonPropertyName("device")]
    public string? Device { get; set; } = "cpu";

    [JsonPropertyName("precision")]
    public string? Precision { get; set; } = "fp32";

    [JsonPropertyName("enable_mkldnn")]
    public bool? EnableMkldnn { get; set; } = true;

    [JsonPropertyName("mkldnn_cache_capacity")]
    public int? MkldnnCacheCapacity { get; set; } = 10;

    [JsonPropertyName("cpu_threads")]
    public int? CpuThreads { get; set; } = 8;

    [JsonPropertyName("thread_num")]
    public int? ThreadNum { get; set; } = 1;

    [JsonPropertyName("paddlex_config_yaml")]
    public string? PaddlexConfigYaml { get; set; }

    [JsonPropertyName("openvino_config_json")]
    public string? OpenVinoConfigJson { get; set; }

    public string ToJson()
        => JsonSerializer.Serialize(this, new JsonSerializerOptions { DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull });
}
