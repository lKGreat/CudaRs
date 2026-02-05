namespace CudaRS.Examples;

using CudaRS.Yolo;

/// <summary>
/// 集中管理所有配置常量和字段
/// </summary>
static class Config
{
    // ========== 模式开关 ==========
    public const bool OnlyFluentBench = true;
    public const bool OnlyOpenVinoBench = false;
    public const bool OnlyOcr = false;
    public const bool UseLegacyMode = false;
    public const bool RunSequentialBenchmark = false;
    public const bool RunParallelBenchmark = false;
    public const bool RunOcrTest = true;
    public const bool UseOpenVinoForOnnx = true;
    public const bool DetailedOutput = true;
    public const bool RunOpenVinoAsyncQueueBench = true;
    public const bool RunAnnotationDemo = true;
    
    // ========== 画框示例配置 ==========
    public const string AnnotatedOutputDir = @"E:\codeding\AI\onnx\output";
    public const bool SaveAnnotatedImages = true;
    public const bool ShowErrorDetails = true;

    // ========== CUDA/TensorRT 路径 ==========
    public const string CudaRoot = @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6";
    public const string TensorRtRoot = @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6";

    // ========== 设备和线程 ==========
    public const int DeviceId = 0;
    public const int CpuThreads = 8;

    // ========== YOLO 通用配置 ==========
    public const int InputWidth = 640;
    public const int InputHeight = 640;
    public const int InputChannels = 3;
    public const float ConfidenceThreshold = 0.25f;
    public const float IouThreshold = 0.45f;
    public const int MaxDetections = 300;
    public const int MaxDetectionsToPrint = 5;

    // ========== Pipeline 配置 ==========
    public const int PipelineMaxInputWidth = 4096;
    public const int PipelineMaxInputHeight = 4096;
    public const int PipelineIterations = 10;

    // ========== Legacy 模式配置 ==========
    public const string LegacyEnginePath = @"E:\codeding\AI\onnx\best\best.engine";
    public const string LegacyImagePath = @"E:\codeding\AI\onnx\best\train_batch0.jpg";
    public const int LegacyWarmup = 0;
    public const int LegacyIterations = 1;
    public const int LegacyMaxDetections = 100;

    // ========== 模型路径配置 ==========
    public static readonly string[] EnginePaths = Array.Empty<string>();

    public static readonly string[] OnnxModelPaths =
    {
        @"E:\codeding\AI\onnx\best",
    };

    public static readonly string[] ImagePaths =
    {
        @"E:\codeding\AI\onnx\best",
    };

    public static readonly YoloVersion[] Versions =
    {
        YoloVersion.V8,
    };

    public static readonly YoloVersion[] OnnxVersions =
    {
        YoloVersion.V8,
    };

    public static readonly YoloTask[] Tasks =
    {
        YoloTask.Detect,
    };

    public static readonly YoloTask[] OnnxTasks =
    {
        YoloTask.Detect,
    };

    public static readonly string? LabelsPath = null;

    // ========== OCR 配置 ==========
    public const string OcrDetModelDir = @"E:\codeding\AI\PP-OCRv5_mobile_det_infer\PP-OCRv5_mobile_det_infer";
    public const string OcrRecModelDir = @"E:\codeding\AI\PP-OCRv5_mobile_det_infer\PP-OCRv5_mobile_rec_infer";
    public const string OcrImagePath = @"E:\codeding\AI\PaddleOCR-3.3.2\deploy\android_demo\app\src\main\assets\images\det_0.jpg";
    public const string OcrVersion = "PP-OCRv5";
    public const string OcrDetModelName = "PP-OCRv5_mobile_det";
    public const string OcrRecModelName = "PP-OCRv5_mobile_rec";

    // ========== OpenVINO 配置 ==========
    public const string OpenVinoDevice = "cpu";
    public const string OpenVinoConfigJson = "";
    public const int OpenVinoIterations = 10;
    public const int OpenVinoWarmupIterations = 0;

    // ========== OpenVINO 测试配置 ==========
    public const bool RunOpenVinoTests = true; // Set to true to run OpenVINO tests
    public const string TestYoloModel = @"E:\models\yolo11n.onnx";
    public const string TestImage1 = @"E:\images\test1.jpg";
    public const string TestImage2 = @"E:\images\test2.jpg";
    public const string TestImage3 = @"E:\images\test3.jpg";
    public const string TestImage4 = @"E:\images\test4.jpg";

    // OpenVINO OCR 模型路径
    public const string OpenVinoOcrDetModelPath = @"E:\codeding\AI\PP-OCRv5_mobile_det_infer\ppocrv5_det.onnx";
    public const string OpenVinoOcrRecModelPath = @"E:\codeding\AI\PP-OCRv5_mobile_det_infer\ppocrv5_rec.onnx";
    public const string OpenVinoOcrDictPath = @"E:\codeding\AI\PaddleOCR-3.3.2\ppocr\utils\dict\ppocrv5_dict.txt";

    // OpenVINO OCR 参数
    public const int OcrDetResizeLong = 960;
    public const int OcrDetStride = 128;
    public const float OcrDetThresh = 0.3f;
    public const float OcrDetBoxThresh = 0.6f;
    public const int OcrDetMaxCandidates = 1000;
    public const int OcrDetMinArea = 10;
    public const int OcrDetBoxPadding = 2;
    public const int OcrRecInputH = 48;
    public const int OcrRecInputW = 320;

    // OpenVINO 设备列表
    public static readonly string[] OpenVinoOcrDevices =
    {
        "cpu",
    };

    public static readonly string[] OpenVinoYoloDevices =
    {
        "cpu",
        "auto",
    };

    // OpenVINO 配置变体
    public static readonly (string Tag, string Json)[] OpenVinoConfigVariants =
    {
        ("default", ""),
        ("latency", "{\"PERFORMANCE_HINT\":\"LATENCY\"}"),
        ("throughput", "{\"PERFORMANCE_HINT\":\"THROUGHPUT\",\"NUM_STREAMS\":\"AUTO\"}"),
        ("latency-pinning", "{\"PERFORMANCE_HINT\":\"LATENCY\",\"INFERENCE_NUM_THREADS\":\"8\",\"AFFINITY\":\"CORE\",\"ENABLE_CPU_PINNING\":\"YES\"}"),
        ("throughput-queue", "{\"PERFORMANCE_HINT\":\"THROUGHPUT\",\"NUM_STREAMS\":\"AUTO\",\"NUM_REQUESTS\":\"4\",\"INFERENCE_NUM_THREADS\":\"8\"}"),
    };

    // ========== OpenVINO AsyncQueue 配置 ==========
    public const int OpenVinoAsyncQueueRequests = 4;
    public const int OpenVinoAsyncQueueIterations = 5;
    public const string OpenVinoAsyncQueueDevice = "cpu";
    public const string OpenVinoAsyncQueueConfigJson = "{\"PERFORMANCE_HINT\":\"THROUGHPUT\",\"NUM_STREAMS\":\"AUTO\"}";
    public const string OpenVinoAsyncQueuePerformanceMode = "throughput";
    public const string OpenVinoAsyncQueueCacheDir = "";
    public static readonly string? OpenVinoAsyncQueueDeviceName = null;
}
