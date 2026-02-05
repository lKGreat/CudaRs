namespace CudaRS.Examples;

using CudaRS.Yolo;

/// <summary>
/// 集中管理所有配置常量和字段
/// </summary>
static class Config
{
    // ========== 模式开关 ==========
    public static readonly bool OnlyFluentBench = false;
    public static readonly bool OnlyOpenVinoBench = false;
    public static readonly bool OnlyOcr = false;
    public static readonly bool OnlyOpenVinoOcr = true;
    public static readonly bool UseLegacyMode = false;
    public static readonly bool RunSequentialBenchmark = false;
    public static readonly bool RunParallelBenchmark = false;
    public static readonly bool RunOcrTest = true;
    public static readonly bool UseOpenVinoForOnnx = true;
    public static readonly bool DetailedOutput = true;
    public static readonly bool RunOpenVinoAsyncQueueBench = true;
    public static readonly bool RunAnnotationDemo = false;
    public static readonly bool RunBackendSmoke = true;
    public static readonly bool RunPaddleOcrSmoke = false;
    public static readonly bool RunOpenVinoOcr = true;
    public static readonly bool OpenVinoOcrAutoConvert = false;
    public static readonly bool OpenVinoOcrForceReconvert = false;
    public static readonly bool OpenVinoOcrCompressToFp16 = false;
    
    // ========== 画框示例配置 ==========
    public const string AnnotatedOutputDir = @"E:\codeding\AI\onnx\output";
    public static readonly bool SaveAnnotatedImages = true;
    public static readonly bool ShowErrorDetails = true;

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
    public const int SmokeWarmupIterations = 3;
    public const int SmokeIterations = 10;

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
    public const string OcrImagePath = @"E:\codeding\AI\onnx\best\train_batch0.jpg";
    public const string OcrVersion = "PP-OCRv5";
    public const string OcrDetModelName = "PP-OCRv5_mobile_det";
    public const string OcrRecModelName = "PP-OCRv5_mobile_rec";

    // ========== OpenVINO 配置 ==========
    public const string OpenVinoDevice = "cpu";
    public const string OpenVinoConfigJson = "";
    public const int OpenVinoIterations = 10;
    public const int OpenVinoWarmupIterations = 0;
    public const int OpenVinoOcrIterations = 10;
    public const int OpenVinoOcrWarmupIterations = 0;
    public const int OpenVinoOcrRecBatchSize = 8;
    public const string OpenVinoIrCacheDir = "";

    // ========== OpenVINO 测试配置 ==========
    public static readonly bool RunOpenVinoTests = false; // Set to true to run OpenVINO tests (requires OpenVINO installed)
    public const string TestYoloModel = @"E:\codeding\AI\onnx\best\best.onnx";
    public const string TestImage1 = @"E:\codeding\AI\onnx\best\train_batch0.jpg";
    public const string TestImage2 = @"E:\codeding\AI\onnx\best\train_batch0.jpg";
    public const string TestImage3 = @"E:\codeding\AI\onnx\best\train_batch0.jpg";
    public const string TestImage4 = @"E:\codeding\AI\onnx\best\train_batch0.jpg";

    // OpenVINO OCR 模型路径（自动转换时使用 Paddle 模型目录）
    // OpenVINO OCR model paths (IR .xml/.bin when auto-convert is off)
    public const string OpenVinoOcrDetModelPath = @"E:\codeding\AI\PP-OCRv4\openvino\FP32\ch_PP-OCRv4_det.xml";
    public const string OpenVinoOcrRecModelPath = @"E:\codeding\AI\PP-OCRv4\openvino\FP32\ch_PP-OCRv4_rec.xml";
    public const string OpenVinoOcrDictPath = @"E:\codeding\AI\PaddleOCR-3.3.2\ppocr\utils\dict\ppocrv4_doc_dict.txt";

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

    // OpenVINO OCR 图像列表（为空则回退到 OcrImagePath）
    public static readonly string[] OpenVinoOcrImagePaths =
    {
        @"E:\codeding\AI\PaddleOCR-3.3.2\docs\images\00006737.jpg",
        @"E:\codeding\AI\PaddleOCR-3.3.2\docs\images\185310636-6ce02f7c-790d-479f-b163-ea97a5a04808-20240708082238739.jpg",
        @"E:\codeding\AI\PaddleOCR-3.3.2\docs\images\185393805-c67ff571-cf7e-4217-a4b0-8b396c4f22bb-20240708082310650.jpg",
        @"E:\codeding\AI\PaddleOCR-3.3.2\docs\images\185539517-ccf2372a-f026-4a7c-ad28-c741c770f60a-20240708082247529.png",
        @"E:\codeding\AI\PaddleOCR-3.3.2\docs\images\185540080-0431e006-9235-4b6d-b63d-0b3c6e1de48f-20240708082316558.jpg",
        @"E:\codeding\AI\PaddleOCR-3.3.2\docs\images\186094813-3a8e16cc-42e5-4982-b9f4-0134dfb5688d-20240708082323916.png",
        @"E:\codeding\AI\PaddleOCR-3.3.2\docs\images\197464552-69de557f-edff-4c7f-acbf-069df1ba097f-20240708082253634.png",
        @"E:\codeding\AI\PaddleOCR-3.3.2\docs\images\Banner.png",
        @"E:\codeding\AI\PaddleOCR-3.3.2\docs\images\Banner_cn.png",
        @"E:\codeding\AI\PaddleOCR-3.3.2\docs\images\Banner_ja.png",
        @"E:\codeding\AI\PaddleOCR-3.3.2\docs\images\en_1.png",
        @"E:\codeding\AI\PaddleOCR-3.3.2\docs\images\en_2.png",
        @"E:\codeding\AI\PaddleOCR-3.3.2\docs\images\en_3-0398013.png",
        @"E:\codeding\AI\PaddleOCR-3.3.2\docs\images\en_3.png",
        @"E:\codeding\AI\PaddleOCR-3.3.2\docs\images\japan_2.jpg",
        @"E:\codeding\AI\PaddleOCR-3.3.2\docs\images\korean_1.jpg",
        @"E:\codeding\AI\PaddleOCR-3.3.2\docs\images\PP-OCRv3-pic001.jpg",
        @"E:\codeding\AI\PaddleOCR-3.3.2\docs\images\PP-OCRv3-pic002.jpg",
        @"E:\codeding\AI\PaddleOCR-3.3.2\docs\images\PP-OCRv3-pic003.jpg",
        @"E:\codeding\AI\PaddleOCR-3.3.2\docs\images\ppocrv4.png",
        @"E:\codeding\AI\PaddleOCR-3.3.2\docs\images\ppocrv4_en.jpg",
        @"E:\codeding\AI\PaddleOCR-3.3.2\docs\images\ppstructure-20240708082235651.gif",
        @"E:\codeding\AI\PaddleOCR-3.3.2\docs\images\test_add_91.jpg",
    };
}
