using System;
using System.Threading;
using System.Threading.Tasks;
using CudaRS;
using CudaRS.Ocr;
using CudaRS.OpenVino;
using CudaRS.Yolo;

namespace CudaRS.Fluent;

/// <summary>
/// Fluent builder that enforces a single provider per pipeline and a unified run API.
/// Located in CudaRS.Yolo package to avoid circular references while still covering OCR/OpenVINO.
/// </summary>
public sealed class FluentPipelineBuilder
{
    private readonly ModelHub _hub = new();
    private FluentTaskKind _task = FluentTaskKind.Unknown;
    private ProviderKind _provider = ProviderKind.None;
    private string _modelId = "default";
    private string _pipelineId = "default";
    private YoloModelDefinition? _yoloDefinition;
    private OcrModelConfig? _ocrConfig;
    private OpenVinoOcrModelConfig? _ovOcrConfig;
    private OpenVinoPipelineConfig? _ovOcrPipelineConfig;
    private OpenVinoModelConfig? _ovModelConfig;
    private OpenVinoPipelineConfig? _ovPipelineConfig;
    private ThroughputOptions _throughput = new();
    private QueueOptions _queue = new();
    private int _deviceId;

    /// <summary>
    /// Select YOLO task and provide model configuration.
    /// </summary>
    public FluentPipelineBuilder ForYolo(string modelPath, Action<YoloConfig>? configure = null, int deviceId = 0)
    {
        var cfg = new YoloConfig();
        configure?.Invoke(cfg);

        _task = FluentTaskKind.Yolo;
        _modelId = string.IsNullOrWhiteSpace(modelPath) ? "yolo" : modelPath;
        _pipelineId = $"{_modelId}-pipeline";
        _deviceId = deviceId;
        _yoloDefinition = new YoloModelDefinition
        {
            ModelId = _modelId,
            ModelPath = modelPath,
            Config = cfg,
            DeviceId = deviceId,
        };
        return this;
    }

    /// <summary>
    /// Select OCR task and provide model configuration.
    /// </summary>
    public FluentPipelineBuilder ForOcr(Action<OcrModelConfig> configure)
    {
        if (configure == null) throw new ArgumentNullException(nameof(configure));
        var cfg = new OcrModelConfig();
        configure(cfg);

        _task = FluentTaskKind.Ocr;
        _modelId = "ocr";
        _pipelineId = "ocr-pipeline";
        _ocrConfig = cfg;
        return this;
    }

    /// <summary>
    /// Select OpenVINO OCR task and provide model configuration.
    /// </summary>
    public FluentPipelineBuilder ForOpenVinoOcr(Action<OpenVinoOcrModelConfig> configure, Action<OpenVinoPipelineConfig>? configurePipeline = null)
    {
        if (configure == null) throw new ArgumentNullException(nameof(configure));
        var cfg = new OpenVinoOcrModelConfig();
        configure(cfg);

        var pipeCfg = new OpenVinoPipelineConfig();
        configurePipeline?.Invoke(pipeCfg);

        _task = FluentTaskKind.Ocr;
        _modelId = "ov-ocr";
        _pipelineId = "ov-ocr-pipeline";
        _ovOcrConfig = cfg;
        _ovOcrPipelineConfig = pipeCfg;
        return this;
    }

    /// <summary>
    /// Select general tensor (OpenVINO) task.
    /// </summary>
    public FluentPipelineBuilder ForTensor(string modelPath, Action<OpenVinoModelConfig>? configureModel = null, Action<OpenVinoPipelineConfig>? configurePipeline = null)
    {
        var modelCfg = new OpenVinoModelConfig { ModelPath = modelPath };
        configureModel?.Invoke(modelCfg);

        var pipeCfg = new OpenVinoPipelineConfig();
        configurePipeline?.Invoke(pipeCfg);

        _task = FluentTaskKind.Tensor;
        _modelId = string.IsNullOrWhiteSpace(modelPath) ? "openvino" : modelPath;
        _pipelineId = $"{_modelId}-pipeline";
        _ovModelConfig = modelCfg;
        _ovPipelineConfig = pipeCfg;
        return this;
    }

    private void SetProvider(ProviderKind provider)
    {
        if (_provider != ProviderKind.None && _provider != provider)
            throw new InvalidOperationException("Only one provider can be selected per pipeline.");
        _provider = provider;
    }

    public FluentPipelineBuilder AsTensorRt()
    {
        SetProvider(ProviderKind.TensorRt);
        return this;
    }

    public FluentPipelineBuilder AsOnnx()
    {
        SetProvider(ProviderKind.Onnx);
        return this;
    }

    public FluentPipelineBuilder AsOpenVino()
    {
        SetProvider(ProviderKind.OpenVinoGpu);
        return this;
    }

    public FluentPipelineBuilder AsCpu()
    {
        SetProvider(ProviderKind.OpenVinoCpu);
        return this;
    }

    public FluentPipelineBuilder AsPaddle()
    {
        SetProvider(ProviderKind.Paddle);
        return this;
    }

    public FluentPipelineBuilder WithThroughput(Action<ThroughputOptions> configure)
    {
        configure?.Invoke(_throughput);
        return this;
    }

    public FluentPipelineBuilder WithQueue(Action<QueueOptions> configure)
    {
        configure?.Invoke(_queue);
        return this;
    }

    /// <summary>
    /// Build and return a typed pipeline with unified Run/RunAsync signatures.
    /// </summary>
    public object Build()
    {
        if (_task == FluentTaskKind.Unknown)
            throw new InvalidOperationException("Task not specified. Call ForYolo/ForOcr/ForTensor first.");
        if (_provider == ProviderKind.None)
            throw new InvalidOperationException("Provider not specified. Call AsTensorRt/AsOpenVino/AsCpu/AsPaddle.");

        return _task switch
        {
            FluentTaskKind.Yolo => BuildYoloInternal(),
            FluentTaskKind.Ocr => BuildOcrInternal(),
            FluentTaskKind.Tensor => BuildTensorInternal(),
            _ => throw new InvalidOperationException("Unsupported task."),
        };
    }

    public IFluentImagePipeline<ModelInferenceResult> BuildYolo()
    {
        _task = FluentTaskKind.Yolo;
        return BuildYoloInternal();
    }

    public IFluentYoloPipeline BuildYoloFluent()
    {
        _task = FluentTaskKind.Yolo;
        return BuildYoloFluentInternal();
    }

    public IFluentImagePipeline<OcrResult> BuildOcr()
    {
        _task = FluentTaskKind.Ocr;
        return BuildOcrInternal();
    }

    public IFluentTensorPipeline<OpenVinoTensorOutput[]> BuildTensor()
    {
        _task = FluentTaskKind.Tensor;
        return BuildTensorInternal();
    }

    private IFluentImagePipeline<ModelInferenceResult> BuildYoloInternal()
    {
        if (_yoloDefinition == null)
            throw new InvalidOperationException("YOLO model definition not provided.");

        _yoloDefinition.Config.Backend = _provider switch
        {
            ProviderKind.TensorRt => InferenceBackend.TensorRT,
            ProviderKind.Onnx => InferenceBackend.OnnxRuntime,
            ProviderKind.OpenVinoGpu => InferenceBackend.OpenVino,
            ProviderKind.OpenVinoCpu => InferenceBackend.OpenVino,
            _ => _yoloDefinition.Config.Backend,
        };

        var modelOptions = new ModelOptions
        {
            ModelId = _yoloDefinition.ModelId,
            Kind = ModelKind.Yolo,
            ConfigJson = YoloModelConfigJson.Build(_yoloDefinition),
        };

        var pipelineOptions = new PipelineOptions
        {
            PipelineId = _pipelineId,
            Kind = MapYoloPipelineKind(),
            ConfigJson = BuildYoloPipelineJson(),
        };

        modelOptions.Pipelines.Add(pipelineOptions);

        var model = _hub.LoadModel(modelOptions);
        var handle = _hub.CreatePipeline(model, pipelineOptions);
        var pipeline = new YoloPipeline(handle, _yoloDefinition.Config, _yoloDefinition.ModelId, ownsHandle: true);
        return new FluentYoloPipeline(pipeline);
    }

    private IFluentYoloPipeline BuildYoloFluentInternal()
    {
        if (_yoloDefinition == null)
            throw new InvalidOperationException("YOLO model definition not provided.");

        _yoloDefinition.Config.Backend = _provider switch
        {
            ProviderKind.TensorRt => InferenceBackend.TensorRT,
            ProviderKind.Onnx => InferenceBackend.OnnxRuntime,
            ProviderKind.OpenVinoGpu => InferenceBackend.OpenVino,
            ProviderKind.OpenVinoCpu => InferenceBackend.OpenVino,
            _ => _yoloDefinition.Config.Backend,
        };

        var modelOptions = new ModelOptions
        {
            ModelId = _yoloDefinition.ModelId,
            Kind = ModelKind.Yolo,
            ConfigJson = YoloModelConfigJson.Build(_yoloDefinition),
        };

        var pipelineOptions = new PipelineOptions
        {
            PipelineId = _pipelineId,
            Kind = MapYoloPipelineKind(),
            ConfigJson = BuildYoloPipelineJson(),
        };

        modelOptions.Pipelines.Add(pipelineOptions);

        var model = _hub.LoadModel(modelOptions);
        var handle = _hub.CreatePipeline(model, pipelineOptions);
        var pipeline = new YoloPipeline(handle, _yoloDefinition.Config, _yoloDefinition.ModelId, ownsHandle: true);
        return new FluentYoloPipelineWithAnnotation(pipeline);
    }

    private IFluentImagePipeline<OcrResult> BuildOcrInternal()
    {
        if (_provider == ProviderKind.Paddle)
        {
            if (_ocrConfig == null)
                throw new InvalidOperationException("OCR configuration not provided.");

            var modelOptions = new ModelOptions
            {
                ModelId = _modelId,
                Kind = ModelKind.PaddleOcr,
                ConfigJson = _ocrConfig.ToJson(),
            };

            var pipelineOptions = new PipelineOptions
            {
                PipelineId = _pipelineId,
                Kind = PipelineKind.PaddleOcr,
                ConfigJson = "{}",
            };

            modelOptions.Pipelines.Add(pipelineOptions);

            var model = _hub.LoadModel(modelOptions);
            var handle = _hub.CreatePipeline(model, pipelineOptions);
            var pipeline = new CudaRS.Ocr.OcrPipeline(handle);
            return new FluentOcrPipeline(pipeline);
        }

        if (_provider == ProviderKind.OpenVinoCpu || _provider == ProviderKind.OpenVinoGpu)
        {
            if (_ovOcrConfig == null)
                throw new InvalidOperationException("OpenVINO OCR configuration not provided.");

            var pipeCfg = _ovOcrPipelineConfig ?? new OpenVinoPipelineConfig();
            pipeCfg.OpenVinoDevice = _provider == ProviderKind.OpenVinoCpu ? "cpu" : "gpu";

            var modelOptions = new ModelOptions
            {
                ModelId = _modelId,
                Kind = ModelKind.OpenVinoOcr,
                ConfigJson = _ovOcrConfig.ToJson(),
            };

            var pipelineOptions = new PipelineOptions
            {
                PipelineId = _pipelineId,
                Kind = PipelineKind.OpenVinoOcr,
                ConfigJson = pipeCfg.ToJson(),
            };

            modelOptions.Pipelines.Add(pipelineOptions);

            var model = _hub.LoadModel(modelOptions);
            var handle = _hub.CreatePipeline(model, pipelineOptions);
            var pipeline = new CudaRS.Ocr.OcrPipeline(handle);
            return new FluentOcrPipeline(pipeline);
        }

        throw new InvalidOperationException("OCR currently supports Paddle/OpenVINO backend only. Use AsPaddle(), AsCpu(), or AsOpenVino().");
    }

    private IFluentTensorPipeline<OpenVinoTensorOutput[]> BuildTensorInternal()
    {
        if (_ovModelConfig == null || _ovPipelineConfig == null)
            throw new InvalidOperationException("OpenVINO model/pipeline config not provided.");
        if (_provider != ProviderKind.OpenVinoGpu && _provider != ProviderKind.OpenVinoCpu)
            throw new InvalidOperationException("Tensor pipeline supports OpenVINO CPU/GPU. Use AsOpenVino() or AsCpu().");

        _ovPipelineConfig.OpenVinoDevice = _provider == ProviderKind.OpenVinoCpu ? "cpu" : "auto";

        var modelOptions = new ModelOptions
        {
            ModelId = _modelId,
            Kind = ModelKind.OpenVino,
            ConfigJson = _ovModelConfig.ToJson(),
        };

        var pipelineOptions = new PipelineOptions
        {
            PipelineId = _pipelineId,
            Kind = PipelineKind.OpenVinoTensor,
            ConfigJson = _ovPipelineConfig.ToJson(),
        };

        modelOptions.Pipelines.Add(pipelineOptions);

        var model = _hub.LoadModel(modelOptions);
        var handle = _hub.CreatePipeline(model, pipelineOptions);
        var pipeline = new OpenVinoPipeline(handle);
        return new FluentTensorPipeline(pipeline);
    }

    private PipelineKind MapYoloPipelineKind()
    {
        return _provider switch
        {
            ProviderKind.TensorRt => PipelineKind.YoloGpuThroughput,
            ProviderKind.OpenVinoGpu => PipelineKind.YoloOpenVino,
            ProviderKind.OpenVinoCpu => PipelineKind.YoloCpu,
            _ => throw new InvalidOperationException("Selected provider not supported for YOLO."),
        };
    }

    private string BuildYoloPipelineJson()
    {
        var opts = new YoloPipelineOptions
        {
            Device = _provider == ProviderKind.OpenVinoCpu ? InferenceDevice.Cpu :
                     _provider == ProviderKind.OpenVinoGpu ? InferenceDevice.OpenVino :
                     InferenceDevice.Gpu,
            BatchSize = Math.Max(1, _throughput.BatchSize),
            MaxBatchDelayMs = Math.Max(0, _throughput.MaxBatchDelayMs),
            WorkerCount = Math.Max(1, _throughput.NumStreams),
            OpenVinoNumStreams = _throughput.NumStreams,
            OpenVinoDevice = _provider == ProviderKind.OpenVinoCpu ? "cpu" :
                             _provider == ProviderKind.OpenVinoGpu ? "gpu" : "auto",
            QueueCapacity = Math.Max(1, _queue.Capacity),
            QueueTimeoutMs = _queue.TimeoutMs,
            QueueBackpressure = _queue.Backpressure,
        };

        return opts.ToJson();
    }
}

/// <summary>
/// Unified image pipeline interface (encoded images).
/// </summary>
public interface IFluentImagePipeline<T>
{
    T Run(ReadOnlyMemory<byte> imageBytes);
    Task<T> RunAsync(ReadOnlyMemory<byte> imageBytes, CancellationToken cancellationToken = default);
}

/// <summary>
/// Unified tensor pipeline interface.
/// </summary>
public interface IFluentTensorPipeline<T>
{
    T Run(ReadOnlyMemory<float> input, ReadOnlyMemory<long> shape);
    Task<T> RunAsync(ReadOnlyMemory<float> input, ReadOnlyMemory<long> shape, CancellationToken cancellationToken = default);
}

internal sealed class FluentYoloPipeline : IFluentImagePipeline<ModelInferenceResult>, IDisposable
{
    private readonly YoloPipeline _pipeline;

    public FluentYoloPipeline(YoloPipeline pipeline)
    {
        _pipeline = pipeline ?? throw new ArgumentNullException(nameof(pipeline));
    }

    public ModelInferenceResult Run(ReadOnlyMemory<byte> imageBytes)
        => _pipeline.Run(imageBytes, channelId: "default");

    public Task<ModelInferenceResult> RunAsync(ReadOnlyMemory<byte> imageBytes, CancellationToken cancellationToken = default)
        => Task.Run(() => Run(imageBytes), cancellationToken);

    public void Dispose() => _pipeline.Dispose();
}

internal sealed class FluentOcrPipeline : IFluentImagePipeline<OcrResult>, IDisposable
{
    private readonly CudaRS.Ocr.OcrPipeline _pipeline;

    public FluentOcrPipeline(CudaRS.Ocr.OcrPipeline pipeline)
    {
        _pipeline = pipeline ?? throw new ArgumentNullException(nameof(pipeline));
    }

    public OcrResult Run(ReadOnlyMemory<byte> imageBytes)
        => _pipeline.RunImage(imageBytes.ToArray());

    public Task<OcrResult> RunAsync(ReadOnlyMemory<byte> imageBytes, CancellationToken cancellationToken = default)
        => Task.Run(() => Run(imageBytes), cancellationToken);

    public void Dispose() => _pipeline.Dispose();
}

internal sealed class FluentTensorPipeline : IFluentTensorPipeline<OpenVinoTensorOutput[]>, IDisposable
{
    private readonly OpenVinoPipeline _pipeline;

    public FluentTensorPipeline(OpenVinoPipeline pipeline)
    {
        _pipeline = pipeline ?? throw new ArgumentNullException(nameof(pipeline));
    }

    public OpenVinoTensorOutput[] Run(ReadOnlyMemory<float> input, ReadOnlyMemory<long> shape)
        => _pipeline.Run(input, shape);

    public Task<OpenVinoTensorOutput[]> RunAsync(ReadOnlyMemory<float> input, ReadOnlyMemory<long> shape, CancellationToken cancellationToken = default)
        => Task.Run(() => Run(input, shape), cancellationToken);

    public void Dispose() => _pipeline.Dispose();
}

internal sealed class FluentYoloPipelineWithAnnotation : IFluentYoloPipeline
{
    private readonly YoloPipeline _pipeline;

    public FluentYoloPipelineWithAnnotation(YoloPipeline pipeline)
    {
        _pipeline = pipeline ?? throw new ArgumentNullException(nameof(pipeline));
    }

    public FluentResultWrapper Run(ReadOnlyMemory<byte> imageBytes)
    {
        var result = _pipeline.Run(imageBytes, channelId: "default");
        return new FluentResultWrapper(imageBytes, result);
    }

    public Task<FluentResultWrapper> RunAsync(ReadOnlyMemory<byte> imageBytes, CancellationToken cancellationToken = default)
        => Task.Run(() => Run(imageBytes), cancellationToken);

    public void Dispose() => _pipeline.Dispose();
}
