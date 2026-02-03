using System;
using CudaRS;

namespace CudaRS.Yolo;

public static class YoloPipelineExtensions
{
    public static PipelineBuilder WithYolo(
        this PipelineBuilder builder,
        string modelId,
        string modelPath,
        Action<YoloModelBuilder>? configure = null,
        Action<YoloPipelineOptions>? configurePipeline = null)
    {
        if (builder == null)
            throw new ArgumentNullException(nameof(builder));
        if (string.IsNullOrWhiteSpace(modelId))
            throw new ArgumentException("ModelId is required.", nameof(modelId));
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("ModelPath is required.", nameof(modelPath));

        var config = new YoloConfig();
        configure?.Invoke(new YoloModelBuilder(config));
        YoloVersionAdapter.ApplyVersionDefaults(config);

        var pipelineOptions = new YoloPipelineOptions();
        configurePipeline?.Invoke(pipelineOptions);

        var definition = new YoloModelDefinition
        {
            ModelId = modelId,
            ModelPath = modelPath,
            Config = config,
        };

        builder.WithModel(modelId, m =>
        {
            m.WithKind(ModelKind.Yolo)
                .WithConfigJson(YoloModelConfigJson.Build(definition))
                .WithPipeline("default", PipelineKind.YoloGpuThroughput, pipelineOptions.ToJson());
        });

        YoloModelRegistry.Register(definition);
        return builder;
    }
}
