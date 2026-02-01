using System;
using CudaRS;

namespace CudaRS.Yolo;

public static class YoloPipelineExtensions
{
    public static PipelineBuilder WithYolo(this PipelineBuilder builder, string modelId, string modelPath, Action<YoloModelBuilder>? configure = null)
    {
        if (builder == null)
            throw new ArgumentNullException(nameof(builder));
        if (string.IsNullOrWhiteSpace(modelId))
            throw new ArgumentException("ModelId is required.", nameof(modelId));
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("ModelPath is required.", nameof(modelPath));

        var cfgBuilder = new YoloConfigBuilder();
        configure?.Invoke(new YoloModelBuilder(cfgBuilder));
        var cfg = cfgBuilder.Build();

        builder.WithModel(modelId, m =>
        {
            m.FromPath(modelPath);
            m.WithBackend(cfg.Backend.ToString().ToLowerInvariant());
        });

        YoloModelRegistry.Register(new YoloModelDefinition
        {
            ModelId = modelId,
            ModelPath = modelPath,
            Config = cfg,
        });

        return builder;
    }
}
