using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using CudaRS;

namespace CudaRS.Yolo;

public sealed class YoloPipelineRunner
{
    private readonly PipelineRun _pipeline;

    public YoloPipelineRunner(PipelineRun pipeline)
    {
        _pipeline = pipeline ?? throw new ArgumentNullException(nameof(pipeline));
    }

    public MultiModelInferenceResult Run(string channelId, YoloImage image, long frameIndex = 0)
    {
        var sw = Stopwatch.StartNew();
        var modelResults = new Dictionary<string, ModelInferenceResult>(StringComparer.OrdinalIgnoreCase);

        foreach (var model in _pipeline.Definition.Models.Values)
        {
            if (!YoloModelRegistry.TryGet(model.ModelId, out var definition))
                continue;

            var deviceId = model.DeviceId ?? 0;
            var yoloModel = YoloModel.Create(new YoloModelDefinition
            {
                ModelId = definition.ModelId,
                ModelPath = definition.ModelPath,
                Config = definition.Config,
                DeviceId = deviceId,
            });
            var result = yoloModel.Run(channelId, image, frameIndex);
            modelResults[model.ModelId] = result;
            yoloModel.Dispose();
        }

        sw.Stop();

        return new MultiModelInferenceResult
        {
            ChannelId = channelId,
            FrameIndex = frameIndex,
            TotalMs = sw.Elapsed.TotalMilliseconds,
            ModelResults = modelResults,
        };
    }
}
