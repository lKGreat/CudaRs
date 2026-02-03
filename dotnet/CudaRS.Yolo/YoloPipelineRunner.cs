using System;
using System.Collections.Generic;
using System.Diagnostics;
using CudaRS;

namespace CudaRS.Yolo;

public sealed class YoloPipelineRunner
{
    private readonly PipelineRun _pipeline;

    public YoloPipelineRunner(PipelineRun pipeline)
    {
        _pipeline = pipeline ?? throw new ArgumentNullException(nameof(pipeline));
    }

    public MultiModelInferenceResult Run(string channelId, ReadOnlyMemory<byte> imageBytes, long frameIndex = 0)
    {
        var sw = Stopwatch.StartNew();
        var modelResults = new Dictionary<string, ModelInferenceResult>(StringComparer.OrdinalIgnoreCase);

        var pipelineId = "default";
        if (_pipeline.Definition.Channels.TryGetValue(channelId, out var channel))
            pipelineId = channel.PipelineId;

        foreach (var model in _pipeline.Definition.Models.Values)
        {
            if (!YoloModelRegistry.TryGet(model.ModelId, out var definition))
                continue;

            var handle = _pipeline.GetPipeline(model.ModelId, pipelineId);
            using var yoloPipeline = new YoloPipeline(handle, definition.Config, definition.ModelId, ownsHandle: false);
            var result = yoloPipeline.Run(imageBytes, channelId, frameIndex);
            modelResults[model.ModelId] = result;
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
