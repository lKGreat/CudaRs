using System;
using CudaRS;

namespace CudaRS.Yolo;

public static class YoloPipelineRunExtensions
{
    public static MultiModelInferenceResult RunYolo(this PipelineRun pipeline, string channelId, YoloImage image, long frameIndex = 0)
    {
        if (pipeline == null)
            throw new ArgumentNullException(nameof(pipeline));

        var runner = new YoloPipelineRunner(pipeline);
        return runner.Run(channelId, image, frameIndex);
    }
}
