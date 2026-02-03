using System;

namespace CudaRS.Yolo;

public sealed class YoloModelBuilder
{
    private readonly YoloConfig _config;

    public YoloModelBuilder(YoloConfig config)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
    }

    public YoloModelBuilder WithVersion(YoloVersion version)
    {
        _config.Version = version;
        return this;
    }

    public YoloModelBuilder WithTask(YoloTask task)
    {
        _config.Task = task;
        return this;
    }

    public YoloModelBuilder WithBackend(InferenceBackend backend)
    {
        _config.Backend = backend;
        return this;
    }

    public YoloModelBuilder WithInput(int width, int height, int channels = 3)
    {
        _config.InputWidth = width;
        _config.InputHeight = height;
        _config.InputChannels = channels;
        return this;
    }

    public YoloModelBuilder WithThresholds(float confidence, float iou)
    {
        _config.ConfidenceThreshold = confidence;
        _config.IouThreshold = iou;
        return this;
    }

    public YoloModelBuilder WithMaxDetections(int maxDetections)
    {
        _config.MaxDetections = maxDetections;
        return this;
    }

    public YoloModelBuilder WithClassNames(string[] classNames)
    {
        _config.ClassNames = classNames ?? Array.Empty<string>();
        return this;
    }

    public YoloModelBuilder WithAnchors(float[][] anchors, int[] strides)
    {
        _config.Anchors = anchors ?? Array.Empty<float[]>();
        _config.Strides = strides ?? Array.Empty<int>();
        return this;
    }

    public YoloModelBuilder WithAnchorFree(bool anchorFree)
    {
        _config.AnchorFree = anchorFree;
        return this;
    }

    public YoloModelBuilder WithMaskProto(int width, int height, int channels)
    {
        _config.MaskProtoWidth = width;
        _config.MaskProtoHeight = height;
        _config.MaskProtoChannels = channels;
        return this;
    }

    public YoloModelBuilder WithKeypoints(int count)
    {
        _config.KeypointCount = count;
        return this;
    }

    public YoloModelBuilder WithAngleInRadians(bool angleInRadians)
    {
        _config.AngleInRadians = angleInRadians;
        return this;
    }
}
