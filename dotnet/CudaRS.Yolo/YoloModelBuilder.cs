using System;

namespace CudaRS.Yolo;

public sealed class YoloModelBuilder
{
    private readonly YoloConfigBuilder _builder;

    internal YoloModelBuilder(YoloConfigBuilder builder)
    {
        _builder = builder;
    }

    public YoloModelBuilder Version(YoloVersion version)
    {
        _builder.Version = version;
        return this;
    }

    public YoloModelBuilder Task(YoloTask task)
    {
        _builder.Task = task;
        return this;
    }

    public YoloModelBuilder Backend(InferenceBackend backend)
    {
        _builder.Backend = backend;
        return this;
    }

    public YoloModelBuilder InputSize(int width, int height)
    {
        _builder.InputWidth = Math.Max(1, width);
        _builder.InputHeight = Math.Max(1, height);
        return this;
    }

    public YoloModelBuilder InputChannels(int channels)
    {
        _builder.InputChannels = Math.Max(1, channels);
        return this;
    }

    public YoloModelBuilder Confidence(float value)
    {
        _builder.ConfidenceThreshold = Math.Clamp(value, 0f, 1f);
        return this;
    }

    public YoloModelBuilder Iou(float value)
    {
        _builder.IouThreshold = Math.Clamp(value, 0f, 1f);
        return this;
    }

    public YoloModelBuilder MaxDetections(int value)
    {
        _builder.MaxDetections = Math.Max(1, value);
        return this;
    }

    public YoloModelBuilder Classes(params string[] names)
    {
        _builder.ClassNames = names ?? Array.Empty<string>();
        return this;
    }

    public YoloModelBuilder ClassesFromFile(string labelsPath)
    {
        _builder.ClassNames = YoloLabels.LoadFromFile(labelsPath);
        return this;
    }

    public YoloModelBuilder AnchorFree(bool anchorFree)
    {
        _builder.AnchorFree = anchorFree;
        return this;
    }

    public YoloModelBuilder Anchors(float[][] anchors, int[] strides)
    {
        _builder.Anchors = anchors ?? Array.Empty<float[]>();
        _builder.Strides = strides ?? Array.Empty<int>();
        return this;
    }

    public YoloModelBuilder MaskProto(int width, int height, int channels = 32)
    {
        _builder.MaskProtoWidth = Math.Max(1, width);
        _builder.MaskProtoHeight = Math.Max(1, height);
        _builder.MaskProtoChannels = Math.Max(1, channels);
        return this;
    }

    public YoloModelBuilder Keypoints(int count)
    {
        _builder.KeypointCount = Math.Max(1, count);
        return this;
    }

    public YoloModelBuilder AngleInRadians(bool value)
    {
        _builder.AngleInRadians = value;
        return this;
    }
}

internal sealed class YoloConfigBuilder : YoloConfig
{
    public YoloConfig Build() => new YoloConfig
    {
        Version = Version,
        Task = Task,
        Backend = Backend,
        InputWidth = InputWidth,
        InputHeight = InputHeight,
        InputChannels = InputChannels,
        ConfidenceThreshold = ConfidenceThreshold,
        IouThreshold = IouThreshold,
        MaxDetections = MaxDetections,
        ClassNames = ClassNames,
        ClassAgnosticNms = ClassAgnosticNms,
        Anchors = Anchors,
        Strides = Strides,
        AnchorFree = AnchorFree,
        MaskProtoWidth = MaskProtoWidth,
        MaskProtoHeight = MaskProtoHeight,
        MaskProtoChannels = MaskProtoChannels,
        KeypointCount = KeypointCount,
        AngleInRadians = AngleInRadians,
    };
}
