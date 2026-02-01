using System;
using System.Collections.Generic;

namespace CudaRS.Yolo;

public class YoloConfig
{
    public YoloVersion Version { get; set; } = YoloVersion.Auto;
    public YoloTask Task { get; set; } = YoloTask.Detect;
    public InferenceBackend Backend { get; set; } = InferenceBackend.Auto;

    public int InputWidth { get; set; } = 640;
    public int InputHeight { get; set; } = 640;
    public int InputChannels { get; set; } = 3;

    public float ConfidenceThreshold { get; set; } = 0.25f;
    public float IouThreshold { get; set; } = 0.45f;
    public int MaxDetections { get; set; } = 300;

    public string[] ClassNames { get; set; } = Array.Empty<string>();
    public bool ClassAgnosticNms { get; set; } = false;

    // Anchor-based models (v3/v4/v5/v6/v7)
    public float[][] Anchors { get; set; } = Array.Empty<float[]>();
    public int[] Strides { get; set; } = Array.Empty<int>();

    // Anchor-free models (v8/v9/v10/v11)
    public bool AnchorFree { get; set; } = true;

    // Segment task
    public int MaskProtoWidth { get; set; } = 160;
    public int MaskProtoHeight { get; set; } = 160;
    public int MaskProtoChannels { get; set; } = 32;

    // Pose task
    public int KeypointCount { get; set; } = 17;

    // OBB task
    public bool AngleInRadians { get; set; } = false;
}
