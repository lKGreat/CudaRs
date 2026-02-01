using System;
using System.Collections.Generic;
using System.Linq;

namespace CudaRS.Yolo;

/// <summary>
/// YOLO version adapter for handling differences between V3-V11.
/// Provides version-specific configurations, anchor handling, and output parsing.
/// </summary>
public static class YoloVersionAdapter
{
    /// <summary>
    /// Gets the default configuration for a specific YOLO version.
    /// </summary>
    public static YoloVersionConfig GetVersionConfig(YoloVersion version)
        => version switch
        {
            YoloVersion.V3 => new YoloVersionConfig
            {
                Version = YoloVersion.V3,
                AnchorBased = true,
                HasObjectness = true,
                DefaultStrides = new[] { 32, 16, 8 },
                DefaultAnchors = GetV3Anchors(),
                OutputOrder = OutputLayout.BatchHeadDetection,
                BoxFormat = BoxFormat.CenterWH,
                SupportedTasks = new[] { YoloTask.Detect },
            },
            YoloVersion.V4 => new YoloVersionConfig
            {
                Version = YoloVersion.V4,
                AnchorBased = true,
                HasObjectness = true,
                DefaultStrides = new[] { 8, 16, 32 },
                DefaultAnchors = GetV4Anchors(),
                OutputOrder = OutputLayout.BatchHeadDetection,
                BoxFormat = BoxFormat.CenterWH,
                SupportedTasks = new[] { YoloTask.Detect },
            },
            YoloVersion.V5 => new YoloVersionConfig
            {
                Version = YoloVersion.V5,
                AnchorBased = true,
                HasObjectness = true,
                DefaultStrides = new[] { 8, 16, 32 },
                DefaultAnchors = GetV5Anchors(),
                OutputOrder = OutputLayout.BatchDetectionChannel,
                BoxFormat = BoxFormat.CenterWH,
                SupportedTasks = new[] { YoloTask.Detect, YoloTask.Segment, YoloTask.Classify },
            },
            YoloVersion.V6 => new YoloVersionConfig
            {
                Version = YoloVersion.V6,
                AnchorBased = false, // V6 can be anchor-free
                HasObjectness = true,
                DefaultStrides = new[] { 8, 16, 32 },
                DefaultAnchors = Array.Empty<float[]>(),
                OutputOrder = OutputLayout.BatchDetectionChannel,
                BoxFormat = BoxFormat.CenterWH,
                SupportedTasks = new[] { YoloTask.Detect },
            },
            YoloVersion.V7 => new YoloVersionConfig
            {
                Version = YoloVersion.V7,
                AnchorBased = true,
                HasObjectness = true,
                DefaultStrides = new[] { 8, 16, 32 },
                DefaultAnchors = GetV7Anchors(),
                OutputOrder = OutputLayout.BatchDetectionChannel,
                BoxFormat = BoxFormat.CenterWH,
                SupportedTasks = new[] { YoloTask.Detect, YoloTask.Segment, YoloTask.Pose },
            },
            YoloVersion.V8 => new YoloVersionConfig
            {
                Version = YoloVersion.V8,
                AnchorBased = false,
                HasObjectness = false,
                DefaultStrides = new[] { 8, 16, 32 },
                DefaultAnchors = Array.Empty<float[]>(),
                OutputOrder = OutputLayout.BatchChannelDetection,
                BoxFormat = BoxFormat.CenterWH,
                SupportedTasks = new[] { YoloTask.Detect, YoloTask.Segment, YoloTask.Pose, YoloTask.Obb, YoloTask.Classify },
            },
            YoloVersion.V9 => new YoloVersionConfig
            {
                Version = YoloVersion.V9,
                AnchorBased = false,
                HasObjectness = false,
                DefaultStrides = new[] { 8, 16, 32 },
                DefaultAnchors = Array.Empty<float[]>(),
                OutputOrder = OutputLayout.BatchChannelDetection,
                BoxFormat = BoxFormat.CenterWH,
                SupportedTasks = new[] { YoloTask.Detect, YoloTask.Segment },
            },
            YoloVersion.V10 => new YoloVersionConfig
            {
                Version = YoloVersion.V10,
                AnchorBased = false,
                HasObjectness = false,
                DefaultStrides = new[] { 8, 16, 32 },
                DefaultAnchors = Array.Empty<float[]>(),
                OutputOrder = OutputLayout.BatchDetectionChannel,
                BoxFormat = BoxFormat.Xyxy, // V10 outputs XYXY directly
                SupportedTasks = new[] { YoloTask.Detect },
                NmsBuiltIn = true, // V10 has built-in NMS
            },
            YoloVersion.V11 => new YoloVersionConfig
            {
                Version = YoloVersion.V11,
                AnchorBased = false,
                HasObjectness = false,
                DefaultStrides = new[] { 8, 16, 32 },
                DefaultAnchors = Array.Empty<float[]>(),
                OutputOrder = OutputLayout.BatchChannelDetection,
                BoxFormat = BoxFormat.CenterWH,
                SupportedTasks = new[] { YoloTask.Detect, YoloTask.Segment, YoloTask.Pose, YoloTask.Obb, YoloTask.Classify },
            },
            _ => GetVersionConfig(YoloVersion.V8), // Default to V8 for Auto
        };

    /// <summary>
    /// Detects YOLO version from model output shape.
    /// </summary>
    public static YoloVersion DetectVersion(int[] outputShape, int numClasses)
    {
        if (outputShape.Length < 2)
            return YoloVersion.V8;

        // Get dimensions
        var dim1 = outputShape.Length >= 2 ? outputShape[1] : 0;
        var dim2 = outputShape.Length >= 3 ? outputShape[2] : 0;

        // V10 signature: [batch, num_detections, 6] - box(4) + score(1) + class(1)
        if (dim2 == 6)
            return YoloVersion.V10;

        // Anchor-free (V8/V9/V11): channels = 4 + num_classes
        // Anchor-based (V5/V7): channels = 5 + num_classes (includes objectness)
        var expectedAnchorFree = 4 + numClasses;
        var expectedAnchorBased = 5 + numClasses;

        // Check channel count
        if (dim1 == expectedAnchorFree || dim2 == expectedAnchorFree)
            return YoloVersion.V8; // Default anchor-free

        if (dim1 == expectedAnchorBased || dim2 == expectedAnchorBased)
            return YoloVersion.V5; // Default anchor-based

        return YoloVersion.V8;
    }

    /// <summary>
    /// Applies version-specific configuration to a YoloConfig.
    /// </summary>
    public static void ApplyVersionDefaults(YoloConfig config)
    {
        var versionConfig = GetVersionConfig(config.Version);

        config.AnchorFree = !versionConfig.AnchorBased;

        if (versionConfig.AnchorBased && (config.Anchors == null || config.Anchors.Length == 0))
            config.Anchors = versionConfig.DefaultAnchors;

        if (config.Strides == null || config.Strides.Length == 0)
            config.Strides = versionConfig.DefaultStrides;
    }

    /// <summary>
    /// Validates that the task is supported by the version.
    /// </summary>
    public static bool IsTaskSupported(YoloVersion version, YoloTask task)
    {
        var config = GetVersionConfig(version);
        return config.SupportedTasks.Contains(task);
    }

    /// <summary>
    /// Gets anchors for YOLOv3 (COCO pre-trained).
    /// </summary>
    private static float[][] GetV3Anchors() => new[]
    {
        // P5/32: large objects
        new[] { 116f, 90f, 156f, 198f, 373f, 326f },
        // P4/16: medium objects
        new[] { 30f, 61f, 62f, 45f, 59f, 119f },
        // P3/8: small objects
        new[] { 10f, 13f, 16f, 30f, 33f, 23f },
    };

    /// <summary>
    /// Gets anchors for YOLOv4 (COCO pre-trained).
    /// </summary>
    private static float[][] GetV4Anchors() => new[]
    {
        // P3/8
        new[] { 12f, 16f, 19f, 36f, 40f, 28f },
        // P4/16
        new[] { 36f, 75f, 76f, 55f, 72f, 146f },
        // P5/32
        new[] { 142f, 110f, 192f, 243f, 459f, 401f },
    };

    /// <summary>
    /// Gets anchors for YOLOv5 (COCO pre-trained).
    /// </summary>
    private static float[][] GetV5Anchors() => new[]
    {
        // P3/8
        new[] { 10f, 13f, 16f, 30f, 33f, 23f },
        // P4/16
        new[] { 30f, 61f, 62f, 45f, 59f, 119f },
        // P5/32
        new[] { 116f, 90f, 156f, 198f, 373f, 326f },
    };

    /// <summary>
    /// Gets anchors for YOLOv7 (COCO pre-trained).
    /// </summary>
    private static float[][] GetV7Anchors() => new[]
    {
        // P3/8
        new[] { 12f, 16f, 19f, 36f, 40f, 28f },
        // P4/16
        new[] { 36f, 75f, 76f, 55f, 72f, 146f },
        // P5/32
        new[] { 142f, 110f, 192f, 243f, 459f, 401f },
    };
}

/// <summary>
/// Configuration specific to a YOLO version.
/// </summary>
public sealed class YoloVersionConfig
{
    public YoloVersion Version { get; init; }
    public bool AnchorBased { get; init; }
    public bool HasObjectness { get; init; }
    public int[] DefaultStrides { get; init; } = Array.Empty<int>();
    public float[][] DefaultAnchors { get; init; } = Array.Empty<float[]>();
    public OutputLayout OutputOrder { get; init; }
    public BoxFormat BoxFormat { get; init; }
    public YoloTask[] SupportedTasks { get; init; } = Array.Empty<YoloTask>();
    public bool NmsBuiltIn { get; init; } = false;
}

/// <summary>
/// Output tensor layout order.
/// </summary>
public enum OutputLayout
{
    /// <summary>[batch, channels, num_detections] - V8/V9/V11 style</summary>
    BatchChannelDetection,

    /// <summary>[batch, num_detections, channels] - V5/V10 style</summary>
    BatchDetectionChannel,

    /// <summary>[batch, head, h, w, channels] - V3/V4 style</summary>
    BatchHeadDetection,
}

/// <summary>
/// Anchor decoder for anchor-based YOLO models (V3/V4/V5/V6/V7).
/// </summary>
public static class AnchorDecoder
{
    /// <summary>
    /// Decodes anchor-based predictions to absolute coordinates.
    /// </summary>
    public static void DecodeAnchors(
        Span<float> output,
        int gridW,
        int gridH,
        int stride,
        float[] anchors,
        int numClasses,
        List<BoundingBox> boxes,
        List<float> scores,
        List<int> classIds,
        float confThreshold,
        bool hasObjectness)
    {
        var numAnchors = anchors.Length / 2;
        var channelsPerAnchor = (hasObjectness ? 5 : 4) + numClasses;

        for (int ay = 0; ay < gridH; ay++)
        {
            for (int ax = 0; ax < gridW; ax++)
            {
                for (int a = 0; a < numAnchors; a++)
                {
                    var offset = (ay * gridW * numAnchors + ax * numAnchors + a) * channelsPerAnchor;

                    // Get objectness
                    float objectness = hasObjectness
                        ? Sigmoid(output[offset + 4])
                        : 1.0f;

                    if (objectness < confThreshold)
                        continue;

                    // Find best class
                    var classStart = hasObjectness ? 5 : 4;
                    var bestClass = 0;
                    var bestProb = 0f;

                    for (int c = 0; c < numClasses; c++)
                    {
                        var prob = Sigmoid(output[offset + classStart + c]);
                        if (prob > bestProb)
                        {
                            bestProb = prob;
                            bestClass = c;
                        }
                    }

                    var confidence = objectness * bestProb;
                    if (confidence < confThreshold)
                        continue;

                    // Decode box
                    var anchorW = anchors[a * 2];
                    var anchorH = anchors[a * 2 + 1];

                    var tx = Sigmoid(output[offset]);
                    var ty = Sigmoid(output[offset + 1]);
                    var tw = output[offset + 2];
                    var th = output[offset + 3];

                    var cx = (ax + tx) * stride;
                    var cy = (ay + ty) * stride;
                    var w = anchorW * MathF.Exp(tw);
                    var h = anchorH * MathF.Exp(th);

                    boxes.Add(BoundingBox.FromCenterWH(cx, cy, w, h));
                    scores.Add(confidence);
                    classIds.Add(bestClass);
                }
            }
        }
    }

    /// <summary>
    /// Decodes V10-style output (already post-NMS with XYXY format).
    /// </summary>
    public static void DecodeV10(
        Span<float> output,
        int numDetections,
        List<BoundingBox> boxes,
        List<float> scores,
        List<int> classIds,
        float confThreshold)
    {
        // V10 output: [batch, num_det, 6] where 6 = [x1, y1, x2, y2, score, class_id]
        for (int i = 0; i < numDetections; i++)
        {
            var offset = i * 6;

            var x1 = output[offset];
            var y1 = output[offset + 1];
            var x2 = output[offset + 2];
            var y2 = output[offset + 3];
            var score = output[offset + 4];
            var classId = (int)output[offset + 5];

            if (score < confThreshold)
                continue;

            // V10 outputs XYXY, convert to XYWH
            var box = new BoundingBox(x1, y1, x2 - x1, y2 - y1);
            boxes.Add(box);
            scores.Add(score);
            classIds.Add(classId);
        }
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));
}

/// <summary>
/// Output tensor parser that handles different YOLO version formats.
/// </summary>
public static class OutputParser
{
    /// <summary>
    /// Parses output tensor according to version-specific layout.
    /// </summary>
    public static ParsedOutput Parse(
        float[] data,
        int[] shape,
        YoloVersionConfig versionConfig,
        int numClasses)
    {
        var layout = versionConfig.OutputOrder;
        var hasObjectness = versionConfig.HasObjectness;

        int batch, channels, count;

        switch (layout)
        {
            case OutputLayout.BatchChannelDetection:
                // [batch, channels, num_detections]
                batch = shape[0];
                channels = shape[1];
                count = shape[2];
                return new ParsedOutput
                {
                    Data = data,
                    Batch = batch,
                    Channels = channels,
                    Count = count,
                    Transposed = true,
                    HasObjectness = hasObjectness,
                    NumClasses = numClasses,
                };

            case OutputLayout.BatchDetectionChannel:
                // [batch, num_detections, channels]
                batch = shape[0];
                count = shape[1];
                channels = shape[2];
                return new ParsedOutput
                {
                    Data = data,
                    Batch = batch,
                    Channels = channels,
                    Count = count,
                    Transposed = false,
                    HasObjectness = hasObjectness,
                    NumClasses = numClasses,
                };

            case OutputLayout.BatchHeadDetection:
                // Multi-head output (V3/V4) - flattened
                // Assume already flattened to [batch, total_anchors, channels]
                batch = shape[0];
                count = shape[1];
                channels = shape[2];
                return new ParsedOutput
                {
                    Data = data,
                    Batch = batch,
                    Channels = channels,
                    Count = count,
                    Transposed = false,
                    HasObjectness = hasObjectness,
                    NumClasses = numClasses,
                };

            default:
                throw new NotSupportedException($"Unknown output layout: {layout}");
        }
    }

    /// <summary>
    /// Gets a value from parsed output.
    /// </summary>
    public static float GetValue(ParsedOutput output, int index, int channel)
    {
        return output.Transposed
            ? output.Data[channel * output.Count + index]
            : output.Data[index * output.Channels + channel];
    }
}

/// <summary>
/// Parsed output tensor information.
/// </summary>
public sealed class ParsedOutput
{
    public float[] Data { get; init; } = Array.Empty<float>();
    public int Batch { get; init; }
    public int Channels { get; init; }
    public int Count { get; init; }
    public bool Transposed { get; init; }
    public bool HasObjectness { get; init; }
    public int NumClasses { get; init; }
}
