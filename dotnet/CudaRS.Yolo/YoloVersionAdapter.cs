using System;
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

    private static float[][] GetV3Anchors() => new[]
    {
        new[] { 116f, 90f, 156f, 198f, 373f, 326f },
        new[] { 30f, 61f, 62f, 45f, 59f, 119f },
        new[] { 10f, 13f, 16f, 30f, 33f, 23f },
    };

    private static float[][] GetV4Anchors() => new[]
    {
        new[] { 12f, 16f, 19f, 36f, 40f, 28f },
        new[] { 36f, 75f, 76f, 55f, 72f, 146f },
        new[] { 142f, 110f, 192f, 243f, 459f, 401f },
    };

    private static float[][] GetV5Anchors() => new[]
    {
        new[] { 10f, 13f, 16f, 30f, 33f, 23f },
        new[] { 30f, 61f, 62f, 45f, 59f, 119f },
        new[] { 116f, 90f, 156f, 198f, 373f, 326f },
    };

    private static float[][] GetV7Anchors() => new[]
    {
        new[] { 12f, 16f, 19f, 36f, 40f, 28f },
        new[] { 36f, 75f, 76f, 55f, 72f, 146f },
        new[] { 142f, 110f, 192f, 243f, 459f, 401f },
    };
}
