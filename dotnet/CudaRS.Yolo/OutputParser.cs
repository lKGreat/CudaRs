using System;

namespace CudaRS.Yolo;

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
