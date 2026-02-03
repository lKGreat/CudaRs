namespace CudaRS.Yolo;

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
