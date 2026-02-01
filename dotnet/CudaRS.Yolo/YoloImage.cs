using System;

namespace CudaRS.Yolo;

public sealed class YoloImage
{
    public int Width { get; }
    public int Height { get; }
    public int Channels { get; }
    public byte[] Data { get; }

    public YoloImage(int width, int height, int channels, byte[] data)
    {
        if (width <= 0 || height <= 0 || channels <= 0)
            throw new ArgumentOutOfRangeException(nameof(width));
        Data = data ?? throw new ArgumentNullException(nameof(data));
        if (data.Length < width * height * channels)
            throw new ArgumentException("Data length does not match image size.", nameof(data));

        Width = width;
        Height = height;
        Channels = channels;
    }
}
