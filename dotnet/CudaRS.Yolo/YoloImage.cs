using System;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

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

    public static YoloImage FromFile(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("Image path is required.", nameof(path));

        using var image = Image.Load<Rgb24>(path);
        var data = new byte[image.Width * image.Height * 3];
        image.CopyPixelDataTo(data);

        return new YoloImage(image.Width, image.Height, 3, data);
    }
}
