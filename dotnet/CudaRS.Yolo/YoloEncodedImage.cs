using System;
using System.IO;

namespace CudaRS.Yolo;

/// <summary>
/// Encoded image bytes (JPEG/PNG/BMP). This is the input expected by the native YOLO pipeline.
/// </summary>
public sealed class YoloEncodedImage
{
    public byte[] Data { get; }

    public YoloEncodedImage(byte[] data)
    {
        Data = data ?? throw new ArgumentNullException(nameof(data));
        if (data.Length == 0)
            throw new ArgumentException("Encoded image bytes required.", nameof(data));
    }

    public static YoloEncodedImage FromFile(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("Image path is required.", nameof(path));
        if (!File.Exists(path))
            throw new FileNotFoundException("Image file not found.", path);

        return new YoloEncodedImage(File.ReadAllBytes(path));
    }
}
