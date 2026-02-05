using System;
using System.Collections.Generic;
using System.IO;
using SixLabors.Fonts;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace CudaRS.Yolo;

public static class ImageAnnotator
{
    private static readonly Dictionary<int, Color> DefaultColors = GenerateDefaultColors();

    public static Image<Rgb24> DrawBoxes(
        Image<Rgb24> image,
        IReadOnlyList<Detection> detections,
        AnnotationOptions? options = null)
    {
        if (image == null) throw new ArgumentNullException(nameof(image));
        if (detections == null) throw new ArgumentNullException(nameof(detections));

        options ??= new AnnotationOptions();
        var colors = options.ClassColors ?? DefaultColors;

        image.Mutate(ctx =>
        {
            foreach (var detection in detections)
            {
                var color = colors.TryGetValue(detection.ClassId, out var c) ? c : Color.Red;
                var box = detection.Box;

                // Draw rectangle
                var rect = new RectangleF(box.X, box.Y, box.Width, box.Height);
                ctx.Draw(color, options.BoxThickness, rect);

                // Draw label if requested
                if (options.ShowLabel || options.ShowConfidence)
                {
                    var labelText = BuildLabel(detection, options);
                    DrawLabel(ctx, labelText, box.X, box.Y, color, options);
                }
            }
        });

        return image;
    }

    public static byte[] DrawBoxesToBytes(
        ReadOnlyMemory<byte> imageBytes,
        IReadOnlyList<Detection> detections,
        AnnotationOptions? options = null,
        ImageFormat format = ImageFormat.Jpeg)
    {
        using var image = Image.Load<Rgb24>(imageBytes.ToArray());
        DrawBoxes(image, detections, options);

        using var ms = new MemoryStream();
        switch (format)
        {
            case ImageFormat.Jpeg:
                image.SaveAsJpeg(ms, new JpegEncoder { Quality = options?.JpegQuality ?? 90 });
                break;
            case ImageFormat.Png:
                image.SaveAsPng(ms);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(format));
        }

        return ms.ToArray();
    }

    public static byte[] DrawBoxesToBytes(
        Image<Rgb24> image,
        ImageFormat format = ImageFormat.Jpeg,
        int jpegQuality = 90)
    {
        using var ms = new MemoryStream();
        switch (format)
        {
            case ImageFormat.Jpeg:
                image.SaveAsJpeg(ms, new JpegEncoder { Quality = jpegQuality });
                break;
            case ImageFormat.Png:
                image.SaveAsPng(ms);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(format));
        }

        return ms.ToArray();
    }

    private static string BuildLabel(Detection detection, AnnotationOptions options)
    {
        var parts = new List<string>();

        if (options.ShowLabel && !string.IsNullOrEmpty(detection.ClassName))
            parts.Add(detection.ClassName);

        if (options.ShowConfidence)
            parts.Add($"{detection.Confidence:P0}");

        return string.Join(" ", parts);
    }

    private static void DrawLabel(
        IImageProcessingContext ctx,
        string text,
        float x,
        float y,
        Color boxColor,
        AnnotationOptions options)
    {
        if (string.IsNullOrEmpty(text))
            return;

        try
        {
            var font = SystemFonts.CreateFont("Arial", options.FontSize, FontStyle.Bold);
            var textOptions = new RichTextOptions(font)
            {
                Origin = new PointF(x + 2, y - options.FontSize - 4)
            };

            // Measure text
            var bounds = TextMeasurer.MeasureBounds(text, textOptions);

            // Draw background
            var bgRect = new RectangleF(
                x,
                y - bounds.Height - 6,
                bounds.Width + 4,
                bounds.Height + 4);
            ctx.Fill(boxColor, bgRect);

            // Draw text
            ctx.DrawText(textOptions, text, Color.White);
        }
        catch
        {
            // Fallback: skip label if font/drawing fails
        }
    }

    private static Dictionary<int, Color> GenerateDefaultColors()
    {
        var colors = new Dictionary<int, Color>();
        var palette = new[]
        {
            Color.Red, Color.Green, Color.Blue, Color.Yellow, Color.Cyan,
            Color.Magenta, Color.Orange, Color.Purple, Color.Pink, Color.Lime,
            Color.Teal, Color.Navy, Color.Maroon, Color.Olive, Color.Aqua
        };

        for (int i = 0; i < 100; i++)
        {
            colors[i] = palette[i % palette.Length];
        }

        return colors;
    }
}

public class AnnotationOptions
{
    public float BoxThickness { get; set; } = 2f;
    public bool ShowLabel { get; set; } = true;
    public bool ShowConfidence { get; set; } = true;
    public float FontSize { get; set; } = 14f;
    public int JpegQuality { get; set; } = 90;
    public Dictionary<int, Color>? ClassColors { get; set; }
}

public enum ImageFormat
{
    Jpeg,
    Png
}
