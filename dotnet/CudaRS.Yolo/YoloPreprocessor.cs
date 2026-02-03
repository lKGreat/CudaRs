using System;
using System.Collections.Generic;

namespace CudaRS.Yolo;

public static class YoloPreprocessor
{
    public static YoloPreprocessResult Letterbox(YoloImage image, int targetWidth, int targetHeight)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));

        var scale = Math.Min((float)targetWidth / image.Width, (float)targetHeight / image.Height);
        var newW = (int)Math.Round(image.Width * scale);
        var newH = (int)Math.Round(image.Height * scale);

        var padX = (targetWidth - newW) / 2;
        var padY = (targetHeight - newH) / 2;

        var resized = ResizeNearest(image, newW, newH);
        var boxed = new byte[targetWidth * targetHeight * image.Channels];

        for (int y = 0; y < newH; y++)
        {
            var dstRow = (y + padY) * targetWidth;
            var srcRow = y * newW;

            for (int x = 0; x < newW; x++)
            {
                var dstIndex = (dstRow + x + padX) * image.Channels;
                var srcIndex = (srcRow + x) * image.Channels;
                for (int c = 0; c < image.Channels; c++)
                    boxed[dstIndex + c] = resized[srcIndex + c];
            }
        }

        var input = HwcToChw(boxed, targetWidth, targetHeight, image.Channels);
        return new YoloPreprocessResult
        {
            Input = input,
            InputShape = new[] { 1, image.Channels, targetHeight, targetWidth },
            Scale = scale,
            PadX = padX,
            PadY = padY,
            OriginalWidth = image.Width,
            OriginalHeight = image.Height,
        };
    }

    public static YoloBatchPreprocessResult LetterboxBatch(IReadOnlyList<YoloImage> images, int targetWidth, int targetHeight)
    {
        if (images == null)
            throw new ArgumentNullException(nameof(images));
        if (images.Count == 0)
            throw new ArgumentException("At least one image is required.", nameof(images));

        var channels = images[0].Channels;
        var perImageSize = targetWidth * targetHeight * channels;
        var batch = images.Count;

        var input = new float[perImageSize * batch];
        var items = new YoloPreprocessResult[batch];

        for (int i = 0; i < batch; i++)
        {
            var image = images[i];
            if (image.Channels != channels)
                throw new ArgumentException("All images must have the same channel count.", nameof(images));

            var res = Letterbox(image, targetWidth, targetHeight);
            items[i] = res;

            Buffer.BlockCopy(
                res.Input,
                0,
                input,
                i * perImageSize * sizeof(float),
                perImageSize * sizeof(float));
        }

        return new YoloBatchPreprocessResult
        {
            Input = input,
            InputShape = new[] { batch, channels, targetHeight, targetWidth },
            Items = items,
        };
    }

    private static byte[] ResizeNearest(YoloImage image, int newW, int newH)
    {
        var result = new byte[newW * newH * image.Channels];
        var scaleX = (float)image.Width / newW;
        var scaleY = (float)image.Height / newH;

        for (int y = 0; y < newH; y++)
        {
            var srcY = Math.Min((int)(y * scaleY), image.Height - 1);
            for (int x = 0; x < newW; x++)
            {
                var srcX = Math.Min((int)(x * scaleX), image.Width - 1);
                var srcIndex = (srcY * image.Width + srcX) * image.Channels;
                var dstIndex = (y * newW + x) * image.Channels;

                for (int c = 0; c < image.Channels; c++)
                    result[dstIndex + c] = image.Data[srcIndex + c];
            }
        }

        return result;
    }

    private static float[] HwcToChw(byte[] input, int width, int height, int channels)
    {
        var output = new float[width * height * channels];
        var hw = width * height;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                var idx = (y * width + x) * channels;
                for (int c = 0; c < channels; c++)
                {
                    output[c * hw + y * width + x] = input[idx + c] / 255f;
                }
            }
        }

        return output;
    }
}
