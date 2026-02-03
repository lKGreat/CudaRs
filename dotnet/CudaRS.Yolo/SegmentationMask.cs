using System;
using System.Collections.Generic;
using System.Linq;

namespace CudaRS.Yolo;

/// <summary>
/// Instance segmentation mask.
/// </summary>
public sealed class SegmentationMask
{
    public int Width { get; init; }
    public int Height { get; init; }
    public float[] Data { get; init; } = Array.Empty<float>();

    public bool[] ToBinary(float threshold = 0.5f)
        => Data.Select(v => v >= threshold).ToArray();

    public int GetArea(float threshold = 0.5f)
        => Data.Count(v => v >= threshold);

    public SegmentationMask ScaleTo(int targetWidth, int targetHeight)
    {
        if (Width == targetWidth && Height == targetHeight)
            return this;

        var scaled = new float[targetWidth * targetHeight];
        var scaleX = (float)Width / targetWidth;
        var scaleY = (float)Height / targetHeight;

        for (int y = 0; y < targetHeight; y++)
        {
            for (int x = 0; x < targetWidth; x++)
            {
                var srcX = (int)(x * scaleX);
                var srcY = (int)(y * scaleY);
                srcX = Math.Clamp(srcX, 0, Width - 1);
                srcY = Math.Clamp(srcY, 0, Height - 1);
                scaled[y * targetWidth + x] = Data[srcY * Width + srcX];
            }
        }

        return new SegmentationMask { Width = targetWidth, Height = targetHeight, Data = scaled };
    }

    public IReadOnlyList<Point2D> GetContour(float threshold = 0.5f)
    {
        var binary = ToBinary(threshold);
        var contour = new List<Point2D>();

        for (int y = 1; y < Height - 1; y++)
        {
            for (int x = 1; x < Width - 1; x++)
            {
                if (!binary[y * Width + x]) continue;

                var isEdge =
                    !binary[(y - 1) * Width + x] ||
                    !binary[(y + 1) * Width + x] ||
                    !binary[y * Width + (x - 1)] ||
                    !binary[y * Width + (x + 1)];

                if (isEdge)
                    contour.Add(new Point2D(x, y));
            }
        }

        return contour;
    }
}
