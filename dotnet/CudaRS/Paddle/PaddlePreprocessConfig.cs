using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace CudaRS.Paddle;

/// <summary>
/// Preprocessing configuration for PaddlePaddle models
/// </summary>
public sealed class PaddlePreprocessConfig
{
    public float[]? Mean { get; set; }
    public float[]? Std { get; set; }
    public float Scale { get; set; } = 1.0f;
    public int[]? ImageShape { get; set; }
    public string[]? Transforms { get; set; }
    public bool IsScale { get; set; } = true;
    public bool IsCHW { get; set; } = true;
    public string ColorSpace { get; set; } = "RGB";

    /// <summary>
    /// Load preprocessing configuration from inference.yml file
    /// </summary>
    public static PaddlePreprocessConfig FromYaml(string yamlFilePath)
    {
        var config = PaddleYamlParser.ParseFile(yamlFilePath);
        return FromDictionary(config);
    }

    /// <summary>
    /// Create preprocessing configuration from parsed YAML dictionary
    /// </summary>
    public static PaddlePreprocessConfig FromDictionary(Dictionary<string, object> config)
    {
        var preprocessConfig = new PaddlePreprocessConfig();

        // Parse mean values
        var mean = PaddleYamlParser.GetArray<float>(config, "mean");
        if (mean != null && mean.Length > 0)
        {
            preprocessConfig.Mean = mean;
        }

        // Parse std values
        var std = PaddleYamlParser.GetArray<float>(config, "std");
        if (std != null && std.Length > 0)
        {
            preprocessConfig.Std = std;
        }

        // Parse scale
        preprocessConfig.Scale = PaddleYamlParser.GetValue(config, "scale", 1.0f);

        // Parse image shape
        var imageShape = PaddleYamlParser.GetArray<int>(config, "image_shape");
        if (imageShape != null && imageShape.Length > 0)
        {
            preprocessConfig.ImageShape = imageShape;
        }

        // Parse is_scale flag
        preprocessConfig.IsScale = PaddleYamlParser.GetValue(config, "is_scale", true);

        // Parse channel order
        var channelOrder = PaddleYamlParser.GetValue(config, "channel_order", "CHW");
        preprocessConfig.IsCHW = channelOrder.Equals("CHW", StringComparison.OrdinalIgnoreCase);

        // Parse color space
        preprocessConfig.ColorSpace = PaddleYamlParser.GetValue(config, "color_space", "RGB");

        // Parse transforms if available
        var transforms = PaddleYamlParser.GetArray<string>(config, "transforms");
        if (transforms != null && transforms.Length > 0)
        {
            preprocessConfig.Transforms = transforms;
        }

        return preprocessConfig;
    }

    /// <summary>
    /// Apply preprocessing to image data
    /// </summary>
    public float[] Preprocess(float[] imageData, int channels, int height, int width)
    {
        if (imageData == null || imageData.Length == 0)
            throw new ArgumentException("Image data is empty", nameof(imageData));

        var expectedSize = channels * height * width;
        if (imageData.Length != expectedSize)
            throw new ArgumentException(
                $"Image data size mismatch. Expected {expectedSize}, got {imageData.Length}", 
                nameof(imageData));

        var result = new float[imageData.Length];
        Array.Copy(imageData, result, imageData.Length);

        // Apply scaling
        if (IsScale && Math.Abs(Scale - 1.0f) > 0.001f)
        {
            for (int i = 0; i < result.Length; i++)
            {
                result[i] *= Scale;
            }
        }

        // Apply mean and std normalization
        if (Mean != null && Mean.Length > 0)
        {
            ApplyNormalization(result, channels, height, width, Mean, Std);
        }

        return result;
    }

    private void ApplyNormalization(float[] data, int channels, int height, int width, 
                                   float[] mean, float[]? std)
    {
        var pixelCount = height * width;

        for (int c = 0; c < channels; c++)
        {
            var meanVal = c < mean.Length ? mean[c] : mean[0];
            var stdVal = std != null && c < std.Length ? std[c] : 1.0f;

            if (IsCHW)
            {
                // CHW format: [C, H, W]
                var channelOffset = c * pixelCount;
                for (int i = 0; i < pixelCount; i++)
                {
                    var idx = channelOffset + i;
                    data[idx] = (data[idx] - meanVal) / stdVal;
                }
            }
            else
            {
                // HWC format: [H, W, C]
                for (int i = 0; i < pixelCount; i++)
                {
                    var idx = i * channels + c;
                    data[idx] = (data[idx] - meanVal) / stdVal;
                }
            }
        }
    }

    /// <summary>
    /// Get the expected input shape for the model
    /// </summary>
    public int[] GetInputShape(int batchSize = 1)
    {
        if (ImageShape == null || ImageShape.Length < 2)
            throw new InvalidOperationException("Image shape not configured");

        // Assume ImageShape is [C, H, W] or [H, W] or [H, W, C]
        if (ImageShape.Length == 3)
        {
            if (IsCHW)
            {
                // [C, H, W] -> [batch, C, H, W]
                return new[] { batchSize, ImageShape[0], ImageShape[1], ImageShape[2] };
            }
            else
            {
                // [H, W, C] -> [batch, H, W, C]
                return new[] { batchSize, ImageShape[0], ImageShape[1], ImageShape[2] };
            }
        }
        else if (ImageShape.Length == 2)
        {
            // [H, W] -> assume 3 channels, CHW format
            return new[] { batchSize, 3, ImageShape[0], ImageShape[1] };
        }

        throw new InvalidOperationException($"Unsupported image shape length: {ImageShape.Length}");
    }

    public override string ToString()
    {
        var parts = new List<string>
        {
            $"Scale={Scale}",
            $"IsScale={IsScale}",
            $"IsCHW={IsCHW}",
            $"ColorSpace={ColorSpace}"
        };

        if (Mean != null && Mean.Length > 0)
            parts.Add($"Mean=[{string.Join(", ", Mean)}]");
        
        if (Std != null && Std.Length > 0)
            parts.Add($"Std=[{string.Join(", ", Std)}]");
        
        if (ImageShape != null && ImageShape.Length > 0)
            parts.Add($"ImageShape=[{string.Join(", ", ImageShape)}]");
        
        if (Transforms != null && Transforms.Length > 0)
            parts.Add($"Transforms=[{string.Join(", ", Transforms)}]");

        return $"PaddlePreprocessConfig {{ {string.Join(", ", parts)} }}";
    }
}
