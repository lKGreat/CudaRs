using System;
using System.Threading;
using System.Threading.Tasks;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace CudaRS.Yolo;

/// <summary>
/// Wrapper for fluent result transformation API
/// </summary>
public sealed class FluentResultWrapper : IDisposable
{
    private readonly ReadOnlyMemory<byte> _imageBytes;
    private readonly ModelInferenceResult _inferenceResult;
    private Image<Rgb24>? _loadedImage;

    public FluentResultWrapper(ReadOnlyMemory<byte> imageBytes, ModelInferenceResult inferenceResult)
    {
        _imageBytes = imageBytes;
        _inferenceResult = inferenceResult ?? throw new ArgumentNullException(nameof(inferenceResult));
    }

    /// <summary>
    /// Return pure detection data (default behavior)
    /// </summary>
    public ModelInferenceResult AsDetections()
    {
        return _inferenceResult;
    }

    /// <summary>
    /// Return image with drawn bounding boxes
    /// </summary>
    public AnnotatedImageResult AsAnnotatedImage(AnnotationOptions? options = null, ImageFormat format = ImageFormat.Jpeg)
    {
        options ??= new AnnotationOptions();
        
        var image = GetOrLoadImage();
        ImageAnnotator.DrawBoxes(image, _inferenceResult.Detections, options);
        
        var bytes = ImageAnnotator.DrawBoxesToBytes(image, format, options.JpegQuality);
        
        return new AnnotatedImageResult
        {
            ImageBytes = bytes,
            Format = format,
            Width = image.Width,
            Height = image.Height
        };
    }

    /// <summary>
    /// Return both detection data and annotated image
    /// </summary>
    public CombinedResult AsCombined(AnnotationOptions? options = null, ImageFormat format = ImageFormat.Jpeg)
    {
        return new CombinedResult
        {
            Inference = _inferenceResult,
            AnnotatedImage = AsAnnotatedImage(options, format)
        };
    }

    private Image<Rgb24> GetOrLoadImage()
    {
        if (_loadedImage == null)
        {
            _loadedImage = Image.Load<Rgb24>(_imageBytes.ToArray());
        }
        return _loadedImage;
    }

    public void Dispose()
    {
        _loadedImage?.Dispose();
    }
}

/// <summary>
/// Extended fluent pipeline interface supporting result transformation
/// </summary>
public interface IFluentYoloPipeline : IDisposable
{
    FluentResultWrapper Run(ReadOnlyMemory<byte> imageBytes);
    Task<FluentResultWrapper> RunAsync(ReadOnlyMemory<byte> imageBytes, CancellationToken cancellationToken = default);
}
