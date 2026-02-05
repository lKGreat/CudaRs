using System;
using System.Text;
using CudaRS.Native;

namespace CudaRS.OpenVino.Preprocessing;

/// <summary>
/// Fluent API for configuring OpenVINO preprocessing steps.
/// </summary>
public sealed class PreprocessBuilder : IDisposable
{
    private ulong _preprocessHandle;
    private ulong _modelHandle;
    private bool _disposed;

    private PreprocessBuilder(ulong preprocessHandle, ulong modelHandle)
    {
        _preprocessHandle = preprocessHandle;
        _modelHandle = modelHandle;
    }

    /// <summary>
    /// Creates a new preprocessing builder for the specified model.
    /// </summary>
    /// <param name="model">The OpenVINO model to configure preprocessing for.</param>
    /// <returns>A new PreprocessBuilder instance.</returns>
    public static PreprocessBuilder Create(OpenVinoModel model)
    {
        if (model == null)
            throw new ArgumentNullException(nameof(model));

        ThrowIfFailed(SdkNative.OpenVinoPreprocessCreate(
            model.Handle,
            out var preprocessHandle));

        return new PreprocessBuilder(preprocessHandle, model.Handle);
    }

    /// <summary>
    /// Configures preprocessing for a specific input.
    /// </summary>
    /// <param name="inputIndex">The index of the input to configure (default: 0).</param>
    /// <returns>A PreprocessInputBuilder for configuring the input.</returns>
    public PreprocessInputBuilder Input(int inputIndex = 0)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(PreprocessBuilder));

        return new PreprocessInputBuilder(this, (ulong)inputIndex);
    }

    /// <summary>
    /// Builds the preprocessed model.
    /// </summary>
    /// <returns>A new OpenVinoModel with preprocessing applied.</returns>
    public OpenVinoModel Build()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(PreprocessBuilder));

        ThrowIfFailed(SdkNative.OpenVinoPreprocessBuild(
            _preprocessHandle,
            _modelHandle,
            out var newModelHandle));

        // Create a new model instance with the preprocessed handle
        return new OpenVinoModel(newModelHandle);
    }

    /// <summary>
    /// Disposes the preprocessing builder.
    /// </summary>
    public void Dispose()
    {
        if (!_disposed)
        {
            if (_preprocessHandle != 0)
            {
                SdkNative.OpenVinoPreprocessFree(_preprocessHandle);
                _preprocessHandle = 0;
            }
            _disposed = true;
        }
    }

    internal ulong PreprocessHandle => _preprocessHandle;

    internal static void ThrowIfFailed(CudaRsResult result)
    {
        if (result != CudaRsResult.Success)
        {
            throw new InvalidOperationException($"OpenVINO preprocessing operation failed with result: {result}");
        }
    }
}

/// <summary>
/// Fluent API for configuring preprocessing for a specific input.
/// </summary>
public sealed class PreprocessInputBuilder
{
    private readonly PreprocessBuilder _builder;
    private readonly ulong _inputIndex;

    internal PreprocessInputBuilder(PreprocessBuilder builder, ulong inputIndex)
    {
        _builder = builder;
        _inputIndex = inputIndex;
    }

    /// <summary>
    /// Sets the tensor format (element type and optional layout).
    /// </summary>
    /// <param name="elementType">The element type of the input tensor.</param>
    /// <param name="layout">Optional tensor layout (e.g., "NHWC").</param>
    /// <returns>This builder for chaining.</returns>
    public PreprocessInputBuilder TensorFormat(TensorElementType elementType, string? layout = null)
    {
        unsafe
        {
            byte* layoutPtr = null;
            byte[]? layoutBytes = null;

            if (layout != null)
            {
                layoutBytes = Encoding.UTF8.GetBytes(layout + '\0');
                fixed (byte* ptr = layoutBytes)
                {
                    layoutPtr = ptr;
                    PreprocessBuilder.ThrowIfFailed(SdkNative.OpenVinoPreprocessSetInputFormat(
                        _builder.PreprocessHandle,
                        _inputIndex,
                        (int)elementType,
                        layoutPtr));
                }
            }
            else
            {
                PreprocessBuilder.ThrowIfFailed(SdkNative.OpenVinoPreprocessSetInputFormat(
                    _builder.PreprocessHandle,
                    _inputIndex,
                    (int)elementType,
                    null));
            }
        }

        return this;
    }

    /// <summary>
    /// Sets the model's expected layout for this input.
    /// </summary>
    /// <param name="layout">The model layout (e.g., "NCHW").</param>
    /// <returns>This builder for chaining.</returns>
    public PreprocessInputBuilder ModelLayout(string layout)
    {
        if (layout == null)
            throw new ArgumentNullException(nameof(layout));

        unsafe
        {
            var layoutBytes = Encoding.UTF8.GetBytes(layout + '\0');
            fixed (byte* ptr = layoutBytes)
            {
                PreprocessBuilder.ThrowIfFailed(SdkNative.OpenVinoPreprocessSetModelLayout(
                    _builder.PreprocessHandle,
                    _inputIndex,
                    ptr));
            }
        }

        return this;
    }

    /// <summary>
    /// Adds a resize preprocessing step.
    /// </summary>
    /// <param name="algorithm">The resize algorithm to use.</param>
    /// <returns>This builder for chaining.</returns>
    public PreprocessInputBuilder Resize(ResizeAlgorithm algorithm)
    {
        PreprocessBuilder.ThrowIfFailed(SdkNative.OpenVinoPreprocessAddResize(
            _builder.PreprocessHandle,
            _inputIndex,
            (int)algorithm));

        return this;
    }

    /// <summary>
    /// Configures another input.
    /// </summary>
    /// <param name="inputIndex">The index of the input to configure.</param>
    /// <returns>A new PreprocessInputBuilder for the specified input.</returns>
    public PreprocessInputBuilder Input(int inputIndex)
    {
        return _builder.Input(inputIndex);
    }

    /// <summary>
    /// Builds the preprocessed model.
    /// </summary>
    /// <returns>A new OpenVinoModel with preprocessing applied.</returns>
    public OpenVinoModel Build()
    {
        return _builder.Build();
    }
}
