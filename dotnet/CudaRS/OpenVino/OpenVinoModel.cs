using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using CudaRS;
using CudaRS.Interop;
using CudaRS.Native;

namespace CudaRS.OpenVino;

public sealed class OpenVinoModel : IDisposable
{
    private readonly ModelHub _hub;
    private readonly bool _ownsHub;
    private readonly ModelHandle _model;

    public OpenVinoModel(string modelId, OpenVinoModelConfig config, ModelHub? hub = null)
    {
        if (string.IsNullOrWhiteSpace(modelId))
            throw new ArgumentException("Model id is required.", nameof(modelId));
        if (config == null)
            throw new ArgumentNullException(nameof(config));
        if (string.IsNullOrWhiteSpace(config.ModelPath))
            throw new ArgumentException("ModelPath is required.", nameof(config));

        _hub = hub ?? new ModelHub();
        _ownsHub = hub == null;
        _model = _hub.LoadModel(new ModelOptions
        {
            ModelId = modelId,
            Kind = ModelKind.OpenVino,
            ConfigJson = config.ToJson()
        });
    }

    public OpenVinoPipeline CreatePipeline(string pipelineId, OpenVinoPipelineConfig? config = null)
    {
        var pipelineConfig = config ?? new OpenVinoPipelineConfig();
        var handle = _hub.CreatePipeline(_model, new PipelineOptions
        {
            PipelineId = string.IsNullOrWhiteSpace(pipelineId) ? "default" : pipelineId,
            Kind = PipelineKind.OpenVinoTensor,
            ConfigJson = pipelineConfig.ToJson()
        });

        return new OpenVinoPipeline(handle);
    }

    /// <summary>
    /// Gets information about the model inputs.
    /// </summary>
    /// <returns>Array of input tensor information.</returns>
    public unsafe ModelTensorInfo[] GetInputs()
    {
        return GetTensorInfo(true);
    }

    /// <summary>
    /// Gets information about the model outputs.
    /// </summary>
    /// <returns>Array of output tensor information.</returns>
    public unsafe ModelTensorInfo[] GetOutputs()
    {
        return GetTensorInfo(false);
    }

    /// <summary>
    /// Reshapes the model to the specified fixed dimensions.
    /// </summary>
    /// <param name="shape">The new shape for the input tensor (e.g., [1, 3, 640, 640]).</param>
    public unsafe void Reshape(params long[] shape)
    {
        if (shape == null || shape.Length == 0)
            throw new ArgumentException("Shape cannot be null or empty.", nameof(shape));

        var handle = _model.Value;

        fixed (long* shapePtr = shape)
        {
            var result = SdkNative.OpenVinoReshapeFixed(handle, shapePtr, (ulong)shape.Length);
            ThrowIfFailed(result);
        }
    }

    /// <summary>
    /// Reshapes the model with a partial shape (supports dynamic dimensions).
    /// </summary>
    /// <param name="shape">The partial shape with static and/or dynamic dimensions.</param>
    public unsafe void Reshape(PartialShape shape)
    {
        if (shape == null)
            throw new ArgumentNullException(nameof(shape));

        var handle = _model.Value;

        // Convert PartialShape to native representation
        var nativeDims = new CudaRsOvPartialDim[shape.Dimensions.Length];
        for (int i = 0; i < shape.Dimensions.Length; i++)
        {
            var dim = shape.Dimensions[i];
            nativeDims[i] = new CudaRsOvPartialDim
            {
                IsStatic = dim.IsStatic ? 1 : 0,
                Value = dim.Value
            };
        }

        fixed (CudaRsOvPartialDim* dimsPtr = nativeDims)
        {
            var partialShapeArray = new CudaRsOvPartialShapeArray
            {
                Dims = (IntPtr)dimsPtr,
                Rank = (ulong)nativeDims.Length
            };

            var result = SdkNative.OpenVinoReshapeDynamic(handle, in partialShapeArray);
            ThrowIfFailed(result);
        }
    }

    private unsafe ModelTensorInfo[] GetTensorInfo(bool isInput)
    {
        var handle = _model.Value;

        // Get count
        var result = isInput 
            ? SdkNative.OpenVinoGetInputCount(handle, out var count)
            : SdkNative.OpenVinoGetOutputCount(handle, out count);
        ThrowIfFailed(result);

        var tensors = new List<ModelTensorInfo>();
        for (ulong i = 0; i < count; i++)
        {
            CudaRsOvTensorInfo info;
            result = isInput
                ? SdkNative.OpenVinoGetInputInfo(handle, i, out info)
                : SdkNative.OpenVinoGetOutputInfo(handle, i, out info);
            ThrowIfFailed(result);

            try
            {
                var tensorInfo = new ModelTensorInfo();

                // Extract name
                if (info.NamePtr != IntPtr.Zero && info.NameLen > 0)
                {
                    var nameBytes = new byte[info.NameLen];
                    Marshal.Copy(info.NamePtr, nameBytes, 0, (int)info.NameLen);
                    tensorInfo.Name = Encoding.UTF8.GetString(nameBytes);
                }

                // Extract shape
                if (info.Shape != IntPtr.Zero && info.ShapeLen > 0)
                {
                    var shape = new long[info.ShapeLen];
                    Marshal.Copy(info.Shape, shape, 0, (int)info.ShapeLen);
                    tensorInfo.Shape = shape;
                }

                // Set element type
                tensorInfo.ElementType = (TensorElementType)info.ElementType;

                tensors.Add(tensorInfo);

                // Free the info
                CudaRsOvTensorInfo* pInfo = &info;
                SdkNative.OpenVinoFreeTensorInfo(pInfo);
            }
            catch
            {
                // Make sure to free on exception
                CudaRsOvTensorInfo* pInfo = &info;
                SdkNative.OpenVinoFreeTensorInfo(pInfo);
                throw;
            }
        }

        return tensors.ToArray();
    }

    private static void ThrowIfFailed(CudaRsResult result)
    {
        if (result != CudaRsResult.Success)
        {
            throw new InvalidOperationException($"OpenVINO operation failed with result: {result}");
        }
    }

    public void Dispose()
    {
        if (_ownsHub)
            _hub.Dispose();
    }
}
