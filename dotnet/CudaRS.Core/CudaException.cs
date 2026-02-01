using System;
using CudaRS.Native;

namespace CudaRS.Core;

/// <summary>
/// CUDA exception thrown when a CUDA operation fails.
/// </summary>
public class CudaException : Exception
{
    public CudaRsResult? ErrorCode { get; }

    public CudaException(CudaRsResult errorCode)
        : base(GetErrorMessage(errorCode))
    {
        ErrorCode = errorCode;
    }

    public CudaException(CudaRsResult errorCode, string message)
        : base(message)
    {
        ErrorCode = errorCode;
    }

    public CudaException(string message)
        : base(message)
    {
        ErrorCode = null;
    }

    public CudaException(string message, Exception innerException)
        : base(message, innerException)
    {
        ErrorCode = null;
    }

    private static string GetErrorMessage(CudaRsResult errorCode)
    {
        return errorCode switch
        {
            CudaRsResult.Success => "Success",
            CudaRsResult.ErrorInvalidValue => "Invalid value",
            CudaRsResult.ErrorOutOfMemory => "Out of memory",
            CudaRsResult.ErrorNotInitialized => "Not initialized",
            CudaRsResult.ErrorInvalidHandle => "Invalid handle",
            CudaRsResult.ErrorNotSupported => "Not supported",
            _ => $"Unknown error: {errorCode}",
        };
    }
}

/// <summary>
/// Helper methods for CUDA operations.
/// </summary>
public static class CudaCheck
{
    public static void ThrowIfError(CudaRsResult result)
    {
        if (result != CudaRsResult.Success)
        {
            throw new CudaException(result);
        }
    }
}
