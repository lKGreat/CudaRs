using System;
using CudaRS.Native;

namespace CudaRS.Core;

public sealed class SdkException : Exception
{
    public SdkException(SdkErr errorCode, string? message)
        : base(message ?? errorCode.ToString())
    {
        ErrorCode = errorCode;
    }

    public SdkErr ErrorCode { get; }
}
