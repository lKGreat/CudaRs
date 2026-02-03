using System;
using System.Text;
using CudaRS.Native;

namespace CudaRS.Core;

public static unsafe class SdkCheck
{
    public static void ThrowIfError(SdkErr err)
    {
        if (err == SdkErr.Ok)
            return;

        var message = GetLastErrorMessage();
        throw new SdkException(err, message);
    }

    public static string GetLastErrorMessage()
    {
        var res = SdkNative.SdkLastErrorMessageUtf8(out var ptr, out var len);
        if (res != SdkErr.Ok || ptr == IntPtr.Zero || len == 0)
            return string.Empty;

        var span = new ReadOnlySpan<byte>((void*)ptr, checked((int)len));
        return Encoding.UTF8.GetString(span);
    }
}
