using System;
using System.Text;
using System.Text.Json;
using CudaRS.Native;

namespace CudaRS.Core;

public static unsafe class SdkCheck
{
    public static void ThrowIfError(SdkErr err)
    {
        if (err == SdkErr.Ok)
            return;

        var (message, missingFile, searchedPaths, suggestion) = GetErrorDetails();
        throw new SdkException(err, message, missingFile, searchedPaths, suggestion);
    }

    public static string GetLastErrorMessage()
    {
        var res = SdkNative.SdkLastErrorMessageUtf8(out var ptr, out var len);
        if (res != SdkErr.Ok || ptr == IntPtr.Zero || len == 0)
            return string.Empty;

        var span = new ReadOnlySpan<byte>((void*)ptr, checked((int)len));
        return Encoding.UTF8.GetString(span);
    }

    public static (string? message, string? missingFile, string[]? searchedPaths, string? suggestion) GetErrorDetails()
    {
        var res = SdkNative.SdkGetErrorDetail(out var detail);
        if (res != SdkErr.Ok)
            return (GetLastErrorMessage(), null, null, null);

        string? message = null;
        string? missingFile = null;
        string[]? searchedPaths = null;
        string? suggestion = null;

        if (detail.MessagePtr != IntPtr.Zero && detail.MessageLen > 0)
        {
            var span = new ReadOnlySpan<byte>((void*)detail.MessagePtr, checked((int)detail.MessageLen));
            message = Encoding.UTF8.GetString(span);
        }

        if (detail.MissingFilePtr != IntPtr.Zero && detail.MissingFileLen > 0)
        {
            var span = new ReadOnlySpan<byte>((void*)detail.MissingFilePtr, checked((int)detail.MissingFileLen));
            missingFile = Encoding.UTF8.GetString(span);
        }

        if (detail.SearchPathsPtr != IntPtr.Zero && detail.SearchPathsLen > 0)
        {
            var span = new ReadOnlySpan<byte>((void*)detail.SearchPathsPtr, checked((int)detail.SearchPathsLen));
            var json = Encoding.UTF8.GetString(span);
            try
            {
                searchedPaths = JsonSerializer.Deserialize<string[]>(json);
            }
            catch
            {
                // Ignore JSON parsing errors
            }
        }

        if (detail.SuggestionPtr != IntPtr.Zero && detail.SuggestionLen > 0)
        {
            var span = new ReadOnlySpan<byte>((void*)detail.SuggestionPtr, checked((int)detail.SuggestionLen));
            suggestion = Encoding.UTF8.GetString(span);
        }

        return (message ?? GetLastErrorMessage(), missingFile, searchedPaths, suggestion);
    }
}
