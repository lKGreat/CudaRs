using System;
using System.Collections.Generic;
using System.Text;
using CudaRS.Core;
using CudaRS.Interop;
using CudaRS.Native;

namespace CudaRS.Ocr;

public sealed unsafe class OcrPipeline : IDisposable
{
    private readonly PipelineHandle _handle;
    private bool _disposed;

    internal OcrPipeline(PipelineHandle handle)
    {
        _handle = handle;
    }

    public OcrResult RunImage(byte[] imageBytes)
    {
        if (imageBytes == null || imageBytes.Length == 0)
            throw new ArgumentException("Image bytes are required.", nameof(imageBytes));

        fixed (byte* data = imageBytes)
        {
            SdkCheck.ThrowIfError(SdkNative.OcrPipelineRunImage(_handle.Value, data, (nuint)imageBytes.Length));
        }

        SdkCheck.ThrowIfError(SdkNative.OcrPipelineGetLineCount(_handle.Value, out var count));
        var lines = new List<OcrLine>((int)count);

        byte[] textBytes = Array.Empty<byte>();
        SdkCheck.ThrowIfError(SdkNative.OcrPipelineGetTextBytes(_handle.Value, out var textLen));
        if (textLen > 0)
        {
            textBytes = new byte[(int)textLen];
            fixed (byte* textPtr = textBytes)
            {
                SdkCheck.ThrowIfError(SdkNative.OcrPipelineWriteText(_handle.Value, textPtr, textLen, out var written));
                if (written < textLen)
                    Array.Resize(ref textBytes, (int)written);
            }
        }

        if (count > 0)
        {
            var nativeLines = new SdkOcrLine[(int)count];
            fixed (SdkOcrLine* linePtr = nativeLines)
            {
                SdkCheck.ThrowIfError(SdkNative.OcrPipelineWriteLines(_handle.Value, linePtr, count, out var written));
                for (var i = 0; i < (int)written; i++)
                {
                    var line = nativeLines[i];
                    var points = new float[8];
                    for (var p = 0; p < 8; p++)
                        points[p] = line.Points[p];

                    var text = string.Empty;
                    if (line.TextLen > 0 && textBytes.Length >= line.TextOffset + line.TextLen)
                    {
                        text = Encoding.UTF8.GetString(textBytes, (int)line.TextOffset, (int)line.TextLen);
                    }

                    lines.Add(new OcrLine
                    {
                        Points = points,
                        Score = line.Score,
                        ClassLabel = line.ClassLabel,
                        ClassScore = line.ClassScore,
                        Text = text,
                    });
                }
            }
        }

        string? structJson = null;
        SdkCheck.ThrowIfError(SdkNative.OcrPipelineGetStructJsonBytes(_handle.Value, out var jsonBytes));
        if (jsonBytes > 0)
        {
            var buffer = new byte[(int)jsonBytes];
            fixed (byte* jsonPtr = buffer)
            {
                SdkCheck.ThrowIfError(SdkNative.OcrPipelineWriteStructJson(_handle.Value, jsonPtr, jsonBytes, out var written));
                structJson = Encoding.UTF8.GetString(buffer, 0, (int)written);
            }
        }

        return new OcrResult { Lines = lines, StructJson = structJson };
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _handle.Dispose();
        _disposed = true;
    }
}
