using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using CudaRS.Core;
using CudaRS.Native;

namespace CudaRS;

/// <summary>
/// Helper to run TensorRT trtexec.exe via Rust FFI.
/// </summary>
public static class TensorRtExec
{
    public static int Run(string trtexecPath, IEnumerable<string>? args, string? workingDirectory = null)
    {
        if (string.IsNullOrWhiteSpace(trtexecPath))
            throw new ArgumentException("trtexec path is required.", nameof(trtexecPath));

        var argArray = args?.ToArray() ?? Array.Empty<string>();

        unsafe
        {
            var ptrs = new IntPtr[argArray.Length];
            var argPtrs = stackalloc byte*[argArray.Length];

            try
            {
                for (var i = 0; i < argArray.Length; i++)
                {
                    var arg = argArray[i] ?? string.Empty;
                    ptrs[i] = Marshal.StringToCoTaskMemUTF8(arg);
                    argPtrs[i] = (byte*)ptrs[i];
                }

                CudaCheck.ThrowIfError(CudaRsNative.TrtExecRun(
                    trtexecPath,
                    argPtrs,
                    (ulong)argArray.Length,
                    workingDirectory,
                    out var exitCode));

                return exitCode;
            }
            finally
            {
                foreach (var ptr in ptrs)
                {
                    if (ptr != IntPtr.Zero)
                        Marshal.FreeCoTaskMem(ptr);
                }
            }
        }
    }

    public static int BuildEngine(
        string trtexecPath,
        string onnxPath,
        string enginePath,
        int workspaceMb = 1024,
        bool fp16 = true,
        IEnumerable<string>? extraArgs = null)
    {
        if (string.IsNullOrWhiteSpace(onnxPath))
            throw new ArgumentException("ONNX path is required.", nameof(onnxPath));
        if (string.IsNullOrWhiteSpace(enginePath))
            throw new ArgumentException("Engine path is required.", nameof(enginePath));

        var args = new List<string>
        {
            $"--onnx={onnxPath}",
            $"--saveEngine={enginePath}",
            $"--memPoolSize=workspace:{Math.Max(16, workspaceMb)}",
        };

        if (fp16)
            args.Add("--fp16");

        if (extraArgs != null)
            args.AddRange(extraArgs);

        var workdir = System.IO.Path.GetDirectoryName(onnxPath);
        return Run(trtexecPath, args, workdir);
    }
}
