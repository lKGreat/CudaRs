using System;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;

namespace CudaRS.Native;

public static unsafe partial class SdkNative
{
    private const string LibraryName = "cudars_ffi";

    static SdkNative()
    {
        if (string.Equals(Environment.GetEnvironmentVariable("CUDARS_DIAG"), "1", StringComparison.Ordinal))
            Console.Error.WriteLine("[CudaRS] Initializing DllImportResolver for cudars_ffi (sdk_*)");

        NativeLibrary.SetDllImportResolver(typeof(SdkNative).Assembly, ResolveCudaRs);
    }

    private static IntPtr ResolveCudaRs(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (!string.Equals(libraryName, LibraryName, StringComparison.Ordinal))
            return IntPtr.Zero;

        if (string.Equals(Environment.GetEnvironmentVariable("CUDARS_DIAG"), "1", StringComparison.Ordinal))
            Console.Error.WriteLine($"[CudaRS] Resolving native library: {libraryName}");

        IntPtr nativeHandle;

        var explicitPath = Environment.GetEnvironmentVariable("CUDARS_FFI_PATH");
        if (!string.IsNullOrWhiteSpace(explicitPath) && NativeLibrary.TryLoad(explicitPath, out nativeHandle))
            return nativeHandle;

        var explicitDir = Environment.GetEnvironmentVariable("CUDARS_FFI_DIR");
        if (!string.IsNullOrWhiteSpace(explicitDir))
        {
            var path = Path.Combine(explicitDir, "cudars_ffi.dll");
            if (NativeLibrary.TryLoad(path, out nativeHandle))
                return nativeHandle;
        }

        var baseDir = AppContext.BaseDirectory;
        if (!string.IsNullOrWhiteSpace(baseDir))
        {
            var path = Path.Combine(baseDir, "cudars_ffi.dll");
            if (NativeLibrary.TryLoad(path, out nativeHandle))
                return nativeHandle;

            path = Path.Combine(baseDir, "runtimes", "win-x64", "native", "cudars_ffi.dll");
            if (NativeLibrary.TryLoad(path, out nativeHandle))
                return nativeHandle;
        }

        var assemblyDir = Path.GetDirectoryName(assembly.Location);
        if (!string.IsNullOrWhiteSpace(assemblyDir))
        {
            var path = Path.Combine(assemblyDir, "cudars_ffi.dll");
            if (NativeLibrary.TryLoad(path, out nativeHandle))
                return nativeHandle;
        }

        if (NativeLibrary.TryLoad(libraryName, assembly, searchPath, out var handle2))
            return handle2;

        return IntPtr.Zero;
    }

    // ========================================================================
    // Library Info
    // ========================================================================

    [LibraryImport(LibraryName, EntryPoint = "sdk_abi_version")]
    public static partial uint SdkAbiVersion();

    [LibraryImport(LibraryName, EntryPoint = "sdk_version_string")]
    public static partial IntPtr SdkVersionString();

    [LibraryImport(LibraryName, EntryPoint = "sdk_version_string_len")]
    public static partial SdkErr SdkVersionStringLen(out nuint len);

    [LibraryImport(LibraryName, EntryPoint = "sdk_version_string_write")]
    public static partial SdkErr SdkVersionStringWrite(byte* buffer, nuint cap, out nuint written);

    [LibraryImport(LibraryName, EntryPoint = "sdk_last_error_message_utf8")]
    public static partial SdkErr SdkLastErrorMessageUtf8(out IntPtr ptr, out nuint len);

    // ========================================================================
    // Model Manager
    // ========================================================================

    [LibraryImport(LibraryName, EntryPoint = "sdk_model_manager_create")]
    public static partial SdkErr ModelManagerCreate(out ulong handle);

    [LibraryImport(LibraryName, EntryPoint = "sdk_model_manager_destroy")]
    public static partial SdkErr ModelManagerDestroy(ulong handle);

    [LibraryImport(LibraryName, EntryPoint = "sdk_model_manager_load_model")]
    public static partial SdkErr ModelManagerLoadModel(ulong managerHandle, in SdkModelSpec spec, out ulong modelHandle);

    [LibraryImport(LibraryName, EntryPoint = "sdk_model_create_pipeline")]
    public static partial SdkErr ModelCreatePipeline(ulong modelHandle, in SdkPipelineSpec spec, out ulong pipelineHandle);

    // ========================================================================
    // Pipelines
    // ========================================================================

    [LibraryImport(LibraryName, EntryPoint = "sdk_pipeline_destroy")]
    public static partial SdkErr PipelineDestroy(ulong pipelineHandle);

    [LibraryImport(LibraryName, EntryPoint = "sdk_yolo_pipeline_run_image")]
    public static partial SdkErr YoloPipelineRunImage(ulong pipelineHandle, byte* data, nuint len, out SdkYoloPreprocessMeta meta);

    [LibraryImport(LibraryName, EntryPoint = "sdk_pipeline_get_output_count")]
    public static partial SdkErr PipelineGetOutputCount(ulong pipelineHandle, out nuint count);

    [LibraryImport(LibraryName, EntryPoint = "sdk_pipeline_get_output_shape_len")]
    public static partial SdkErr PipelineGetOutputShapeLen(ulong pipelineHandle, nuint index, out nuint len);

    [LibraryImport(LibraryName, EntryPoint = "sdk_pipeline_get_output_shape_write")]
    public static partial SdkErr PipelineGetOutputShapeWrite(ulong pipelineHandle, nuint index, long* dst, nuint cap, out nuint written);

    [LibraryImport(LibraryName, EntryPoint = "sdk_pipeline_get_output_bytes")]
    public static partial SdkErr PipelineGetOutputBytes(ulong pipelineHandle, nuint index, out nuint bytes);

    [LibraryImport(LibraryName, EntryPoint = "sdk_pipeline_read_output")]
    public static partial SdkErr PipelineReadOutput(ulong pipelineHandle, nuint index, byte* dst, nuint cap, out nuint written);

    // ========================================================================
    // PaddleOCR
    // ========================================================================

    [LibraryImport(LibraryName, EntryPoint = "sdk_ocr_pipeline_run_image")]
    public static partial SdkErr OcrPipelineRunImage(ulong pipelineHandle, byte* data, nuint len);

    [LibraryImport(LibraryName, EntryPoint = "sdk_ocr_pipeline_get_line_count")]
    public static partial SdkErr OcrPipelineGetLineCount(ulong pipelineHandle, out nuint count);

    [LibraryImport(LibraryName, EntryPoint = "sdk_ocr_pipeline_write_lines")]
    public static partial SdkErr OcrPipelineWriteLines(ulong pipelineHandle, SdkOcrLine* dst, nuint cap, out nuint written);

    [LibraryImport(LibraryName, EntryPoint = "sdk_ocr_pipeline_get_text_bytes")]
    public static partial SdkErr OcrPipelineGetTextBytes(ulong pipelineHandle, out nuint bytes);

    [LibraryImport(LibraryName, EntryPoint = "sdk_ocr_pipeline_write_text")]
    public static partial SdkErr OcrPipelineWriteText(ulong pipelineHandle, byte* dst, nuint cap, out nuint written);

    [LibraryImport(LibraryName, EntryPoint = "sdk_ocr_pipeline_get_struct_json_bytes")]
    public static partial SdkErr OcrPipelineGetStructJsonBytes(ulong pipelineHandle, out nuint bytes);

    [LibraryImport(LibraryName, EntryPoint = "sdk_ocr_pipeline_write_struct_json")]
    public static partial SdkErr OcrPipelineWriteStructJson(ulong pipelineHandle, byte* dst, nuint cap, out nuint written);
}
