using System;
using System.Runtime.InteropServices;

namespace CudaRS.Native;

/// <summary>
/// Result codes for CudaRS operations.
/// </summary>
public enum CudaRsResult : int
{
    Success = 0,
    ErrorInvalidValue = 1,
    ErrorOutOfMemory = 2,
    ErrorNotInitialized = 3,
    ErrorInvalidHandle = 4,
    ErrorNotSupported = 5,
    ErrorUnknown = 999,
}

/// <summary>
/// Memory information structure.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct CudaRsMemoryInfo
{
    public ulong Total;
    public ulong Free;
    public ulong Used;
}

/// <summary>
/// Utilization rates structure.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct CudaRsUtilizationRates
{
    public uint Gpu;
    public uint Memory;
}

/// <summary>
/// Memory quota configuration for a GPU memory pool.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct CudaRsMemoryQuota
{
    public ulong MaxBytes;
    public ulong PreallocateBytes;
    [MarshalAs(UnmanagedType.I1)]
    public bool AllowFallbackToShared;
    public CudaRsOomPolicy OomPolicy;
}

public enum CudaRsOomPolicy : int
{
    Fail = 0,
    Wait = 1,
    Skip = 2,
    FallbackCpu = 3,
}

/// <summary>
/// Memory pool statistics.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct CudaRsMemoryPoolStats
{
    public ulong Quota;
    public ulong Used;
    public ulong Peak;
    public uint AllocationCount;
    public float FragmentationRate;
}

/// <summary>
/// GPU memory statistics.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct CudaRsGpuMemoryStats
{
    public int DeviceId;
    public ulong Total;
    public ulong Free;
    public ulong Used;
    public float FragmentationRate;
}

/// <summary>
/// Tensor output from ONNX Runtime FFI.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct CudaRsTensor
{
    public IntPtr Data;
    public ulong DataLen;
    public IntPtr Shape;
    public ulong ShapeLen;
}

/// <summary>
/// TensorRT build configuration.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct CudaRsTrtBuildConfig
{
    public int Fp16Enabled;
    public int Int8Enabled;
    public int MaxBatchSize;
    public int WorkspaceSizeMb;
    public int DlaCore; // -1 = disabled
}

/// <summary>
/// TensorRT tensor output.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct CudaRsTrtTensor
{
    public IntPtr Data;
    public ulong DataLen;
    public IntPtr Shape;
    public ulong ShapeLen;
}

/// <summary>
/// TorchScript tensor output.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct CudaRsTorchTensor
{
    public IntPtr Data;
    public ulong DataLen;
    public IntPtr Shape;
    public ulong ShapeLen;
}

/// <summary>
/// TorchScript multi-input descriptor.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct CudaRsTorchInputDesc
{
    public IntPtr Data;
    public ulong DataLen;
    public IntPtr Shape;
    public ulong ShapeLen;
}

/// <summary>
/// OpenVINO device type.
/// </summary>
public enum CudaRsOvDevice : int
{
    Cpu = 0,
    Gpu = 1,
    GpuIndex = 2,
    Npu = 3,
    Auto = 4,
}

/// <summary>
/// OpenVINO inference configuration.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct CudaRsOvConfig
{
    public CudaRsOvDevice Device;
    public int DeviceIndex;
    public int NumStreams;
    public int EnableProfiling;
}

/// <summary>
/// OpenVINO tensor output.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct CudaRsOvTensor
{
    public IntPtr Data;
    public ulong DataLen;
    public IntPtr Shape;
    public ulong ShapeLen;
}

/// <summary>
/// Native P/Invoke bindings for CudaRS.
/// </summary>
public static unsafe partial class CudaRsNative
{
    private const string LibraryName = "cudars_ffi";

    // ========================================================================
    // Library Info
    // ========================================================================

    [LibraryImport(LibraryName, EntryPoint = "cudars_get_version")]
    public static partial IntPtr GetVersion();

    [LibraryImport(LibraryName, EntryPoint = "cudars_get_error_string")]
    public static partial IntPtr GetErrorString(CudaRsResult result);

    // ========================================================================
    // Device Management
    // ========================================================================

    [LibraryImport(LibraryName, EntryPoint = "cudars_device_get_count")]
    public static partial CudaRsResult DeviceGetCount(out int count);

    [LibraryImport(LibraryName, EntryPoint = "cudars_device_set")]
    public static partial CudaRsResult DeviceSet(int device);

    [LibraryImport(LibraryName, EntryPoint = "cudars_device_get")]
    public static partial CudaRsResult DeviceGet(out int device);

    [LibraryImport(LibraryName, EntryPoint = "cudars_device_synchronize")]
    public static partial CudaRsResult DeviceSynchronize();

    [LibraryImport(LibraryName, EntryPoint = "cudars_device_reset")]
    public static partial CudaRsResult DeviceReset();

    // ========================================================================
    // Stream Management
    // ========================================================================

    [LibraryImport(LibraryName, EntryPoint = "cudars_stream_create")]
    public static partial CudaRsResult StreamCreate(out ulong stream);

    [LibraryImport(LibraryName, EntryPoint = "cudars_stream_destroy")]
    public static partial CudaRsResult StreamDestroy(ulong stream);

    [LibraryImport(LibraryName, EntryPoint = "cudars_stream_synchronize")]
    public static partial CudaRsResult StreamSynchronize(ulong stream);

    // ========================================================================
    // Event Management
    // ========================================================================

    [LibraryImport(LibraryName, EntryPoint = "cudars_event_create")]
    public static partial CudaRsResult EventCreate(out ulong eventHandle);

    [LibraryImport(LibraryName, EntryPoint = "cudars_event_destroy")]
    public static partial CudaRsResult EventDestroy(ulong eventHandle);

    [LibraryImport(LibraryName, EntryPoint = "cudars_event_record")]
    public static partial CudaRsResult EventRecord(ulong eventHandle, ulong stream);

    [LibraryImport(LibraryName, EntryPoint = "cudars_event_synchronize")]
    public static partial CudaRsResult EventSynchronize(ulong eventHandle);

    [LibraryImport(LibraryName, EntryPoint = "cudars_event_elapsed_time")]
    public static partial CudaRsResult EventElapsedTime(out float ms, ulong start, ulong end);

    // ========================================================================
    // Memory Management
    // ========================================================================

    [LibraryImport(LibraryName, EntryPoint = "cudars_malloc")]
    public static partial CudaRsResult Malloc(out ulong buffer, nuint size);

    [LibraryImport(LibraryName, EntryPoint = "cudars_free")]
    public static partial CudaRsResult Free(ulong buffer);

    [LibraryImport(LibraryName, EntryPoint = "cudars_memcpy_htod")]
    public static partial CudaRsResult MemcpyHtoD(ulong buffer, void* src, nuint size);

    [LibraryImport(LibraryName, EntryPoint = "cudars_memcpy_dtoh")]
    public static partial CudaRsResult MemcpyDtoH(void* dst, ulong buffer, nuint size);

    [LibraryImport(LibraryName, EntryPoint = "cudars_memset")]
    public static partial CudaRsResult Memset(ulong buffer, int value);

    // ========================================================================
    // Memory Pool Management
    // ========================================================================

    [LibraryImport(LibraryName, EntryPoint = "cudars_memory_pool_create", StringMarshalling = StringMarshalling.Utf8)]
    public static partial CudaRsResult MemoryPoolCreate(string poolId, CudaRsMemoryQuota quota, out ulong handle);

    [LibraryImport(LibraryName, EntryPoint = "cudars_memory_pool_create_with_device", StringMarshalling = StringMarshalling.Utf8)]
    public static partial CudaRsResult MemoryPoolCreateWithDevice(string poolId, int deviceId, CudaRsMemoryQuota quota, out ulong handle);

    [LibraryImport(LibraryName, EntryPoint = "cudars_memory_pool_destroy")]
    public static partial CudaRsResult MemoryPoolDestroy(ulong handle);

    [LibraryImport(LibraryName, EntryPoint = "cudars_memory_pool_allocate")]
    public static partial CudaRsResult MemoryPoolAllocate(ulong handle, ulong size, out ulong ptr);

    [LibraryImport(LibraryName, EntryPoint = "cudars_memory_pool_free")]
    public static partial CudaRsResult MemoryPoolFree(ulong handle, ulong ptr);

    [LibraryImport(LibraryName, EntryPoint = "cudars_memory_pool_get_stats")]
    public static partial CudaRsResult MemoryPoolGetStats(ulong handle, out CudaRsMemoryPoolStats stats);

    [LibraryImport(LibraryName, EntryPoint = "cudars_memory_pool_defragment")]
    public static partial CudaRsResult MemoryPoolDefragment(ulong handle);

    [LibraryImport(LibraryName, EntryPoint = "cudars_gpu_get_memory_stats")]
    public static partial CudaRsResult GpuGetMemoryStats(int deviceId, out CudaRsGpuMemoryStats stats);

    // ========================================================================
    // ONNX Runtime
    // ========================================================================

    [LibraryImport(LibraryName, EntryPoint = "cudars_onnx_create", StringMarshalling = StringMarshalling.Utf8)]
    public static partial CudaRsResult OnnxCreate(string modelPath, int deviceId, out ulong handle);

    [LibraryImport(LibraryName, EntryPoint = "cudars_onnx_destroy")]
    public static partial CudaRsResult OnnxDestroy(ulong handle);

    [LibraryImport(LibraryName, EntryPoint = "cudars_onnx_run")]
    public static unsafe partial CudaRsResult OnnxRun(
        ulong handle,
        float* input,
        ulong inputLen,
        long* shape,
        ulong shapeLen,
        out IntPtr tensors,
        out ulong tensorCount);

    [LibraryImport(LibraryName, EntryPoint = "cudars_onnx_free_tensors")]
    public static partial CudaRsResult OnnxFreeTensors(IntPtr tensors, ulong tensorCount);

    // ========================================================================
    // TensorRT
    // ========================================================================

    [LibraryImport(LibraryName, EntryPoint = "cudars_trt_build_engine", StringMarshalling = StringMarshalling.Utf8)]
    public static partial CudaRsResult TrtBuildEngine(
        string onnxPath,
        int deviceId,
        in CudaRsTrtBuildConfig config,
        out ulong handle);

    [LibraryImport(LibraryName, EntryPoint = "cudars_trt_load_engine", StringMarshalling = StringMarshalling.Utf8)]
    public static partial CudaRsResult TrtLoadEngine(string enginePath, int deviceId, out ulong handle);

    [LibraryImport(LibraryName, EntryPoint = "cudars_trt_save_engine", StringMarshalling = StringMarshalling.Utf8)]
    public static partial CudaRsResult TrtSaveEngine(ulong handle, string path);

    [LibraryImport(LibraryName, EntryPoint = "cudars_trt_run")]
    public static unsafe partial CudaRsResult TrtRun(
        ulong handle,
        float* input,
        ulong inputLen,
        out IntPtr tensors,
        out ulong tensorCount);

    [LibraryImport(LibraryName, EntryPoint = "cudars_trt_get_input_info")]
    public static unsafe partial CudaRsResult TrtGetInputInfo(
        ulong handle,
        int index,
        long* shape,
        out int shapeLen,
        int maxShapeLen);

    [LibraryImport(LibraryName, EntryPoint = "cudars_trt_get_output_info")]
    public static unsafe partial CudaRsResult TrtGetOutputInfo(
        ulong handle,
        int index,
        long* shape,
        out int shapeLen,
        int maxShapeLen);

    [LibraryImport(LibraryName, EntryPoint = "cudars_trt_destroy")]
    public static partial CudaRsResult TrtDestroy(ulong handle);

    [LibraryImport(LibraryName, EntryPoint = "cudars_trt_free_tensors")]
    public static partial CudaRsResult TrtFreeTensors(IntPtr tensors, ulong tensorCount);

    // ========================================================================
    // TorchScript
    // ========================================================================

    [LibraryImport(LibraryName, EntryPoint = "cudars_torch_cuda_available")]
    public static partial int TorchCudaAvailable();

    [LibraryImport(LibraryName, EntryPoint = "cudars_torch_cuda_device_count")]
    public static partial int TorchCudaDeviceCount();

    [LibraryImport(LibraryName, EntryPoint = "cudars_torch_load", StringMarshalling = StringMarshalling.Utf8)]
    public static partial CudaRsResult TorchLoad(string modelPath, int deviceId, out ulong handle);

    [LibraryImport(LibraryName, EntryPoint = "cudars_torch_run")]
    public static unsafe partial CudaRsResult TorchRun(
        ulong handle,
        float* input,
        ulong inputLen,
        long* shape,
        ulong shapeLen,
        out IntPtr tensors,
        out ulong tensorCount);

    [LibraryImport(LibraryName, EntryPoint = "cudars_torch_run_multi")]
    public static unsafe partial CudaRsResult TorchRunMulti(
        ulong handle,
        CudaRsTorchInputDesc* inputs,
        int numInputs,
        out IntPtr tensors,
        out ulong tensorCount);

    [LibraryImport(LibraryName, EntryPoint = "cudars_torch_eval")]
    public static partial CudaRsResult TorchEval(ulong handle);

    [LibraryImport(LibraryName, EntryPoint = "cudars_torch_destroy")]
    public static partial CudaRsResult TorchDestroy(ulong handle);

    [LibraryImport(LibraryName, EntryPoint = "cudars_torch_free_tensors")]
    public static partial CudaRsResult TorchFreeTensors(IntPtr tensors, ulong tensorCount);

    // ========================================================================
    // OpenVINO
    // ========================================================================

    [LibraryImport(LibraryName, EntryPoint = "cudars_ov_get_devices")]
    public static unsafe partial CudaRsResult OvGetDevices(
        IntPtr* devices,
        out int count,
        int maxDevices);

    [LibraryImport(LibraryName, EntryPoint = "cudars_ov_load", StringMarshalling = StringMarshalling.Utf8)]
    public static partial CudaRsResult OvLoad(string modelPath, in CudaRsOvConfig config, out ulong handle);

    [LibraryImport(LibraryName, EntryPoint = "cudars_ov_run")]
    public static unsafe partial CudaRsResult OvRun(
        ulong handle,
        float* input,
        ulong inputLen,
        long* shape,
        ulong shapeLen,
        out IntPtr tensors,
        out ulong tensorCount);

    [LibraryImport(LibraryName, EntryPoint = "cudars_ov_run_async")]
    public static unsafe partial CudaRsResult OvRunAsync(
        ulong handle,
        float* input,
        ulong inputLen,
        long* shape,
        ulong shapeLen);

    [LibraryImport(LibraryName, EntryPoint = "cudars_ov_wait")]
    public static partial CudaRsResult OvWait(
        ulong handle,
        out IntPtr tensors,
        out ulong tensorCount);

    [LibraryImport(LibraryName, EntryPoint = "cudars_ov_get_input_info")]
    public static unsafe partial CudaRsResult OvGetInputInfo(
        ulong handle,
        int index,
        long* shape,
        out int shapeLen,
        int maxShapeLen);

    [LibraryImport(LibraryName, EntryPoint = "cudars_ov_get_output_info")]
    public static unsafe partial CudaRsResult OvGetOutputInfo(
        ulong handle,
        int index,
        long* shape,
        out int shapeLen,
        int maxShapeLen);

    [LibraryImport(LibraryName, EntryPoint = "cudars_ov_destroy")]
    public static partial CudaRsResult OvDestroy(ulong handle);

    [LibraryImport(LibraryName, EntryPoint = "cudars_ov_free_tensors")]
    public static partial CudaRsResult OvFreeTensors(IntPtr tensors, ulong tensorCount);

    // ========================================================================
    // Driver API
    // ========================================================================

    [LibraryImport(LibraryName, EntryPoint = "cudars_driver_init")]
    public static partial CudaRsResult DriverInit();

    [LibraryImport(LibraryName, EntryPoint = "cudars_driver_get_version")]
    public static partial CudaRsResult DriverGetVersion(out int version);

    [LibraryImport(LibraryName, EntryPoint = "cudars_context_create")]
    public static partial CudaRsResult ContextCreate(out ulong ctx, int device);

    [LibraryImport(LibraryName, EntryPoint = "cudars_context_destroy")]
    public static partial CudaRsResult ContextDestroy(ulong ctx);

    [LibraryImport(LibraryName, EntryPoint = "cudars_context_synchronize")]
    public static partial CudaRsResult ContextSynchronize(ulong ctx);

    [LibraryImport(LibraryName, EntryPoint = "cudars_module_load_data")]
    public static partial CudaRsResult ModuleLoadData(out ulong module, byte* data, nuint size);

    [LibraryImport(LibraryName, EntryPoint = "cudars_module_unload")]
    public static partial CudaRsResult ModuleUnload(ulong module);

    // ========================================================================
    // cuBLAS
    // ========================================================================

    [LibraryImport(LibraryName, EntryPoint = "cudars_cublas_create")]
    public static partial CudaRsResult CublasCreate(out ulong handle);

    [LibraryImport(LibraryName, EntryPoint = "cudars_cublas_destroy")]
    public static partial CudaRsResult CublasDestroy(ulong handle);

    [LibraryImport(LibraryName, EntryPoint = "cudars_cublas_set_stream")]
    public static partial CudaRsResult CublasSetStream(ulong handle, ulong stream);

    [LibraryImport(LibraryName, EntryPoint = "cudars_cublas_get_version")]
    public static partial CudaRsResult CublasGetVersion(ulong handle, out int version);

    // ========================================================================
    // cuFFT
    // ========================================================================

    [LibraryImport(LibraryName, EntryPoint = "cudars_fft_plan_1d_c2c")]
    public static partial CudaRsResult FftPlan1dC2C(out ulong plan, int nx);

    [LibraryImport(LibraryName, EntryPoint = "cudars_fft_plan_2d_c2c")]
    public static partial CudaRsResult FftPlan2dC2C(out ulong plan, int nx, int ny);

    [LibraryImport(LibraryName, EntryPoint = "cudars_fft_plan_1d_destroy")]
    public static partial CudaRsResult FftPlan1dDestroy(ulong plan);

    [LibraryImport(LibraryName, EntryPoint = "cudars_fft_plan_2d_destroy")]
    public static partial CudaRsResult FftPlan2dDestroy(ulong plan);

    // ========================================================================
    // cuRAND
    // ========================================================================

    public const int RngPseudoXorwow = 0;
    public const int RngPseudoMrg32k3a = 1;
    public const int RngPseudoPhilox = 2;
    public const int RngQuasiSobol32 = 3;

    [LibraryImport(LibraryName, EntryPoint = "cudars_rand_create")]
    public static partial CudaRsResult RandCreate(out ulong rng, int rngType);

    [LibraryImport(LibraryName, EntryPoint = "cudars_rand_destroy")]
    public static partial CudaRsResult RandDestroy(ulong rng);

    [LibraryImport(LibraryName, EntryPoint = "cudars_rand_set_seed")]
    public static partial CudaRsResult RandSetSeed(ulong rng, ulong seed);

    [LibraryImport(LibraryName, EntryPoint = "cudars_rand_generate_uniform")]
    public static partial CudaRsResult RandGenerateUniform(ulong rng, float* output, nuint n);

    [LibraryImport(LibraryName, EntryPoint = "cudars_rand_generate_normal")]
    public static partial CudaRsResult RandGenerateNormal(ulong rng, float* output, nuint n, float mean, float stddev);

    // ========================================================================
    // cuSPARSE
    // ========================================================================

    [LibraryImport(LibraryName, EntryPoint = "cudars_sparse_create")]
    public static partial CudaRsResult SparseCreate(out ulong handle);

    [LibraryImport(LibraryName, EntryPoint = "cudars_sparse_destroy")]
    public static partial CudaRsResult SparseDestroy(ulong handle);

    [LibraryImport(LibraryName, EntryPoint = "cudars_sparse_get_version")]
    public static partial CudaRsResult SparseGetVersion(ulong handle, out int version);

    // ========================================================================
    // cuSOLVER
    // ========================================================================

    [LibraryImport(LibraryName, EntryPoint = "cudars_solver_dn_create")]
    public static partial CudaRsResult SolverDnCreate(out ulong handle);

    [LibraryImport(LibraryName, EntryPoint = "cudars_solver_dn_destroy")]
    public static partial CudaRsResult SolverDnDestroy(ulong handle);

    // ========================================================================
    // cuDNN
    // ========================================================================

    [LibraryImport(LibraryName, EntryPoint = "cudars_cudnn_create")]
    public static partial CudaRsResult CudnnCreate(out ulong handle);

    [LibraryImport(LibraryName, EntryPoint = "cudars_cudnn_destroy")]
    public static partial CudaRsResult CudnnDestroy(ulong handle);

    [LibraryImport(LibraryName, EntryPoint = "cudars_cudnn_get_version")]
    public static partial nuint CudnnGetVersion();

    // ========================================================================
    // NVRTC
    // ========================================================================

    [LibraryImport(LibraryName, EntryPoint = "cudars_nvrtc_version")]
    public static partial CudaRsResult NvrtcVersion(out int major, out int minor);

    [LibraryImport(LibraryName, EntryPoint = "cudars_program_create", StringMarshalling = StringMarshalling.Utf8)]
    public static partial CudaRsResult ProgramCreate(out ulong program, string src, string name);

    [LibraryImport(LibraryName, EntryPoint = "cudars_program_destroy")]
    public static partial CudaRsResult ProgramDestroy(ulong program);

    [LibraryImport(LibraryName, EntryPoint = "cudars_program_compile")]
    public static partial CudaRsResult ProgramCompile(ulong program, byte** options, int numOptions);

    [LibraryImport(LibraryName, EntryPoint = "cudars_program_get_ptx_size")]
    public static partial CudaRsResult ProgramGetPtxSize(ulong program, out nuint size);

    // ========================================================================
    // NVML
    // ========================================================================

    [LibraryImport(LibraryName, EntryPoint = "cudars_nvml_init")]
    public static partial CudaRsResult NvmlInit();

    [LibraryImport(LibraryName, EntryPoint = "cudars_nvml_shutdown")]
    public static partial CudaRsResult NvmlShutdown();

    [LibraryImport(LibraryName, EntryPoint = "cudars_nvml_device_get_count")]
    public static partial CudaRsResult NvmlDeviceGetCount(out uint count);

    [LibraryImport(LibraryName, EntryPoint = "cudars_nvml_device_get_memory_info")]
    public static partial CudaRsResult NvmlDeviceGetMemoryInfo(uint index, out CudaRsMemoryInfo info);

    [LibraryImport(LibraryName, EntryPoint = "cudars_nvml_device_get_utilization_rates")]
    public static partial CudaRsResult NvmlDeviceGetUtilizationRates(uint index, out CudaRsUtilizationRates util);

    [LibraryImport(LibraryName, EntryPoint = "cudars_nvml_device_get_temperature")]
    public static partial CudaRsResult NvmlDeviceGetTemperature(uint index, out uint temp);

    [LibraryImport(LibraryName, EntryPoint = "cudars_nvml_device_get_power_usage")]
    public static partial CudaRsResult NvmlDeviceGetPowerUsage(uint index, out uint power);

    [LibraryImport(LibraryName, EntryPoint = "cudars_nvml_device_get_fan_speed")]
    public static partial CudaRsResult NvmlDeviceGetFanSpeed(uint index, out uint speed);
}
