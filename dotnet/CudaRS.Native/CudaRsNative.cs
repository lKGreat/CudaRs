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
