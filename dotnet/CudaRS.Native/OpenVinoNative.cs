using System.Runtime.InteropServices;

namespace CudaRS.Native;

public static unsafe partial class SdkNative
{
    [LibraryImport("cudars_ffi", EntryPoint = "cudars_ov_load")]
    public static partial CudaRsResult OpenVinoLoad(byte* modelPath, in CudaRsOvConfig config, out ulong handle);

    [LibraryImport("cudars_ffi", EntryPoint = "cudars_ov_destroy")]
    public static partial CudaRsResult OpenVinoDestroy(ulong handle);

    [LibraryImport("cudars_ffi", EntryPoint = "cudars_ov_async_queue_submit")]
    public static partial CudaRsResult OpenVinoAsyncQueueSubmit(
        ulong handle,
        float* input,
        ulong inputLen,
        long* shape,
        ulong shapeLen,
        out int requestId);

    [LibraryImport("cudars_ffi", EntryPoint = "cudars_ov_async_queue_wait")]
    public static partial CudaRsResult OpenVinoAsyncQueueWait(
        ulong handle,
        int requestId,
        out CudaRsOvTensor* tensors,
        out ulong count);

    [LibraryImport("cudars_ffi", EntryPoint = "cudars_ov_free_tensors")]
    public static partial CudaRsResult OpenVinoFreeTensors(CudaRsOvTensor* tensors, ulong count);
}
