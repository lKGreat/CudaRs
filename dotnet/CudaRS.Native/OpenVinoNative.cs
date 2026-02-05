using System.Runtime.InteropServices;

namespace CudaRS.Native;

public static unsafe partial class SdkNative
{
    [LibraryImport("cudars_ffi", EntryPoint = "cudars_ov_load")]
    public static partial CudaRsResult OpenVinoLoad(byte* modelPath, in CudaRsOvConfig config, out ulong handle);

    [LibraryImport("cudars_ffi", EntryPoint = "cudars_ov_load_v2")]
    public static partial CudaRsResult OpenVinoLoadV2(byte* modelPath, in CudaRsOvConfigV2 config, out ulong handle);

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

    [LibraryImport("cudars_ffi", EntryPoint = "cudars_ov_get_input_count")]
    public static partial CudaRsResult OpenVinoGetInputCount(ulong handle, out ulong count);

    [LibraryImport("cudars_ffi", EntryPoint = "cudars_ov_get_input_info")]
    public static partial CudaRsResult OpenVinoGetInputInfo(ulong handle, ulong index, out CudaRsOvTensorInfo info);

    [LibraryImport("cudars_ffi", EntryPoint = "cudars_ov_get_output_count")]
    public static partial CudaRsResult OpenVinoGetOutputCount(ulong handle, out ulong count);

    [LibraryImport("cudars_ffi", EntryPoint = "cudars_ov_get_output_info")]
    public static partial CudaRsResult OpenVinoGetOutputInfo(ulong handle, ulong index, out CudaRsOvTensorInfo info);

    [LibraryImport("cudars_ffi", EntryPoint = "cudars_ov_free_tensor_info")]
    public static partial CudaRsResult OpenVinoFreeTensorInfo(CudaRsOvTensorInfo* info);

    [LibraryImport("cudars_ffi", EntryPoint = "cudars_ov_reshape_fixed")]
    public static partial CudaRsResult OpenVinoReshapeFixed(ulong handle, long* shape, ulong shapeLen);

    [LibraryImport("cudars_ffi", EntryPoint = "cudars_ov_reshape_dynamic")]
    public static partial CudaRsResult OpenVinoReshapeDynamic(ulong handle, in CudaRsOvPartialShapeArray partialShape);

    [LibraryImport("cudars_ffi", EntryPoint = "cudars_ov_run_batch")]
    public static partial CudaRsResult OpenVinoRunBatch(
        ulong handle,
        float** batchInputs,
        ulong* batchInputLens,
        ulong batchSize,
        long* singleShape,
        ulong singleShapeLen,
        out CudaRsOvTensor** outBatchTensors,
        out ulong* outBatchCounts);

    [LibraryImport("cudars_ffi", EntryPoint = "cudars_ov_free_batch_tensors")]
    public static partial CudaRsResult OpenVinoFreeBatchTensors(CudaRsOvTensor** batchTensors, ulong* batchCounts, ulong batchSize);

    [LibraryImport("cudars_ffi", EntryPoint = "cudars_ov_preprocess_create")]
    public static partial CudaRsResult OpenVinoPreprocessCreate(ulong modelHandle, out ulong preprocessHandle);

    [LibraryImport("cudars_ffi", EntryPoint = "cudars_ov_preprocess_free")]
    public static partial CudaRsResult OpenVinoPreprocessFree(ulong preprocessHandle);

    [LibraryImport("cudars_ffi", EntryPoint = "cudars_ov_preprocess_set_input_format")]
    public static partial CudaRsResult OpenVinoPreprocessSetInputFormat(
        ulong preprocessHandle,
        ulong inputIndex,
        int elementType,
        byte* tensorLayout);

    [LibraryImport("cudars_ffi", EntryPoint = "cudars_ov_preprocess_set_model_layout")]
    public static partial CudaRsResult OpenVinoPreprocessSetModelLayout(
        ulong preprocessHandle,
        ulong inputIndex,
        byte* modelLayout);

    [LibraryImport("cudars_ffi", EntryPoint = "cudars_ov_preprocess_add_resize")]
    public static partial CudaRsResult OpenVinoPreprocessAddResize(
        ulong preprocessHandle,
        ulong inputIndex,
        int resizeAlgorithm);

    [LibraryImport("cudars_ffi", EntryPoint = "cudars_ov_preprocess_build")]
    public static partial CudaRsResult OpenVinoPreprocessBuild(
        ulong preprocessHandle,
        ulong originalModelHandle,
        out ulong newModelHandle);

    [LibraryImport("cudars_ffi", EntryPoint = "cudars_ov_get_profiling_info")]
    public static partial CudaRsResult OpenVinoGetProfilingInfo(
        ulong modelHandle,
        out byte* jsonPtr,
        out ulong jsonLen);

    [LibraryImport("cudars_ffi", EntryPoint = "cudars_ov_free_profiling_info")]
    public static partial CudaRsResult OpenVinoFreeProfilingInfo(
        byte* jsonPtr,
        ulong jsonLen);
}
