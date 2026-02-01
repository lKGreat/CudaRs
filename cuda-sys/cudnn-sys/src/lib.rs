//! Raw FFI bindings to cuDNN.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use cuda_runtime_sys::cudaStream_t;
use libc::{c_char, c_int, c_void, size_t};

pub type cudnnStatus_t = c_int;
pub const CUDNN_STATUS_SUCCESS: cudnnStatus_t = 0;
pub const CUDNN_STATUS_NOT_INITIALIZED: cudnnStatus_t = 1;
pub const CUDNN_STATUS_ALLOC_FAILED: cudnnStatus_t = 2;
pub const CUDNN_STATUS_BAD_PARAM: cudnnStatus_t = 3;
pub const CUDNN_STATUS_INTERNAL_ERROR: cudnnStatus_t = 4;
pub const CUDNN_STATUS_INVALID_VALUE: cudnnStatus_t = 5;
pub const CUDNN_STATUS_ARCH_MISMATCH: cudnnStatus_t = 6;
pub const CUDNN_STATUS_MAPPING_ERROR: cudnnStatus_t = 7;
pub const CUDNN_STATUS_EXECUTION_FAILED: cudnnStatus_t = 8;
pub const CUDNN_STATUS_NOT_SUPPORTED: cudnnStatus_t = 9;
pub const CUDNN_STATUS_LICENSE_ERROR: cudnnStatus_t = 10;
pub const CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING: cudnnStatus_t = 11;
pub const CUDNN_STATUS_RUNTIME_IN_PROGRESS: cudnnStatus_t = 12;
pub const CUDNN_STATUS_RUNTIME_FP_OVERFLOW: cudnnStatus_t = 13;
pub const CUDNN_STATUS_VERSION_MISMATCH: cudnnStatus_t = 14;

pub type cudnnDataType_t = c_int;
pub const CUDNN_DATA_FLOAT: cudnnDataType_t = 0;
pub const CUDNN_DATA_DOUBLE: cudnnDataType_t = 1;
pub const CUDNN_DATA_HALF: cudnnDataType_t = 2;
pub const CUDNN_DATA_INT8: cudnnDataType_t = 3;
pub const CUDNN_DATA_INT32: cudnnDataType_t = 4;
pub const CUDNN_DATA_INT8x4: cudnnDataType_t = 5;
pub const CUDNN_DATA_UINT8: cudnnDataType_t = 6;
pub const CUDNN_DATA_UINT8x4: cudnnDataType_t = 7;
pub const CUDNN_DATA_INT8x32: cudnnDataType_t = 8;
pub const CUDNN_DATA_BFLOAT16: cudnnDataType_t = 9;
pub const CUDNN_DATA_INT64: cudnnDataType_t = 10;
pub const CUDNN_DATA_BOOLEAN: cudnnDataType_t = 11;

pub type cudnnTensorFormat_t = c_int;
pub const CUDNN_TENSOR_NCHW: cudnnTensorFormat_t = 0;
pub const CUDNN_TENSOR_NHWC: cudnnTensorFormat_t = 1;
pub const CUDNN_TENSOR_NCHW_VECT_C: cudnnTensorFormat_t = 2;

pub type cudnnConvolutionMode_t = c_int;
pub const CUDNN_CONVOLUTION: cudnnConvolutionMode_t = 0;
pub const CUDNN_CROSS_CORRELATION: cudnnConvolutionMode_t = 1;

pub type cudnnActivationMode_t = c_int;
pub const CUDNN_ACTIVATION_SIGMOID: cudnnActivationMode_t = 0;
pub const CUDNN_ACTIVATION_RELU: cudnnActivationMode_t = 1;
pub const CUDNN_ACTIVATION_TANH: cudnnActivationMode_t = 2;
pub const CUDNN_ACTIVATION_CLIPPED_RELU: cudnnActivationMode_t = 3;
pub const CUDNN_ACTIVATION_ELU: cudnnActivationMode_t = 4;
pub const CUDNN_ACTIVATION_IDENTITY: cudnnActivationMode_t = 5;
pub const CUDNN_ACTIVATION_SWISH: cudnnActivationMode_t = 6;

pub type cudnnPoolingMode_t = c_int;
pub const CUDNN_POOLING_MAX: cudnnPoolingMode_t = 0;
pub const CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING: cudnnPoolingMode_t = 1;
pub const CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING: cudnnPoolingMode_t = 2;
pub const CUDNN_POOLING_MAX_DETERMINISTIC: cudnnPoolingMode_t = 3;

pub type cudnnSoftmaxAlgorithm_t = c_int;
pub const CUDNN_SOFTMAX_FAST: cudnnSoftmaxAlgorithm_t = 0;
pub const CUDNN_SOFTMAX_ACCURATE: cudnnSoftmaxAlgorithm_t = 1;
pub const CUDNN_SOFTMAX_LOG: cudnnSoftmaxAlgorithm_t = 2;

pub type cudnnSoftmaxMode_t = c_int;
pub const CUDNN_SOFTMAX_MODE_INSTANCE: cudnnSoftmaxMode_t = 0;
pub const CUDNN_SOFTMAX_MODE_CHANNEL: cudnnSoftmaxMode_t = 1;

pub type cudnnNanPropagation_t = c_int;
pub const CUDNN_NOT_PROPAGATE_NAN: cudnnNanPropagation_t = 0;
pub const CUDNN_PROPAGATE_NAN: cudnnNanPropagation_t = 1;

pub type cudnnConvolutionFwdAlgo_t = c_int;
pub const CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM: cudnnConvolutionFwdAlgo_t = 0;
pub const CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM: cudnnConvolutionFwdAlgo_t = 1;
pub const CUDNN_CONVOLUTION_FWD_ALGO_GEMM: cudnnConvolutionFwdAlgo_t = 2;
pub const CUDNN_CONVOLUTION_FWD_ALGO_DIRECT: cudnnConvolutionFwdAlgo_t = 3;
pub const CUDNN_CONVOLUTION_FWD_ALGO_FFT: cudnnConvolutionFwdAlgo_t = 4;
pub const CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING: cudnnConvolutionFwdAlgo_t = 5;
pub const CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD: cudnnConvolutionFwdAlgo_t = 6;
pub const CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED: cudnnConvolutionFwdAlgo_t = 7;

#[repr(C)]
pub struct cudnnContext { _unused: [u8; 0] }
pub type cudnnHandle_t = *mut cudnnContext;

#[repr(C)]
pub struct cudnnTensorStruct { _unused: [u8; 0] }
pub type cudnnTensorDescriptor_t = *mut cudnnTensorStruct;

#[repr(C)]
pub struct cudnnFilterStruct { _unused: [u8; 0] }
pub type cudnnFilterDescriptor_t = *mut cudnnFilterStruct;

#[repr(C)]
pub struct cudnnConvolutionStruct { _unused: [u8; 0] }
pub type cudnnConvolutionDescriptor_t = *mut cudnnConvolutionStruct;

#[repr(C)]
pub struct cudnnPoolingStruct { _unused: [u8; 0] }
pub type cudnnPoolingDescriptor_t = *mut cudnnPoolingStruct;

#[repr(C)]
pub struct cudnnActivationStruct { _unused: [u8; 0] }
pub type cudnnActivationDescriptor_t = *mut cudnnActivationStruct;

#[repr(C)]
pub struct cudnnDropoutStruct { _unused: [u8; 0] }
pub type cudnnDropoutDescriptor_t = *mut cudnnDropoutStruct;

extern "C" {
    pub fn cudnnGetVersion() -> size_t;
    pub fn cudnnGetCudartVersion() -> size_t;
    pub fn cudnnGetErrorString(status: cudnnStatus_t) -> *const c_char;

    pub fn cudnnCreate(handle: *mut cudnnHandle_t) -> cudnnStatus_t;
    pub fn cudnnDestroy(handle: cudnnHandle_t) -> cudnnStatus_t;
    pub fn cudnnSetStream(handle: cudnnHandle_t, streamId: cudaStream_t) -> cudnnStatus_t;
    pub fn cudnnGetStream(handle: cudnnHandle_t, streamId: *mut cudaStream_t) -> cudnnStatus_t;

    // Tensor descriptor
    pub fn cudnnCreateTensorDescriptor(tensorDesc: *mut cudnnTensorDescriptor_t) -> cudnnStatus_t;
    pub fn cudnnDestroyTensorDescriptor(tensorDesc: cudnnTensorDescriptor_t) -> cudnnStatus_t;
    pub fn cudnnSetTensor4dDescriptor(
        tensorDesc: cudnnTensorDescriptor_t,
        format: cudnnTensorFormat_t,
        dataType: cudnnDataType_t,
        n: c_int,
        c: c_int,
        h: c_int,
        w: c_int,
    ) -> cudnnStatus_t;
    pub fn cudnnGetTensor4dDescriptor(
        tensorDesc: cudnnTensorDescriptor_t,
        dataType: *mut cudnnDataType_t,
        n: *mut c_int,
        c: *mut c_int,
        h: *mut c_int,
        w: *mut c_int,
        nStride: *mut c_int,
        cStride: *mut c_int,
        hStride: *mut c_int,
        wStride: *mut c_int,
    ) -> cudnnStatus_t;

    // Filter descriptor
    pub fn cudnnCreateFilterDescriptor(filterDesc: *mut cudnnFilterDescriptor_t) -> cudnnStatus_t;
    pub fn cudnnDestroyFilterDescriptor(filterDesc: cudnnFilterDescriptor_t) -> cudnnStatus_t;
    pub fn cudnnSetFilter4dDescriptor(
        filterDesc: cudnnFilterDescriptor_t,
        dataType: cudnnDataType_t,
        format: cudnnTensorFormat_t,
        k: c_int,
        c: c_int,
        h: c_int,
        w: c_int,
    ) -> cudnnStatus_t;

    // Convolution descriptor
    pub fn cudnnCreateConvolutionDescriptor(convDesc: *mut cudnnConvolutionDescriptor_t) -> cudnnStatus_t;
    pub fn cudnnDestroyConvolutionDescriptor(convDesc: cudnnConvolutionDescriptor_t) -> cudnnStatus_t;
    pub fn cudnnSetConvolution2dDescriptor(
        convDesc: cudnnConvolutionDescriptor_t,
        pad_h: c_int,
        pad_w: c_int,
        u: c_int,
        v: c_int,
        dilation_h: c_int,
        dilation_w: c_int,
        mode: cudnnConvolutionMode_t,
        computeType: cudnnDataType_t,
    ) -> cudnnStatus_t;
    pub fn cudnnSetConvolutionMathType(
        convDesc: cudnnConvolutionDescriptor_t,
        mathType: c_int,
    ) -> cudnnStatus_t;

    // Convolution forward
    pub fn cudnnGetConvolutionForwardWorkspaceSize(
        handle: cudnnHandle_t,
        xDesc: cudnnTensorDescriptor_t,
        wDesc: cudnnFilterDescriptor_t,
        convDesc: cudnnConvolutionDescriptor_t,
        yDesc: cudnnTensorDescriptor_t,
        algo: cudnnConvolutionFwdAlgo_t,
        sizeInBytes: *mut size_t,
    ) -> cudnnStatus_t;

    pub fn cudnnConvolutionForward(
        handle: cudnnHandle_t,
        alpha: *const c_void,
        xDesc: cudnnTensorDescriptor_t,
        x: *const c_void,
        wDesc: cudnnFilterDescriptor_t,
        w: *const c_void,
        convDesc: cudnnConvolutionDescriptor_t,
        algo: cudnnConvolutionFwdAlgo_t,
        workSpace: *mut c_void,
        workSpaceSizeInBytes: size_t,
        beta: *const c_void,
        yDesc: cudnnTensorDescriptor_t,
        y: *mut c_void,
    ) -> cudnnStatus_t;

    // Activation
    pub fn cudnnCreateActivationDescriptor(activationDesc: *mut cudnnActivationDescriptor_t) -> cudnnStatus_t;
    pub fn cudnnDestroyActivationDescriptor(activationDesc: cudnnActivationDescriptor_t) -> cudnnStatus_t;
    pub fn cudnnSetActivationDescriptor(
        activationDesc: cudnnActivationDescriptor_t,
        mode: cudnnActivationMode_t,
        reluNanOpt: cudnnNanPropagation_t,
        coef: f64,
    ) -> cudnnStatus_t;
    pub fn cudnnActivationForward(
        handle: cudnnHandle_t,
        activationDesc: cudnnActivationDescriptor_t,
        alpha: *const c_void,
        xDesc: cudnnTensorDescriptor_t,
        x: *const c_void,
        beta: *const c_void,
        yDesc: cudnnTensorDescriptor_t,
        y: *mut c_void,
    ) -> cudnnStatus_t;

    // Pooling
    pub fn cudnnCreatePoolingDescriptor(poolingDesc: *mut cudnnPoolingDescriptor_t) -> cudnnStatus_t;
    pub fn cudnnDestroyPoolingDescriptor(poolingDesc: cudnnPoolingDescriptor_t) -> cudnnStatus_t;
    pub fn cudnnSetPooling2dDescriptor(
        poolingDesc: cudnnPoolingDescriptor_t,
        mode: cudnnPoolingMode_t,
        maxpoolingNanOpt: cudnnNanPropagation_t,
        windowHeight: c_int,
        windowWidth: c_int,
        verticalPadding: c_int,
        horizontalPadding: c_int,
        verticalStride: c_int,
        horizontalStride: c_int,
    ) -> cudnnStatus_t;
    pub fn cudnnPoolingForward(
        handle: cudnnHandle_t,
        poolingDesc: cudnnPoolingDescriptor_t,
        alpha: *const c_void,
        xDesc: cudnnTensorDescriptor_t,
        x: *const c_void,
        beta: *const c_void,
        yDesc: cudnnTensorDescriptor_t,
        y: *mut c_void,
    ) -> cudnnStatus_t;

    // Softmax
    pub fn cudnnSoftmaxForward(
        handle: cudnnHandle_t,
        algo: cudnnSoftmaxAlgorithm_t,
        mode: cudnnSoftmaxMode_t,
        alpha: *const c_void,
        xDesc: cudnnTensorDescriptor_t,
        x: *const c_void,
        beta: *const c_void,
        yDesc: cudnnTensorDescriptor_t,
        y: *mut c_void,
    ) -> cudnnStatus_t;

    // Batch normalization
    pub fn cudnnBatchNormalizationForwardInference(
        handle: cudnnHandle_t,
        mode: c_int,
        alpha: *const c_void,
        beta: *const c_void,
        xDesc: cudnnTensorDescriptor_t,
        x: *const c_void,
        yDesc: cudnnTensorDescriptor_t,
        y: *mut c_void,
        bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t,
        bnScale: *const c_void,
        bnBias: *const c_void,
        estimatedMean: *const c_void,
        estimatedVariance: *const c_void,
        epsilon: f64,
    ) -> cudnnStatus_t;

    // Tensor ops
    pub fn cudnnAddTensor(
        handle: cudnnHandle_t,
        alpha: *const c_void,
        aDesc: cudnnTensorDescriptor_t,
        A: *const c_void,
        beta: *const c_void,
        cDesc: cudnnTensorDescriptor_t,
        C: *mut c_void,
    ) -> cudnnStatus_t;

    pub fn cudnnTransformTensor(
        handle: cudnnHandle_t,
        alpha: *const c_void,
        xDesc: cudnnTensorDescriptor_t,
        x: *const c_void,
        beta: *const c_void,
        yDesc: cudnnTensorDescriptor_t,
        y: *mut c_void,
    ) -> cudnnStatus_t;
}
