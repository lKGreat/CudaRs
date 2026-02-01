//! Raw FFI bindings to NPP (NVIDIA Performance Primitives).

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use cuda_runtime_sys::cudaStream_t;
use libc::{c_int, c_uchar, size_t};

pub type NppStatus = c_int;
pub const NPP_NO_ERROR: NppStatus = 0;
pub const NPP_SUCCESS: NppStatus = 0;
pub const NPP_NOT_SUPPORTED_MODE_ERROR: NppStatus = -9999;
pub const NPP_INVALID_HOST_POINTER_ERROR: NppStatus = -1032;
pub const NPP_INVALID_DEVICE_POINTER_ERROR: NppStatus = -1031;
pub const NPP_LUT_PALETTE_BITSIZE_ERROR: NppStatus = -1030;
pub const NPP_ZC_MODE_NOT_SUPPORTED_ERROR: NppStatus = -1028;
pub const NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY: NppStatus = -1027;
pub const NPP_TEXTURE_BIND_ERROR: NppStatus = -1024;
pub const NPP_WRONG_INTERSECTION_ROI_ERROR: NppStatus = -1020;
pub const NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR: NppStatus = -1006;
pub const NPP_MEMFREE_ERROR: NppStatus = -1005;
pub const NPP_MEMSET_ERROR: NppStatus = -1004;
pub const NPP_MEMCPY_ERROR: NppStatus = -1003;
pub const NPP_ALIGNMENT_ERROR: NppStatus = -1002;
pub const NPP_CUDA_KERNEL_EXECUTION_ERROR: NppStatus = -1000;
pub const NPP_ROUND_MODE_NOT_SUPPORTED_ERROR: NppStatus = -213;
pub const NPP_QUALITY_INDEX_ERROR: NppStatus = -210;
pub const NPP_RESIZE_NO_OPERATION_ERROR: NppStatus = -201;
pub const NPP_OVERFLOW_ERROR: NppStatus = -109;
pub const NPP_NOT_EVEN_STEP_ERROR: NppStatus = -108;
pub const NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR: NppStatus = -107;
pub const NPP_LUT_NUMBER_OF_LEVELS_ERROR: NppStatus = -106;
pub const NPP_CORRUPTED_DATA_ERROR: NppStatus = -61;
pub const NPP_CHANNEL_ORDER_ERROR: NppStatus = -60;
pub const NPP_ZERO_MASK_VALUE_ERROR: NppStatus = -59;
pub const NPP_QUADRANGLE_ERROR: NppStatus = -58;
pub const NPP_RECTANGLE_ERROR: NppStatus = -57;
pub const NPP_COEFFICIENT_ERROR: NppStatus = -56;
pub const NPP_NUMBER_OF_CHANNELS_ERROR: NppStatus = -53;
pub const NPP_COI_ERROR: NppStatus = -52;
pub const NPP_DIVISOR_ERROR: NppStatus = -51;
pub const NPP_CHANNEL_ERROR: NppStatus = -47;
pub const NPP_STRIDE_ERROR: NppStatus = -37;
pub const NPP_ANCHOR_ERROR: NppStatus = -34;
pub const NPP_MASK_SIZE_ERROR: NppStatus = -33;
pub const NPP_RESIZE_FACTOR_ERROR: NppStatus = -23;
pub const NPP_INTERPOLATION_ERROR: NppStatus = -22;
pub const NPP_MIRROR_FLIP_ERROR: NppStatus = -21;
pub const NPP_MOMENT_00_ZERO_ERROR: NppStatus = -20;
pub const NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR: NppStatus = -19;
pub const NPP_THRESHOLD_ERROR: NppStatus = -18;
pub const NPP_CONTEXT_MATCH_ERROR: NppStatus = -17;
pub const NPP_FFT_FLAG_ERROR: NppStatus = -16;
pub const NPP_FFT_ORDER_ERROR: NppStatus = -15;
pub const NPP_STEP_ERROR: NppStatus = -14;
pub const NPP_SCALE_RANGE_ERROR: NppStatus = -13;
pub const NPP_DATA_TYPE_ERROR: NppStatus = -12;
pub const NPP_OUT_OFF_RANGE_ERROR: NppStatus = -11;
pub const NPP_DIVIDE_BY_ZERO_ERROR: NppStatus = -10;
pub const NPP_MEMORY_ALLOCATION_ERR: NppStatus = -9;
pub const NPP_NULL_POINTER_ERROR: NppStatus = -8;
pub const NPP_RANGE_ERROR: NppStatus = -7;
pub const NPP_SIZE_ERROR: NppStatus = -6;
pub const NPP_BAD_ARGUMENT_ERROR: NppStatus = -5;
pub const NPP_NO_MEMORY_ERROR: NppStatus = -4;
pub const NPP_NOT_IMPLEMENTED_ERROR: NppStatus = -3;
pub const NPP_ERROR: NppStatus = -2;
pub const NPP_ERROR_RESERVED: NppStatus = -1;

pub type NppRoundMode = c_int;
pub const NPP_RND_NEAR: NppRoundMode = 0;
pub const NPP_RND_FINANCIAL: NppRoundMode = 1;
pub const NPP_RND_ZERO: NppRoundMode = 2;

pub type NppiInterpolationMode = c_int;
pub const NPPI_INTER_NN: NppiInterpolationMode = 1;
pub const NPPI_INTER_LINEAR: NppiInterpolationMode = 2;
pub const NPPI_INTER_CUBIC: NppiInterpolationMode = 4;
pub const NPPI_INTER_CUBIC2P_BSPLINE: NppiInterpolationMode = 5;
pub const NPPI_INTER_CUBIC2P_CATMULLROM: NppiInterpolationMode = 6;
pub const NPPI_INTER_CUBIC2P_B05C03: NppiInterpolationMode = 7;
pub const NPPI_INTER_SUPER: NppiInterpolationMode = 8;
pub const NPPI_INTER_LANCZOS: NppiInterpolationMode = 16;
pub const NPPI_INTER_LANCZOS3_ADVANCED: NppiInterpolationMode = 17;
pub const NPPI_SMOOTH_EDGE: NppiInterpolationMode = 0x8000000;

pub type Npp8u = c_uchar;
pub type Npp8s = i8;
pub type Npp16u = u16;
pub type Npp16s = i16;
pub type Npp32u = u32;
pub type Npp32s = i32;
pub type Npp64u = u64;
pub type Npp64s = i64;
pub type Npp32f = f32;
pub type Npp64f = f64;

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct NppiSize {
    pub width: c_int,
    pub height: c_int,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct NppiPoint {
    pub x: c_int,
    pub y: c_int,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct NppiRect {
    pub x: c_int,
    pub y: c_int,
    pub width: c_int,
    pub height: c_int,
}

#[repr(C)]
pub struct NppStreamContext {
    pub hStream: cudaStream_t,
    pub nCudaDeviceId: c_int,
    pub nMultiProcessorCount: c_int,
    pub nMaxThreadsPerMultiProcessor: c_int,
    pub nMaxThreadsPerBlock: c_int,
    pub nSharedMemPerBlock: size_t,
    pub nCudaDevAttrComputeCapabilityMajor: c_int,
    pub nCudaDevAttrComputeCapabilityMinor: c_int,
}

extern "C" {
    // Version
    pub fn nppGetLibVersion() -> *const c_int;
    pub fn nppGetGpuComputeCapability() -> c_int;
    pub fn nppGetGpuNumSMs() -> c_int;
    pub fn nppGetMaxThreadsPerBlock() -> c_int;
    pub fn nppGetMaxThreadsPerSM() -> c_int;

    // Stream context
    pub fn nppSetStream(hStream: cudaStream_t) -> NppStatus;
    pub fn nppGetStream() -> cudaStream_t;
    pub fn nppGetStreamContext(pNppStreamContext: *mut NppStreamContext) -> NppStatus;

    // Memory allocation
    pub fn nppiMalloc_8u_C1(nWidthPixels: c_int, nHeightPixels: c_int, pStepBytes: *mut c_int) -> *mut Npp8u;
    pub fn nppiMalloc_8u_C3(nWidthPixels: c_int, nHeightPixels: c_int, pStepBytes: *mut c_int) -> *mut Npp8u;
    pub fn nppiMalloc_8u_C4(nWidthPixels: c_int, nHeightPixels: c_int, pStepBytes: *mut c_int) -> *mut Npp8u;
    pub fn nppiMalloc_32f_C1(nWidthPixels: c_int, nHeightPixels: c_int, pStepBytes: *mut c_int) -> *mut Npp32f;
    pub fn nppiMalloc_32f_C3(nWidthPixels: c_int, nHeightPixels: c_int, pStepBytes: *mut c_int) -> *mut Npp32f;
    pub fn nppiMalloc_32f_C4(nWidthPixels: c_int, nHeightPixels: c_int, pStepBytes: *mut c_int) -> *mut Npp32f;
    pub fn nppiFree(pData: *mut libc::c_void);

    // Image resize
    pub fn nppiResize_8u_C1R(
        pSrc: *const Npp8u,
        nSrcStep: c_int,
        oSrcSize: NppiSize,
        oSrcRectROI: NppiRect,
        pDst: *mut Npp8u,
        nDstStep: c_int,
        oDstSize: NppiSize,
        oDstRectROI: NppiRect,
        eInterpolation: NppiInterpolationMode,
    ) -> NppStatus;

    pub fn nppiResize_8u_C3R(
        pSrc: *const Npp8u,
        nSrcStep: c_int,
        oSrcSize: NppiSize,
        oSrcRectROI: NppiRect,
        pDst: *mut Npp8u,
        nDstStep: c_int,
        oDstSize: NppiSize,
        oDstRectROI: NppiRect,
        eInterpolation: NppiInterpolationMode,
    ) -> NppStatus;

    // Color conversion
    pub fn nppiRGBToGray_8u_C3C1R(
        pSrc: *const Npp8u,
        nSrcStep: c_int,
        pDst: *mut Npp8u,
        nDstStep: c_int,
        oSizeROI: NppiSize,
    ) -> NppStatus;

    pub fn nppiBGRToGray_8u_C3C1R(
        pSrc: *const Npp8u,
        nSrcStep: c_int,
        pDst: *mut Npp8u,
        nDstStep: c_int,
        oSizeROI: NppiSize,
    ) -> NppStatus;

    // Filter
    pub fn nppiFilterGauss_8u_C1R(
        pSrc: *const Npp8u,
        nSrcStep: c_int,
        pDst: *mut Npp8u,
        nDstStep: c_int,
        oSizeROI: NppiSize,
        eMaskSize: c_int,
    ) -> NppStatus;
}
