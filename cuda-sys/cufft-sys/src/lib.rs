//! Raw FFI bindings to cuFFT.
//!
//! cuFFT is NVIDIA's GPU-accelerated library for Fast Fourier Transforms.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use cuda_runtime_sys::cudaStream_t;
use libc::{c_int, c_void, size_t};

// ============================================================================
// Types
// ============================================================================

pub type cufftResult = c_int;

pub const CUFFT_SUCCESS: cufftResult = 0x0;
pub const CUFFT_INVALID_PLAN: cufftResult = 0x1;
pub const CUFFT_ALLOC_FAILED: cufftResult = 0x2;
pub const CUFFT_INVALID_TYPE: cufftResult = 0x3;
pub const CUFFT_INVALID_VALUE: cufftResult = 0x4;
pub const CUFFT_INTERNAL_ERROR: cufftResult = 0x5;
pub const CUFFT_EXEC_FAILED: cufftResult = 0x6;
pub const CUFFT_SETUP_FAILED: cufftResult = 0x7;
pub const CUFFT_INVALID_SIZE: cufftResult = 0x8;
pub const CUFFT_UNALIGNED_DATA: cufftResult = 0x9;
pub const CUFFT_INCOMPLETE_PARAMETER_LIST: cufftResult = 0xA;
pub const CUFFT_INVALID_DEVICE: cufftResult = 0xB;
pub const CUFFT_PARSE_ERROR: cufftResult = 0xC;
pub const CUFFT_NO_WORKSPACE: cufftResult = 0xD;
pub const CUFFT_NOT_IMPLEMENTED: cufftResult = 0xE;
pub const CUFFT_LICENSE_ERROR: cufftResult = 0x0F;
pub const CUFFT_NOT_SUPPORTED: cufftResult = 0x10;

pub type cufftType = c_int;

pub const CUFFT_R2C: cufftType = 0x2a;  // Real to Complex (interleaved)
pub const CUFFT_C2R: cufftType = 0x2c;  // Complex (interleaved) to Real
pub const CUFFT_C2C: cufftType = 0x29;  // Complex to Complex (interleaved)
pub const CUFFT_D2Z: cufftType = 0x6a;  // Double to Double-Complex
pub const CUFFT_Z2D: cufftType = 0x6c;  // Double-Complex to Double
pub const CUFFT_Z2Z: cufftType = 0x69;  // Double-Complex to Double-Complex

pub type cufftHandle = c_int;

// ============================================================================
// Complex Types
// ============================================================================

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct cufftComplex {
    pub x: f32,
    pub y: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct cufftDoubleComplex {
    pub x: f64,
    pub y: f64,
}

pub type cufftReal = f32;
pub type cufftDoubleReal = f64;

// ============================================================================
// External Functions - Plan Management
// ============================================================================

extern "C" {
    pub fn cufftPlan1d(
        plan: *mut cufftHandle,
        nx: c_int,
        type_: cufftType,
        batch: c_int,
    ) -> cufftResult;

    pub fn cufftPlan2d(
        plan: *mut cufftHandle,
        nx: c_int,
        ny: c_int,
        type_: cufftType,
    ) -> cufftResult;

    pub fn cufftPlan3d(
        plan: *mut cufftHandle,
        nx: c_int,
        ny: c_int,
        nz: c_int,
        type_: cufftType,
    ) -> cufftResult;

    pub fn cufftPlanMany(
        plan: *mut cufftHandle,
        rank: c_int,
        n: *mut c_int,
        inembed: *mut c_int,
        istride: c_int,
        idist: c_int,
        onembed: *mut c_int,
        ostride: c_int,
        odist: c_int,
        type_: cufftType,
        batch: c_int,
    ) -> cufftResult;

    pub fn cufftMakePlan1d(
        plan: cufftHandle,
        nx: c_int,
        type_: cufftType,
        batch: c_int,
        workSize: *mut size_t,
    ) -> cufftResult;

    pub fn cufftMakePlan2d(
        plan: cufftHandle,
        nx: c_int,
        ny: c_int,
        type_: cufftType,
        workSize: *mut size_t,
    ) -> cufftResult;

    pub fn cufftMakePlan3d(
        plan: cufftHandle,
        nx: c_int,
        ny: c_int,
        nz: c_int,
        type_: cufftType,
        workSize: *mut size_t,
    ) -> cufftResult;

    pub fn cufftMakePlanMany(
        plan: cufftHandle,
        rank: c_int,
        n: *mut c_int,
        inembed: *mut c_int,
        istride: c_int,
        idist: c_int,
        onembed: *mut c_int,
        ostride: c_int,
        odist: c_int,
        type_: cufftType,
        batch: c_int,
        workSize: *mut size_t,
    ) -> cufftResult;

    pub fn cufftCreate(plan: *mut cufftHandle) -> cufftResult;
    pub fn cufftDestroy(plan: cufftHandle) -> cufftResult;

    pub fn cufftEstimate1d(
        nx: c_int,
        type_: cufftType,
        batch: c_int,
        workSize: *mut size_t,
    ) -> cufftResult;

    pub fn cufftEstimate2d(
        nx: c_int,
        ny: c_int,
        type_: cufftType,
        workSize: *mut size_t,
    ) -> cufftResult;

    pub fn cufftEstimate3d(
        nx: c_int,
        ny: c_int,
        nz: c_int,
        type_: cufftType,
        workSize: *mut size_t,
    ) -> cufftResult;

    pub fn cufftGetSize1d(
        plan: cufftHandle,
        nx: c_int,
        type_: cufftType,
        batch: c_int,
        workSize: *mut size_t,
    ) -> cufftResult;

    pub fn cufftGetSize2d(
        plan: cufftHandle,
        nx: c_int,
        ny: c_int,
        type_: cufftType,
        workSize: *mut size_t,
    ) -> cufftResult;

    pub fn cufftGetSize3d(
        plan: cufftHandle,
        nx: c_int,
        ny: c_int,
        nz: c_int,
        type_: cufftType,
        workSize: *mut size_t,
    ) -> cufftResult;

    pub fn cufftGetSize(plan: cufftHandle, workSize: *mut size_t) -> cufftResult;
}

// ============================================================================
// External Functions - Plan Configuration
// ============================================================================

extern "C" {
    pub fn cufftSetStream(plan: cufftHandle, stream: cudaStream_t) -> cufftResult;
    pub fn cufftSetAutoAllocation(plan: cufftHandle, autoAllocate: c_int) -> cufftResult;
    pub fn cufftSetWorkArea(plan: cufftHandle, workArea: *mut c_void) -> cufftResult;
}

// ============================================================================
// External Functions - Execution (Single Precision)
// ============================================================================

pub const CUFFT_FORWARD: c_int = -1;
pub const CUFFT_INVERSE: c_int = 1;

extern "C" {
    pub fn cufftExecC2C(
        plan: cufftHandle,
        idata: *mut cufftComplex,
        odata: *mut cufftComplex,
        direction: c_int,
    ) -> cufftResult;

    pub fn cufftExecR2C(
        plan: cufftHandle,
        idata: *mut cufftReal,
        odata: *mut cufftComplex,
    ) -> cufftResult;

    pub fn cufftExecC2R(
        plan: cufftHandle,
        idata: *mut cufftComplex,
        odata: *mut cufftReal,
    ) -> cufftResult;
}

// ============================================================================
// External Functions - Execution (Double Precision)
// ============================================================================

extern "C" {
    pub fn cufftExecZ2Z(
        plan: cufftHandle,
        idata: *mut cufftDoubleComplex,
        odata: *mut cufftDoubleComplex,
        direction: c_int,
    ) -> cufftResult;

    pub fn cufftExecD2Z(
        plan: cufftHandle,
        idata: *mut cufftDoubleReal,
        odata: *mut cufftDoubleComplex,
    ) -> cufftResult;

    pub fn cufftExecZ2D(
        plan: cufftHandle,
        idata: *mut cufftDoubleComplex,
        odata: *mut cufftDoubleReal,
    ) -> cufftResult;
}

// ============================================================================
// External Functions - Callbacks (CUDA 12+)
// ============================================================================

#[cfg(feature = "cuda-12")]
pub type cufftCallbackLoadC = Option<
    unsafe extern "C" fn(
        dataIn: *mut c_void,
        offset: size_t,
        callerInfo: *mut c_void,
        sharedPointer: *mut c_void,
    ) -> cufftComplex,
>;

#[cfg(feature = "cuda-12")]
pub type cufftCallbackStoreC = Option<
    unsafe extern "C" fn(
        dataOut: *mut c_void,
        offset: size_t,
        element: cufftComplex,
        callerInfo: *mut c_void,
        sharedPointer: *mut c_void,
    ),
>;

// ============================================================================
// Version Info
// ============================================================================

extern "C" {
    pub fn cufftGetVersion(version: *mut c_int) -> cufftResult;
    pub fn cufftGetProperty(type_: c_int, value: *mut c_int) -> cufftResult;
}
