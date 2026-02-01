//! Raw FFI bindings to cuBLAS.
//!
//! cuBLAS is NVIDIA's GPU-accelerated library for basic linear algebra subroutines.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use cuda_runtime_sys::cudaStream_t;
use libc::{c_char, c_int, c_void};

// ============================================================================
// Types
// ============================================================================

pub type cublasStatus_t = c_int;

pub const CUBLAS_STATUS_SUCCESS: cublasStatus_t = 0;
pub const CUBLAS_STATUS_NOT_INITIALIZED: cublasStatus_t = 1;
pub const CUBLAS_STATUS_ALLOC_FAILED: cublasStatus_t = 3;
pub const CUBLAS_STATUS_INVALID_VALUE: cublasStatus_t = 7;
pub const CUBLAS_STATUS_ARCH_MISMATCH: cublasStatus_t = 8;
pub const CUBLAS_STATUS_MAPPING_ERROR: cublasStatus_t = 11;
pub const CUBLAS_STATUS_EXECUTION_FAILED: cublasStatus_t = 13;
pub const CUBLAS_STATUS_INTERNAL_ERROR: cublasStatus_t = 14;
pub const CUBLAS_STATUS_NOT_SUPPORTED: cublasStatus_t = 15;
pub const CUBLAS_STATUS_LICENSE_ERROR: cublasStatus_t = 16;

pub type cublasOperation_t = c_int;

pub const CUBLAS_OP_N: cublasOperation_t = 0;
pub const CUBLAS_OP_T: cublasOperation_t = 1;
pub const CUBLAS_OP_C: cublasOperation_t = 2;

pub type cublasFillMode_t = c_int;

pub const CUBLAS_FILL_MODE_LOWER: cublasFillMode_t = 0;
pub const CUBLAS_FILL_MODE_UPPER: cublasFillMode_t = 1;
pub const CUBLAS_FILL_MODE_FULL: cublasFillMode_t = 2;

pub type cublasDiagType_t = c_int;

pub const CUBLAS_DIAG_NON_UNIT: cublasDiagType_t = 0;
pub const CUBLAS_DIAG_UNIT: cublasDiagType_t = 1;

pub type cublasSideMode_t = c_int;

pub const CUBLAS_SIDE_LEFT: cublasSideMode_t = 0;
pub const CUBLAS_SIDE_RIGHT: cublasSideMode_t = 1;

pub type cublasPointerMode_t = c_int;

pub const CUBLAS_POINTER_MODE_HOST: cublasPointerMode_t = 0;
pub const CUBLAS_POINTER_MODE_DEVICE: cublasPointerMode_t = 1;

pub type cublasAtomicsMode_t = c_int;

pub const CUBLAS_ATOMICS_NOT_ALLOWED: cublasAtomicsMode_t = 0;
pub const CUBLAS_ATOMICS_ALLOWED: cublasAtomicsMode_t = 1;

pub type cublasGemmAlgo_t = c_int;

pub const CUBLAS_GEMM_DFALT: cublasGemmAlgo_t = -1;
pub const CUBLAS_GEMM_DEFAULT: cublasGemmAlgo_t = -1;
pub const CUBLAS_GEMM_ALGO0: cublasGemmAlgo_t = 0;
pub const CUBLAS_GEMM_ALGO1: cublasGemmAlgo_t = 1;
pub const CUBLAS_GEMM_ALGO2: cublasGemmAlgo_t = 2;
pub const CUBLAS_GEMM_ALGO3: cublasGemmAlgo_t = 3;

pub type cublasMath_t = c_int;

pub const CUBLAS_DEFAULT_MATH: cublasMath_t = 0;
pub const CUBLAS_TENSOR_OP_MATH: cublasMath_t = 1;
pub const CUBLAS_PEDANTIC_MATH: cublasMath_t = 2;
pub const CUBLAS_TF32_TENSOR_OP_MATH: cublasMath_t = 3;

pub type cublasComputeType_t = c_int;

pub const CUBLAS_COMPUTE_16F: cublasComputeType_t = 64;
pub const CUBLAS_COMPUTE_16F_PEDANTIC: cublasComputeType_t = 65;
pub const CUBLAS_COMPUTE_32F: cublasComputeType_t = 68;
pub const CUBLAS_COMPUTE_32F_PEDANTIC: cublasComputeType_t = 69;
pub const CUBLAS_COMPUTE_32F_FAST_16F: cublasComputeType_t = 74;
pub const CUBLAS_COMPUTE_32F_FAST_16BF: cublasComputeType_t = 75;
pub const CUBLAS_COMPUTE_32F_FAST_TF32: cublasComputeType_t = 77;
pub const CUBLAS_COMPUTE_64F: cublasComputeType_t = 70;
pub const CUBLAS_COMPUTE_64F_PEDANTIC: cublasComputeType_t = 71;
pub const CUBLAS_COMPUTE_32I: cublasComputeType_t = 72;
pub const CUBLAS_COMPUTE_32I_PEDANTIC: cublasComputeType_t = 73;

// ============================================================================
// Opaque Types
// ============================================================================

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cublasContext {
    _unused: [u8; 0],
}
pub type cublasHandle_t = *mut cublasContext;

// ============================================================================
// External Functions - Helper
// ============================================================================

extern "C" {
    pub fn cublasCreate(handle: *mut cublasHandle_t) -> cublasStatus_t;
    pub fn cublasDestroy(handle: cublasHandle_t) -> cublasStatus_t;
    pub fn cublasGetVersion(handle: cublasHandle_t, version: *mut c_int) -> cublasStatus_t;
    pub fn cublasGetProperty(type_: c_int, value: *mut c_int) -> cublasStatus_t;
    pub fn cublasGetStatusName(status: cublasStatus_t) -> *const c_char;
    pub fn cublasGetStatusString(status: cublasStatus_t) -> *const c_char;
    pub fn cublasSetStream(handle: cublasHandle_t, streamId: cudaStream_t) -> cublasStatus_t;
    pub fn cublasGetStream(handle: cublasHandle_t, streamId: *mut cudaStream_t) -> cublasStatus_t;
    pub fn cublasSetPointerMode(handle: cublasHandle_t, mode: cublasPointerMode_t) -> cublasStatus_t;
    pub fn cublasGetPointerMode(handle: cublasHandle_t, mode: *mut cublasPointerMode_t) -> cublasStatus_t;
    pub fn cublasSetAtomicsMode(handle: cublasHandle_t, mode: cublasAtomicsMode_t) -> cublasStatus_t;
    pub fn cublasGetAtomicsMode(handle: cublasHandle_t, mode: *mut cublasAtomicsMode_t) -> cublasStatus_t;
    pub fn cublasSetMathMode(handle: cublasHandle_t, mode: cublasMath_t) -> cublasStatus_t;
    pub fn cublasGetMathMode(handle: cublasHandle_t, mode: *mut cublasMath_t) -> cublasStatus_t;
}

// ============================================================================
// External Functions - Level 1 BLAS
// ============================================================================

extern "C" {
    // SAXPY: y = alpha * x + y
    pub fn cublasSaxpy(
        handle: cublasHandle_t,
        n: c_int,
        alpha: *const f32,
        x: *const f32,
        incx: c_int,
        y: *mut f32,
        incy: c_int,
    ) -> cublasStatus_t;

    pub fn cublasDaxpy(
        handle: cublasHandle_t,
        n: c_int,
        alpha: *const f64,
        x: *const f64,
        incx: c_int,
        y: *mut f64,
        incy: c_int,
    ) -> cublasStatus_t;

    // SCOPY: y = x
    pub fn cublasScopy(
        handle: cublasHandle_t,
        n: c_int,
        x: *const f32,
        incx: c_int,
        y: *mut f32,
        incy: c_int,
    ) -> cublasStatus_t;

    pub fn cublasDcopy(
        handle: cublasHandle_t,
        n: c_int,
        x: *const f64,
        incx: c_int,
        y: *mut f64,
        incy: c_int,
    ) -> cublasStatus_t;

    // SDOT: result = x . y
    pub fn cublasSdot(
        handle: cublasHandle_t,
        n: c_int,
        x: *const f32,
        incx: c_int,
        y: *const f32,
        incy: c_int,
        result: *mut f32,
    ) -> cublasStatus_t;

    pub fn cublasDdot(
        handle: cublasHandle_t,
        n: c_int,
        x: *const f64,
        incx: c_int,
        y: *const f64,
        incy: c_int,
        result: *mut f64,
    ) -> cublasStatus_t;

    // SNRM2: result = ||x||
    pub fn cublasSnrm2(
        handle: cublasHandle_t,
        n: c_int,
        x: *const f32,
        incx: c_int,
        result: *mut f32,
    ) -> cublasStatus_t;

    pub fn cublasDnrm2(
        handle: cublasHandle_t,
        n: c_int,
        x: *const f64,
        incx: c_int,
        result: *mut f64,
    ) -> cublasStatus_t;

    // SSCAL: x = alpha * x
    pub fn cublasSscal(
        handle: cublasHandle_t,
        n: c_int,
        alpha: *const f32,
        x: *mut f32,
        incx: c_int,
    ) -> cublasStatus_t;

    pub fn cublasDscal(
        handle: cublasHandle_t,
        n: c_int,
        alpha: *const f64,
        x: *mut f64,
        incx: c_int,
    ) -> cublasStatus_t;

    // ISAMAX: index of max |x[i]|
    pub fn cublasIsamax(
        handle: cublasHandle_t,
        n: c_int,
        x: *const f32,
        incx: c_int,
        result: *mut c_int,
    ) -> cublasStatus_t;

    pub fn cublasIdamax(
        handle: cublasHandle_t,
        n: c_int,
        x: *const f64,
        incx: c_int,
        result: *mut c_int,
    ) -> cublasStatus_t;

    // ISAMIN: index of min |x[i]|
    pub fn cublasIsamin(
        handle: cublasHandle_t,
        n: c_int,
        x: *const f32,
        incx: c_int,
        result: *mut c_int,
    ) -> cublasStatus_t;

    pub fn cublasIdamin(
        handle: cublasHandle_t,
        n: c_int,
        x: *const f64,
        incx: c_int,
        result: *mut c_int,
    ) -> cublasStatus_t;

    // SASUM: result = sum |x[i]|
    pub fn cublasSasum(
        handle: cublasHandle_t,
        n: c_int,
        x: *const f32,
        incx: c_int,
        result: *mut f32,
    ) -> cublasStatus_t;

    pub fn cublasDasum(
        handle: cublasHandle_t,
        n: c_int,
        x: *const f64,
        incx: c_int,
        result: *mut f64,
    ) -> cublasStatus_t;

    // SSWAP: swap x and y
    pub fn cublasSswap(
        handle: cublasHandle_t,
        n: c_int,
        x: *mut f32,
        incx: c_int,
        y: *mut f32,
        incy: c_int,
    ) -> cublasStatus_t;

    pub fn cublasDswap(
        handle: cublasHandle_t,
        n: c_int,
        x: *mut f64,
        incx: c_int,
        y: *mut f64,
        incy: c_int,
    ) -> cublasStatus_t;
}

// ============================================================================
// External Functions - Level 2 BLAS
// ============================================================================

extern "C" {
    // SGEMV: y = alpha * A * x + beta * y
    pub fn cublasSgemv(
        handle: cublasHandle_t,
        trans: cublasOperation_t,
        m: c_int,
        n: c_int,
        alpha: *const f32,
        A: *const f32,
        lda: c_int,
        x: *const f32,
        incx: c_int,
        beta: *const f32,
        y: *mut f32,
        incy: c_int,
    ) -> cublasStatus_t;

    pub fn cublasDgemv(
        handle: cublasHandle_t,
        trans: cublasOperation_t,
        m: c_int,
        n: c_int,
        alpha: *const f64,
        A: *const f64,
        lda: c_int,
        x: *const f64,
        incx: c_int,
        beta: *const f64,
        y: *mut f64,
        incy: c_int,
    ) -> cublasStatus_t;

    // SGER: A = alpha * x * y^T + A
    pub fn cublasSger(
        handle: cublasHandle_t,
        m: c_int,
        n: c_int,
        alpha: *const f32,
        x: *const f32,
        incx: c_int,
        y: *const f32,
        incy: c_int,
        A: *mut f32,
        lda: c_int,
    ) -> cublasStatus_t;

    pub fn cublasDger(
        handle: cublasHandle_t,
        m: c_int,
        n: c_int,
        alpha: *const f64,
        x: *const f64,
        incx: c_int,
        y: *const f64,
        incy: c_int,
        A: *mut f64,
        lda: c_int,
    ) -> cublasStatus_t;
}

// ============================================================================
// External Functions - Level 3 BLAS
// ============================================================================

extern "C" {
    // SGEMM: C = alpha * A * B + beta * C
    pub fn cublasSgemm(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const f32,
        A: *const f32,
        lda: c_int,
        B: *const f32,
        ldb: c_int,
        beta: *const f32,
        C: *mut f32,
        ldc: c_int,
    ) -> cublasStatus_t;

    pub fn cublasDgemm(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const f64,
        A: *const f64,
        lda: c_int,
        B: *const f64,
        ldb: c_int,
        beta: *const f64,
        C: *mut f64,
        ldc: c_int,
    ) -> cublasStatus_t;

    // HGEMM (half precision)
    pub fn cublasHgemm(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const u16,  // __half
        A: *const u16,
        lda: c_int,
        B: *const u16,
        ldb: c_int,
        beta: *const u16,
        C: *mut u16,
        ldc: c_int,
    ) -> cublasStatus_t;

    // SGEMM batched
    pub fn cublasSgemmBatched(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const f32,
        Aarray: *const *const f32,
        lda: c_int,
        Barray: *const *const f32,
        ldb: c_int,
        beta: *const f32,
        Carray: *const *mut f32,
        ldc: c_int,
        batchCount: c_int,
    ) -> cublasStatus_t;

    pub fn cublasDgemmBatched(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const f64,
        Aarray: *const *const f64,
        lda: c_int,
        Barray: *const *const f64,
        ldb: c_int,
        beta: *const f64,
        Carray: *const *mut f64,
        ldc: c_int,
        batchCount: c_int,
    ) -> cublasStatus_t;

    // SGEMM strided batched
    pub fn cublasSgemmStridedBatched(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const f32,
        A: *const f32,
        lda: c_int,
        strideA: i64,
        B: *const f32,
        ldb: c_int,
        strideB: i64,
        beta: *const f32,
        C: *mut f32,
        ldc: c_int,
        strideC: i64,
        batchCount: c_int,
    ) -> cublasStatus_t;

    pub fn cublasDgemmStridedBatched(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const f64,
        A: *const f64,
        lda: c_int,
        strideA: i64,
        B: *const f64,
        ldb: c_int,
        strideB: i64,
        beta: *const f64,
        C: *mut f64,
        ldc: c_int,
        strideC: i64,
        batchCount: c_int,
    ) -> cublasStatus_t;

    // GemmEx (mixed precision)
    pub fn cublasGemmEx(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_void,
        A: *const c_void,
        Atype: c_int,
        lda: c_int,
        B: *const c_void,
        Btype: c_int,
        ldb: c_int,
        beta: *const c_void,
        C: *mut c_void,
        Ctype: c_int,
        ldc: c_int,
        computeType: cublasComputeType_t,
        algo: cublasGemmAlgo_t,
    ) -> cublasStatus_t;

    // SSYRK: C = alpha * A * A^T + beta * C
    pub fn cublasSsyrk(
        handle: cublasHandle_t,
        uplo: cublasFillMode_t,
        trans: cublasOperation_t,
        n: c_int,
        k: c_int,
        alpha: *const f32,
        A: *const f32,
        lda: c_int,
        beta: *const f32,
        C: *mut f32,
        ldc: c_int,
    ) -> cublasStatus_t;

    pub fn cublasDsyrk(
        handle: cublasHandle_t,
        uplo: cublasFillMode_t,
        trans: cublasOperation_t,
        n: c_int,
        k: c_int,
        alpha: *const f64,
        A: *const f64,
        lda: c_int,
        beta: *const f64,
        C: *mut f64,
        ldc: c_int,
    ) -> cublasStatus_t;

    // STRSM: solve A * X = alpha * B or X * A = alpha * B
    pub fn cublasStrsm(
        handle: cublasHandle_t,
        side: cublasSideMode_t,
        uplo: cublasFillMode_t,
        trans: cublasOperation_t,
        diag: cublasDiagType_t,
        m: c_int,
        n: c_int,
        alpha: *const f32,
        A: *const f32,
        lda: c_int,
        B: *mut f32,
        ldb: c_int,
    ) -> cublasStatus_t;

    pub fn cublasDtrsm(
        handle: cublasHandle_t,
        side: cublasSideMode_t,
        uplo: cublasFillMode_t,
        trans: cublasOperation_t,
        diag: cublasDiagType_t,
        m: c_int,
        n: c_int,
        alpha: *const f64,
        A: *const f64,
        lda: c_int,
        B: *mut f64,
        ldb: c_int,
    ) -> cublasStatus_t;
}

// ============================================================================
// cuBLASLt (Lightweight) for advanced GEMM
// ============================================================================

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cublasLtContext {
    _unused: [u8; 0],
}
pub type cublasLtHandle_t = *mut cublasLtContext;

extern "C" {
    pub fn cublasLtCreate(lightHandle: *mut cublasLtHandle_t) -> cublasStatus_t;
    pub fn cublasLtDestroy(lightHandle: cublasLtHandle_t) -> cublasStatus_t;
}
