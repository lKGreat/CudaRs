//! Raw FFI bindings to cuSOLVER.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use cuda_runtime_sys::cudaStream_t;
use libc::c_int;

pub type cusolverStatus_t = c_int;
pub const CUSOLVER_STATUS_SUCCESS: cusolverStatus_t = 0;
pub const CUSOLVER_STATUS_NOT_INITIALIZED: cusolverStatus_t = 1;
pub const CUSOLVER_STATUS_ALLOC_FAILED: cusolverStatus_t = 2;
pub const CUSOLVER_STATUS_INVALID_VALUE: cusolverStatus_t = 3;
pub const CUSOLVER_STATUS_ARCH_MISMATCH: cusolverStatus_t = 4;
pub const CUSOLVER_STATUS_EXECUTION_FAILED: cusolverStatus_t = 5;
pub const CUSOLVER_STATUS_INTERNAL_ERROR: cusolverStatus_t = 6;
pub const CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: cusolverStatus_t = 7;
pub const CUSOLVER_STATUS_NOT_SUPPORTED: cusolverStatus_t = 8;

pub type cublasFillMode_t = c_int;
pub const CUBLAS_FILL_MODE_LOWER: cublasFillMode_t = 0;
pub const CUBLAS_FILL_MODE_UPPER: cublasFillMode_t = 1;

pub type cublasOperation_t = c_int;
pub const CUBLAS_OP_N: cublasOperation_t = 0;
pub const CUBLAS_OP_T: cublasOperation_t = 1;
pub const CUBLAS_OP_C: cublasOperation_t = 2;

#[repr(C)]
pub struct cusolverDnContext { _unused: [u8; 0] }
pub type cusolverDnHandle_t = *mut cusolverDnContext;

#[repr(C)]
pub struct cusolverSpContext { _unused: [u8; 0] }
pub type cusolverSpHandle_t = *mut cusolverSpContext;

#[repr(C)]
pub struct syevjInfo { _unused: [u8; 0] }
pub type syevjInfo_t = *mut syevjInfo;

#[repr(C)]
pub struct gesvdjInfo { _unused: [u8; 0] }
pub type gesvdjInfo_t = *mut gesvdjInfo;

pub type cusolverEigType_t = c_int;
pub const CUSOLVER_EIG_TYPE_1: cusolverEigType_t = 1;
pub const CUSOLVER_EIG_TYPE_2: cusolverEigType_t = 2;
pub const CUSOLVER_EIG_TYPE_3: cusolverEigType_t = 3;

pub type cusolverEigMode_t = c_int;
pub const CUSOLVER_EIG_MODE_NOVECTOR: cusolverEigMode_t = 0;
pub const CUSOLVER_EIG_MODE_VECTOR: cusolverEigMode_t = 1;

extern "C" {
    // Dense handle
    pub fn cusolverDnCreate(handle: *mut cusolverDnHandle_t) -> cusolverStatus_t;
    pub fn cusolverDnDestroy(handle: cusolverDnHandle_t) -> cusolverStatus_t;
    pub fn cusolverDnSetStream(handle: cusolverDnHandle_t, streamId: cudaStream_t) -> cusolverStatus_t;
    pub fn cusolverDnGetStream(handle: cusolverDnHandle_t, streamId: *mut cudaStream_t) -> cusolverStatus_t;

    // LU factorization
    pub fn cusolverDnSgetrf_bufferSize(
        handle: cusolverDnHandle_t,
        m: c_int,
        n: c_int,
        A: *mut f32,
        lda: c_int,
        Lwork: *mut c_int,
    ) -> cusolverStatus_t;

    pub fn cusolverDnSgetrf(
        handle: cusolverDnHandle_t,
        m: c_int,
        n: c_int,
        A: *mut f32,
        lda: c_int,
        Workspace: *mut f32,
        devIpiv: *mut c_int,
        devInfo: *mut c_int,
    ) -> cusolverStatus_t;

    pub fn cusolverDnDgetrf_bufferSize(
        handle: cusolverDnHandle_t,
        m: c_int,
        n: c_int,
        A: *mut f64,
        lda: c_int,
        Lwork: *mut c_int,
    ) -> cusolverStatus_t;

    pub fn cusolverDnDgetrf(
        handle: cusolverDnHandle_t,
        m: c_int,
        n: c_int,
        A: *mut f64,
        lda: c_int,
        Workspace: *mut f64,
        devIpiv: *mut c_int,
        devInfo: *mut c_int,
    ) -> cusolverStatus_t;

    // Solve with LU
    pub fn cusolverDnSgetrs(
        handle: cusolverDnHandle_t,
        trans: cublasOperation_t,
        n: c_int,
        nrhs: c_int,
        A: *const f32,
        lda: c_int,
        devIpiv: *const c_int,
        B: *mut f32,
        ldb: c_int,
        devInfo: *mut c_int,
    ) -> cusolverStatus_t;

    pub fn cusolverDnDgetrs(
        handle: cusolverDnHandle_t,
        trans: cublasOperation_t,
        n: c_int,
        nrhs: c_int,
        A: *const f64,
        lda: c_int,
        devIpiv: *const c_int,
        B: *mut f64,
        ldb: c_int,
        devInfo: *mut c_int,
    ) -> cusolverStatus_t;

    // Cholesky factorization
    pub fn cusolverDnSpotrf_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: c_int,
        A: *mut f32,
        lda: c_int,
        Lwork: *mut c_int,
    ) -> cusolverStatus_t;

    pub fn cusolverDnSpotrf(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: c_int,
        A: *mut f32,
        lda: c_int,
        Workspace: *mut f32,
        Lwork: c_int,
        devInfo: *mut c_int,
    ) -> cusolverStatus_t;

    pub fn cusolverDnDpotrf_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: c_int,
        A: *mut f64,
        lda: c_int,
        Lwork: *mut c_int,
    ) -> cusolverStatus_t;

    pub fn cusolverDnDpotrf(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: c_int,
        A: *mut f64,
        lda: c_int,
        Workspace: *mut f64,
        Lwork: c_int,
        devInfo: *mut c_int,
    ) -> cusolverStatus_t;

    // QR factorization
    pub fn cusolverDnSgeqrf_bufferSize(
        handle: cusolverDnHandle_t,
        m: c_int,
        n: c_int,
        A: *mut f32,
        lda: c_int,
        Lwork: *mut c_int,
    ) -> cusolverStatus_t;

    pub fn cusolverDnSgeqrf(
        handle: cusolverDnHandle_t,
        m: c_int,
        n: c_int,
        A: *mut f32,
        lda: c_int,
        TAU: *mut f32,
        Workspace: *mut f32,
        Lwork: c_int,
        devInfo: *mut c_int,
    ) -> cusolverStatus_t;

    // SVD
    pub fn cusolverDnSgesvd_bufferSize(
        handle: cusolverDnHandle_t,
        m: c_int,
        n: c_int,
        Lwork: *mut c_int,
    ) -> cusolverStatus_t;

    pub fn cusolverDnSgesvd(
        handle: cusolverDnHandle_t,
        jobu: c_int,
        jobvt: c_int,
        m: c_int,
        n: c_int,
        A: *mut f32,
        lda: c_int,
        S: *mut f32,
        U: *mut f32,
        ldu: c_int,
        VT: *mut f32,
        ldvt: c_int,
        Work: *mut f32,
        Lwork: c_int,
        rwork: *mut f32,
        devInfo: *mut c_int,
    ) -> cusolverStatus_t;

    // Eigenvalue
    pub fn cusolverDnSsyevd_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: c_int,
        A: *const f32,
        lda: c_int,
        W: *const f32,
        Lwork: *mut c_int,
    ) -> cusolverStatus_t;

    pub fn cusolverDnSsyevd(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: c_int,
        A: *mut f32,
        lda: c_int,
        W: *mut f32,
        Work: *mut f32,
        Lwork: c_int,
        devInfo: *mut c_int,
    ) -> cusolverStatus_t;

    // Sparse handle
    pub fn cusolverSpCreate(handle: *mut cusolverSpHandle_t) -> cusolverStatus_t;
    pub fn cusolverSpDestroy(handle: cusolverSpHandle_t) -> cusolverStatus_t;
    pub fn cusolverSpSetStream(handle: cusolverSpHandle_t, streamId: cudaStream_t) -> cusolverStatus_t;
}
