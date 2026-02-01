//! Safe Rust wrapper for cuSOLVER.

use cuda_runtime::Stream;
use cusolver_sys::*;
use std::ptr;
use thiserror::Error;

#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
#[error("cuSOLVER Error: {0}")]
pub struct CusolverError(pub i32);

pub type Result<T> = std::result::Result<T, CusolverError>;

#[inline]
fn check(code: cusolverStatus_t) -> Result<()> {
    if code == CUSOLVER_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(CusolverError(code))
    }
}

/// cuSOLVER Dense Handle wrapper with automatic resource management.
pub struct DnHandle {
    handle: cusolverDnHandle_t,
}

impl DnHandle {
    /// Create a new cuSOLVER dense handle.
    pub fn new() -> Result<Self> {
        let mut handle = ptr::null_mut();
        unsafe { check(cusolverDnCreate(&mut handle))? };
        Ok(Self { handle })
    }

    /// Set the stream for this handle.
    pub fn set_stream(&self, stream: &Stream) -> Result<()> {
        unsafe { check(cusolverDnSetStream(self.handle, stream.as_raw())) }
    }

    /// Get the buffer size for LU factorization (single precision).
    pub fn sgetrf_buffer_size(&self, m: i32, n: i32, a: *const f32, lda: i32) -> Result<i32> {
        let mut lwork = 0;
        unsafe { check(cusolverDnSgetrf_bufferSize(self.handle, m, n, a as *mut _, lda, &mut lwork))? };
        Ok(lwork)
    }

    /// LU factorization (single precision).
    pub fn sgetrf(
        &self,
        m: i32,
        n: i32,
        a: *mut f32,
        lda: i32,
        workspace: *mut f32,
        ipiv: *mut i32,
        info: *mut i32,
    ) -> Result<()> {
        unsafe { check(cusolverDnSgetrf(self.handle, m, n, a, lda, workspace, ipiv, info)) }
    }

    /// Solve linear system using LU factorization (single precision).
    pub fn sgetrs(
        &self,
        trans: i32,
        n: i32,
        nrhs: i32,
        a: *const f32,
        lda: i32,
        ipiv: *const i32,
        b: *mut f32,
        ldb: i32,
        info: *mut i32,
    ) -> Result<()> {
        unsafe { check(cusolverDnSgetrs(self.handle, trans, n, nrhs, a, lda, ipiv, b, ldb, info)) }
    }

    /// Get the buffer size for Cholesky factorization (single precision).
    pub fn spotrf_buffer_size(&self, uplo: i32, n: i32, a: *const f32, lda: i32) -> Result<i32> {
        let mut lwork = 0;
        unsafe { check(cusolverDnSpotrf_bufferSize(self.handle, uplo, n, a as *mut _, lda, &mut lwork))? };
        Ok(lwork)
    }

    /// Cholesky factorization (single precision).
    pub fn spotrf(
        &self,
        uplo: i32,
        n: i32,
        a: *mut f32,
        lda: i32,
        workspace: *mut f32,
        lwork: i32,
        info: *mut i32,
    ) -> Result<()> {
        unsafe { check(cusolverDnSpotrf(self.handle, uplo, n, a, lda, workspace, lwork, info)) }
    }

    /// Get the buffer size for QR factorization (single precision).
    pub fn sgeqrf_buffer_size(&self, m: i32, n: i32, a: *const f32, lda: i32) -> Result<i32> {
        let mut lwork = 0;
        unsafe { check(cusolverDnSgeqrf_bufferSize(self.handle, m, n, a as *mut _, lda, &mut lwork))? };
        Ok(lwork)
    }

    /// QR factorization (single precision).
    pub fn sgeqrf(
        &self,
        m: i32,
        n: i32,
        a: *mut f32,
        lda: i32,
        tau: *mut f32,
        workspace: *mut f32,
        lwork: i32,
        info: *mut i32,
    ) -> Result<()> {
        unsafe { check(cusolverDnSgeqrf(self.handle, m, n, a, lda, tau, workspace, lwork, info)) }
    }

    /// Get the buffer size for LU factorization (double precision).
    pub fn dgetrf_buffer_size(&self, m: i32, n: i32, a: *const f64, lda: i32) -> Result<i32> {
        let mut lwork = 0;
        unsafe { check(cusolverDnDgetrf_bufferSize(self.handle, m, n, a as *mut _, lda, &mut lwork))? };
        Ok(lwork)
    }

    /// LU factorization (double precision).
    pub fn dgetrf(
        &self,
        m: i32,
        n: i32,
        a: *mut f64,
        lda: i32,
        workspace: *mut f64,
        ipiv: *mut i32,
        info: *mut i32,
    ) -> Result<()> {
        unsafe { check(cusolverDnDgetrf(self.handle, m, n, a, lda, workspace, ipiv, info)) }
    }

    /// Solve linear system using LU factorization (double precision).
    pub fn dgetrs(
        &self,
        trans: i32,
        n: i32,
        nrhs: i32,
        a: *const f64,
        lda: i32,
        ipiv: *const i32,
        b: *mut f64,
        ldb: i32,
        info: *mut i32,
    ) -> Result<()> {
        unsafe { check(cusolverDnDgetrs(self.handle, trans, n, nrhs, a, lda, ipiv, b, ldb, info)) }
    }

    /// Get the raw handle.
    pub fn as_raw(&self) -> cusolverDnHandle_t {
        self.handle
    }
}

impl Drop for DnHandle {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { cusolverDnDestroy(self.handle) };
        }
    }
}

unsafe impl Send for DnHandle {}
unsafe impl Sync for DnHandle {}

/// Fill mode for symmetric/triangular matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillMode {
    Lower,
    Upper,
}

impl FillMode {
    pub fn to_cusolver(self) -> i32 {
        match self {
            FillMode::Lower => CUBLAS_FILL_MODE_LOWER,
            FillMode::Upper => CUBLAS_FILL_MODE_UPPER,
        }
    }
}

/// Transpose operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    None,
    Transpose,
    ConjugateTranspose,
}

impl Operation {
    pub fn to_cusolver(self) -> i32 {
        match self {
            Operation::None => CUBLAS_OP_N,
            Operation::Transpose => CUBLAS_OP_T,
            Operation::ConjugateTranspose => CUBLAS_OP_C,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_handle() {
        let _ = DnHandle::new();
    }
}
