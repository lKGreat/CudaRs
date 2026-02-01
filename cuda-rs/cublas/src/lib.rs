//! Safe Rust wrapper for cuBLAS.

use cublas_sys::*;
use cuda_runtime::{DeviceBuffer, Stream};
use std::ptr;
use thiserror::Error;

#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
#[error("cuBLAS Error: {0}")]
pub struct CublasError(pub i32);

pub type Result<T> = std::result::Result<T, CublasError>;

#[inline]
fn check(code: cublasStatus_t) -> Result<()> {
    if code == CUBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(CublasError(code))
    }
}

/// cuBLAS operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    None,
    Transpose,
    ConjugateTranspose,
}

impl Operation {
    fn to_cublas(self) -> cublasOperation_t {
        match self {
            Operation::None => CUBLAS_OP_N,
            Operation::Transpose => CUBLAS_OP_T,
            Operation::ConjugateTranspose => CUBLAS_OP_C,
        }
    }
}

/// cuBLAS Handle wrapper with automatic resource management.
pub struct CublasHandle {
    handle: cublasHandle_t,
}

impl CublasHandle {
    /// Create a new cuBLAS handle.
    pub fn new() -> Result<Self> {
        let mut handle = ptr::null_mut();
        unsafe { check(cublasCreate(&mut handle))? };
        Ok(Self { handle })
    }

    /// Set the stream for this handle.
    pub fn set_stream(&self, stream: &Stream) -> Result<()> {
        unsafe { check(cublasSetStream(self.handle, stream.as_raw())) }
    }

    /// Get cuBLAS version.
    pub fn version(&self) -> Result<i32> {
        let mut version = 0;
        unsafe { check(cublasGetVersion(self.handle, &mut version))? };
        Ok(version)
    }

    /// SGEMM: Single precision general matrix multiply.
    /// C = alpha * op(A) * op(B) + beta * C
    pub fn sgemm(
        &self,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: &DeviceBuffer<f32>,
        lda: i32,
        b: &DeviceBuffer<f32>,
        ldb: i32,
        beta: f32,
        c: &mut DeviceBuffer<f32>,
        ldc: i32,
    ) -> Result<()> {
        unsafe {
            check(cublasSgemm(
                self.handle,
                transa.to_cublas(),
                transb.to_cublas(),
                m, n, k,
                &alpha,
                a.as_ptr(),
                lda,
                b.as_ptr(),
                ldb,
                &beta,
                c.as_mut_ptr(),
                ldc,
            ))
        }
    }

    /// DGEMM: Double precision general matrix multiply.
    /// C = alpha * op(A) * op(B) + beta * C
    pub fn dgemm(
        &self,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: &DeviceBuffer<f64>,
        lda: i32,
        b: &DeviceBuffer<f64>,
        ldb: i32,
        beta: f64,
        c: &mut DeviceBuffer<f64>,
        ldc: i32,
    ) -> Result<()> {
        unsafe {
            check(cublasDgemm(
                self.handle,
                transa.to_cublas(),
                transb.to_cublas(),
                m, n, k,
                &alpha,
                a.as_ptr(),
                lda,
                b.as_ptr(),
                ldb,
                &beta,
                c.as_mut_ptr(),
                ldc,
            ))
        }
    }

    /// SAXPY: y = alpha * x + y (single precision)
    pub fn saxpy(
        &self,
        n: i32,
        alpha: f32,
        x: &DeviceBuffer<f32>,
        incx: i32,
        y: &mut DeviceBuffer<f32>,
        incy: i32,
    ) -> Result<()> {
        unsafe {
            check(cublasSaxpy(
                self.handle,
                n,
                &alpha,
                x.as_ptr(),
                incx,
                y.as_mut_ptr(),
                incy,
            ))
        }
    }

    /// DAXPY: y = alpha * x + y (double precision)
    pub fn daxpy(
        &self,
        n: i32,
        alpha: f64,
        x: &DeviceBuffer<f64>,
        incx: i32,
        y: &mut DeviceBuffer<f64>,
        incy: i32,
    ) -> Result<()> {
        unsafe {
            check(cublasDaxpy(
                self.handle,
                n,
                &alpha,
                x.as_ptr(),
                incx,
                y.as_mut_ptr(),
                incy,
            ))
        }
    }

    /// SDOT: Dot product (single precision)
    pub fn sdot(
        &self,
        n: i32,
        x: &DeviceBuffer<f32>,
        incx: i32,
        y: &DeviceBuffer<f32>,
        incy: i32,
    ) -> Result<f32> {
        let mut result = 0.0f32;
        unsafe {
            check(cublasSdot(
                self.handle,
                n,
                x.as_ptr(),
                incx,
                y.as_ptr(),
                incy,
                &mut result,
            ))?;
        }
        Ok(result)
    }

    /// DDOT: Dot product (double precision)
    pub fn ddot(
        &self,
        n: i32,
        x: &DeviceBuffer<f64>,
        incx: i32,
        y: &DeviceBuffer<f64>,
        incy: i32,
    ) -> Result<f64> {
        let mut result = 0.0f64;
        unsafe {
            check(cublasDdot(
                self.handle,
                n,
                x.as_ptr(),
                incx,
                y.as_ptr(),
                incy,
                &mut result,
            ))?;
        }
        Ok(result)
    }

    /// SNRM2: Euclidean norm (single precision)
    pub fn snrm2(&self, n: i32, x: &DeviceBuffer<f32>, incx: i32) -> Result<f32> {
        let mut result = 0.0f32;
        unsafe {
            check(cublasSnrm2(self.handle, n, x.as_ptr(), incx, &mut result))?;
        }
        Ok(result)
    }

    /// DNRM2: Euclidean norm (double precision)
    pub fn dnrm2(&self, n: i32, x: &DeviceBuffer<f64>, incx: i32) -> Result<f64> {
        let mut result = 0.0f64;
        unsafe {
            check(cublasDnrm2(self.handle, n, x.as_ptr(), incx, &mut result))?;
        }
        Ok(result)
    }

    /// SSCAL: Scale vector (single precision)
    pub fn sscal(&self, n: i32, alpha: f32, x: &mut DeviceBuffer<f32>, incx: i32) -> Result<()> {
        unsafe { check(cublasSscal(self.handle, n, &alpha, x.as_mut_ptr(), incx)) }
    }

    /// DSCAL: Scale vector (double precision)
    pub fn dscal(&self, n: i32, alpha: f64, x: &mut DeviceBuffer<f64>, incx: i32) -> Result<()> {
        unsafe { check(cublasDscal(self.handle, n, &alpha, x.as_mut_ptr(), incx)) }
    }

    /// Get the raw handle.
    pub fn as_raw(&self) -> cublasHandle_t {
        self.handle
    }
}

impl Drop for CublasHandle {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { cublasDestroy(self.handle) };
        }
    }
}

unsafe impl Send for CublasHandle {}
unsafe impl Sync for CublasHandle {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_handle() {
        // This will fail if no CUDA device is available
        let _ = CublasHandle::new();
    }
}
