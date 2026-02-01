//! Safe Rust wrapper for cuSPARSE.

use cusparse_sys::*;
use cuda_runtime::Stream;
use std::ptr;
use thiserror::Error;

#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
#[error("cuSPARSE Error: {0}")]
pub struct CusparseError(pub i32);

pub type Result<T> = std::result::Result<T, CusparseError>;

#[inline]
fn check(code: cusparseStatus_t) -> Result<()> {
    if code == CUSPARSE_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(CusparseError(code))
    }
}

/// cuSPARSE Handle wrapper with automatic resource management.
pub struct SparseHandle {
    handle: cusparseHandle_t,
}

impl SparseHandle {
    /// Create a new cuSPARSE handle.
    pub fn new() -> Result<Self> {
        let mut handle = ptr::null_mut();
        unsafe { check(cusparseCreate(&mut handle))? };
        Ok(Self { handle })
    }

    /// Set the stream for this handle.
    pub fn set_stream(&self, stream: &Stream) -> Result<()> {
        unsafe { check(cusparseSetStream(self.handle, stream.as_raw())) }
    }

    /// Get the cuSPARSE version.
    pub fn version(&self) -> Result<i32> {
        let mut version = 0;
        unsafe { check(cusparseGetVersion(self.handle, &mut version))? };
        Ok(version)
    }

    /// Get the raw handle.
    pub fn as_raw(&self) -> cusparseHandle_t {
        self.handle
    }
}

impl Drop for SparseHandle {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { cusparseDestroy(self.handle) };
        }
    }
}

unsafe impl Send for SparseHandle {}
unsafe impl Sync for SparseHandle {}

/// Sparse matrix descriptor (generic).
pub struct SpMatDescr {
    handle: cusparseSpMatDescr_t,
}

impl SpMatDescr {
    /// Create a CSR matrix descriptor (single precision).
    pub fn create_csr_f32(
        rows: i64,
        cols: i64,
        nnz: i64,
        row_offsets: *mut i32,
        col_indices: *mut i32,
        values: *mut f32,
    ) -> Result<Self> {
        let mut handle = ptr::null_mut();
        unsafe {
            check(cusparseCreateCsr(
                &mut handle,
                rows,
                cols,
                nnz,
                row_offsets as *mut _,
                col_indices as *mut _,
                values as *mut _,
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO,
                CUDA_R_32F,
            ))?;
        }
        Ok(Self { handle })
    }

    /// Create a CSR matrix descriptor (double precision).
    pub fn create_csr_f64(
        rows: i64,
        cols: i64,
        nnz: i64,
        row_offsets: *mut i32,
        col_indices: *mut i32,
        values: *mut f64,
    ) -> Result<Self> {
        let mut handle = ptr::null_mut();
        unsafe {
            check(cusparseCreateCsr(
                &mut handle,
                rows,
                cols,
                nnz,
                row_offsets as *mut _,
                col_indices as *mut _,
                values as *mut _,
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO,
                CUDA_R_64F,
            ))?;
        }
        Ok(Self { handle })
    }

    /// Get the raw descriptor.
    pub fn as_raw(&self) -> cusparseSpMatDescr_t {
        self.handle
    }
}

impl Drop for SpMatDescr {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { cusparseDestroySpMat(self.handle) };
        }
    }
}

/// Dense vector descriptor.
pub struct DnVecDescr {
    handle: cusparseDnVecDescr_t,
}

impl DnVecDescr {
    /// Create a dense vector descriptor (single precision).
    pub fn create_f32(size: i64, values: *mut f32) -> Result<Self> {
        let mut handle = ptr::null_mut();
        unsafe {
            check(cusparseCreateDnVec(&mut handle, size, values as *mut _, CUDA_R_32F))?;
        }
        Ok(Self { handle })
    }

    /// Create a dense vector descriptor (double precision).
    pub fn create_f64(size: i64, values: *mut f64) -> Result<Self> {
        let mut handle = ptr::null_mut();
        unsafe {
            check(cusparseCreateDnVec(&mut handle, size, values as *mut _, CUDA_R_64F))?;
        }
        Ok(Self { handle })
    }

    /// Get the raw descriptor.
    pub fn as_raw(&self) -> cusparseDnVecDescr_t {
        self.handle
    }
}

impl Drop for DnVecDescr {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { cusparseDestroyDnVec(self.handle) };
        }
    }
}

/// Dense matrix descriptor.
pub struct DnMatDescr {
    handle: cusparseDnMatDescr_t,
}

impl DnMatDescr {
    /// Create a dense matrix descriptor (single precision, column-major).
    pub fn create_f32(rows: i64, cols: i64, ld: i64, values: *mut f32) -> Result<Self> {
        let mut handle = ptr::null_mut();
        unsafe {
            check(cusparseCreateDnMat(
                &mut handle,
                rows,
                cols,
                ld,
                values as *mut _,
                CUDA_R_32F,
                CUSPARSE_ORDER_COL,
            ))?;
        }
        Ok(Self { handle })
    }

    /// Create a dense matrix descriptor (double precision, column-major).
    pub fn create_f64(rows: i64, cols: i64, ld: i64, values: *mut f64) -> Result<Self> {
        let mut handle = ptr::null_mut();
        unsafe {
            check(cusparseCreateDnMat(
                &mut handle,
                rows,
                cols,
                ld,
                values as *mut _,
                CUDA_R_64F,
                CUSPARSE_ORDER_COL,
            ))?;
        }
        Ok(Self { handle })
    }

    /// Get the raw descriptor.
    pub fn as_raw(&self) -> cusparseDnMatDescr_t {
        self.handle
    }
}

impl Drop for DnMatDescr {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { cusparseDestroyDnMat(self.handle) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_handle() {
        let _ = SparseHandle::new();
    }
}
