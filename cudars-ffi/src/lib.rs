//! CudaRS FFI - C Foreign Function Interface for CUDA operations.
//!
//! This crate provides a C-compatible API for all CUDA operations,
//! designed for consumption by C# and other languages via P/Invoke.

#![allow(clippy::missing_safety_doc)]

use libc::c_char;

mod runtime;
mod driver;
mod blas;
mod fft;
mod rand_api;
mod sparse;
mod solver;
mod dnn;
mod rtc;
mod management;

pub use runtime::*;
pub use driver::*;
pub use blas::*;
pub use fft::*;
pub use rand_api::*;
pub use sparse::*;
pub use solver::*;
pub use dnn::*;
pub use rtc::*;
pub use management::*;

/// Result code for all CudaRS operations.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaRsResult {
    Success = 0,
    ErrorInvalidValue = 1,
    ErrorOutOfMemory = 2,
    ErrorNotInitialized = 3,
    ErrorInvalidHandle = 4,
    ErrorNotSupported = 5,
    ErrorUnknown = 999,
}

/// Get the version string of CudaRS.
#[no_mangle]
pub extern "C" fn cudars_get_version() -> *const c_char {
    static VERSION: &[u8] = b"0.1.0\0";
    VERSION.as_ptr() as *const c_char
}

/// Get error message for a result code.
#[no_mangle]
pub extern "C" fn cudars_get_error_string(result: CudaRsResult) -> *const c_char {
    match result {
        CudaRsResult::Success => b"Success\0".as_ptr() as *const c_char,
        CudaRsResult::ErrorInvalidValue => b"Invalid value\0".as_ptr() as *const c_char,
        CudaRsResult::ErrorOutOfMemory => b"Out of memory\0".as_ptr() as *const c_char,
        CudaRsResult::ErrorNotInitialized => b"Not initialized\0".as_ptr() as *const c_char,
        CudaRsResult::ErrorInvalidHandle => b"Invalid handle\0".as_ptr() as *const c_char,
        CudaRsResult::ErrorNotSupported => b"Not supported\0".as_ptr() as *const c_char,
        CudaRsResult::ErrorUnknown => b"Unknown error\0".as_ptr() as *const c_char,
    }
}
