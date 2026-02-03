//! CudaRS FFI - C Foreign Function Interface for CUDA operations.
//!
//! This crate provides a C-compatible API for all CUDA operations,
//! designed for consumption by C# and other languages via P/Invoke.

#![allow(clippy::missing_safety_doc)]

use libc::c_char;

mod runtime;
mod driver;
mod memory_pool;
mod preprocess_gpu;
mod image_decode;
mod sdk;
#[cfg(feature = "onnxruntime")]
mod onnx_runtime;
#[cfg(not(feature = "onnxruntime"))]
mod onnx_runtime_stub;
mod trtexec;
#[cfg(feature = "tensorrt")]
mod tensorrt;
#[cfg(not(feature = "tensorrt"))]
mod tensorrt_stub_api;
#[cfg(feature = "tensorrt-stub")]
mod tensorrt_stubs;
#[cfg(feature = "torchscript")]
mod torchscript;
#[cfg(feature = "openvino")]
mod openvino;
#[cfg(feature = "blas")]
mod blas;
#[cfg(feature = "fft")]
mod fft;
#[cfg(feature = "rand")]
mod rand_api;
#[cfg(feature = "sparse")]
mod sparse;
#[cfg(feature = "solver")]
mod solver;
#[cfg(feature = "dnn")]
mod dnn;
#[cfg(feature = "rtc")]
mod rtc;
#[cfg(feature = "management")]
mod management;

pub use runtime::*;
pub use driver::*;
pub use memory_pool::*;
pub use preprocess_gpu::*;
pub use image_decode::*;
pub use sdk::*;
#[cfg(feature = "onnxruntime")]
pub use onnx_runtime::*;
#[cfg(not(feature = "onnxruntime"))]
pub use onnx_runtime_stub::*;
pub use trtexec::*;
#[cfg(feature = "tensorrt")]
pub use tensorrt::*;
#[cfg(not(feature = "tensorrt"))]
pub use tensorrt_stub_api::*;
#[cfg(feature = "torchscript")]
pub use torchscript::*;
#[cfg(feature = "openvino")]
pub use openvino::*;
#[cfg(feature = "blas")]
pub use blas::*;
#[cfg(feature = "fft")]
pub use fft::*;
#[cfg(feature = "rand")]
pub use rand_api::*;
#[cfg(feature = "sparse")]
pub use sparse::*;
#[cfg(feature = "solver")]
pub use solver::*;
#[cfg(feature = "dnn")]
pub use dnn::*;
#[cfg(feature = "rtc")]
pub use rtc::*;
#[cfg(feature = "management")]
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
