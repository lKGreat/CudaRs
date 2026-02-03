//! ONNX Runtime stubs (when feature disabled).

use crate::CudaRsResult;
use libc::{c_char, c_int, c_ulonglong};

/// Opaque handle for ONNX Runtime session.
pub type CudaRsOnnxSession = u64;

#[repr(C)]
pub struct CudaRsTensor {
    pub data: *mut f32,
    pub data_len: c_ulonglong,
    pub shape: *mut i64,
    pub shape_len: c_ulonglong,
}

/// Create an ONNX Runtime session (unsupported).
#[no_mangle]
pub extern "C" fn cudars_onnx_create(
    _model_path: *const c_char,
    _device_id: c_int,
    _out_handle: *mut CudaRsOnnxSession,
) -> CudaRsResult {
    CudaRsResult::ErrorNotSupported
}

/// Destroy an ONNX Runtime session (unsupported).
#[no_mangle]
pub extern "C" fn cudars_onnx_destroy(_handle: CudaRsOnnxSession) -> CudaRsResult {
    CudaRsResult::ErrorNotSupported
}

/// Run an ONNX Runtime session (unsupported).
#[no_mangle]
pub extern "C" fn cudars_onnx_run(
    _handle: CudaRsOnnxSession,
    _input_ptr: *const f32,
    _input_len: c_ulonglong,
    _shape_ptr: *const i64,
    _shape_len: c_ulonglong,
    _out_tensors: *mut *mut CudaRsTensor,
    _out_count: *mut c_ulonglong,
) -> CudaRsResult {
    CudaRsResult::ErrorNotSupported
}

/// Free tensors returned by cudars_onnx_run (unsupported).
#[no_mangle]
pub extern "C" fn cudars_onnx_free_tensors(_tensors: *mut CudaRsTensor, _count: c_ulonglong) -> CudaRsResult {
    CudaRsResult::ErrorNotSupported
}
