//! TensorRT API stubs used when the real TensorRT backend is not enabled.
//!
//! These functions preserve the C ABI but always report "not supported".

use libc::{c_char, c_int, c_ulonglong, c_void};

use crate::{CudaRsEvent, CudaRsResult, CudaRsStream};

/// Opaque handle for TensorRT engine (stub).
pub type CudaRsTrtEngine = u64;

/// TensorRT tensor descriptor for I/O (stub).
#[repr(C)]
pub struct CudaRsTrtTensor {
    pub data: *mut f32,
    pub data_len: c_ulonglong,
    pub shape: *mut i64,
    pub shape_len: c_ulonglong,
}

/// TensorRT build configuration (stub).
#[repr(C)]
pub struct CudaRsTrtBuildConfig {
    pub fp16_enabled: c_int,
    pub int8_enabled: c_int,
    pub max_batch_size: c_int,
    pub workspace_size_mb: c_int,
    pub dla_core: c_int,
}

impl Default for CudaRsTrtBuildConfig {
    fn default() -> Self {
        Self {
            fp16_enabled: 0,
            int8_enabled: 0,
            max_batch_size: 1,
            workspace_size_mb: 1024,
            dla_core: -1,
        }
    }
}

#[no_mangle]
pub extern "C" fn cudars_trt_build_engine(
    onnx_path: *const c_char,
    _device_id: c_int,
    _config: *const CudaRsTrtBuildConfig,
    out_handle: *mut CudaRsTrtEngine,
) -> CudaRsResult {
    if onnx_path.is_null() || out_handle.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }
    unsafe { *out_handle = 0 };
    CudaRsResult::ErrorNotSupported
}

#[no_mangle]
pub extern "C" fn cudars_trt_load_engine(
    engine_path: *const c_char,
    _device_id: c_int,
    out_handle: *mut CudaRsTrtEngine,
) -> CudaRsResult {
    if engine_path.is_null() || out_handle.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }
    unsafe { *out_handle = 0 };
    CudaRsResult::ErrorNotSupported
}

#[no_mangle]
pub extern "C" fn cudars_trt_save_engine(
    _handle: CudaRsTrtEngine,
    path: *const c_char,
) -> CudaRsResult {
    if path.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }
    CudaRsResult::ErrorNotSupported
}

#[no_mangle]
pub extern "C" fn cudars_trt_run(
    _handle: CudaRsTrtEngine,
    input_ptr: *const f32,
    _input_len: c_ulonglong,
    out_tensors: *mut *mut CudaRsTrtTensor,
    out_count: *mut c_ulonglong,
) -> CudaRsResult {
    if input_ptr.is_null() || out_tensors.is_null() || out_count.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }
    unsafe {
        *out_tensors = std::ptr::null_mut();
        *out_count = 0;
    }
    CudaRsResult::ErrorNotSupported
}

#[no_mangle]
pub extern "C" fn cudars_trt_run_on_stream(
    _handle: CudaRsTrtEngine,
    input_ptr: *const f32,
    _input_len: c_ulonglong,
    _stream: CudaRsStream,
    out_tensors: *mut *mut CudaRsTrtTensor,
    out_count: *mut c_ulonglong,
) -> CudaRsResult {
    if input_ptr.is_null() || out_tensors.is_null() || out_count.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }
    unsafe {
        *out_tensors = std::ptr::null_mut();
        *out_count = 0;
    }
    CudaRsResult::ErrorNotSupported
}

#[no_mangle]
pub extern "C" fn cudars_trt_get_output_count(
    _handle: CudaRsTrtEngine,
    out_count: *mut c_int,
) -> CudaRsResult {
    if out_count.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }
    unsafe { *out_count = 0 };
    CudaRsResult::ErrorNotSupported
}

#[no_mangle]
pub extern "C" fn cudars_trt_get_output_device_ptr(
    _handle: CudaRsTrtEngine,
    _index: c_int,
    out_ptr: *mut *mut c_void,
    out_bytes: *mut c_ulonglong,
) -> CudaRsResult {
    if out_ptr.is_null() || out_bytes.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }
    unsafe {
        *out_ptr = std::ptr::null_mut();
        *out_bytes = 0;
    }
    CudaRsResult::ErrorNotSupported
}

#[no_mangle]
pub extern "C" fn cudars_trt_get_input_device_ptr(
    _handle: CudaRsTrtEngine,
    _index: c_int,
    out_ptr: *mut *mut c_void,
    out_bytes: *mut c_ulonglong,
) -> CudaRsResult {
    if out_ptr.is_null() || out_bytes.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }
    unsafe {
        *out_ptr = std::ptr::null_mut();
        *out_bytes = 0;
    }
    CudaRsResult::ErrorNotSupported
}

#[no_mangle]
pub extern "C" fn cudars_trt_enqueue_device(
    _handle: CudaRsTrtEngine,
    input_device_ptr: *const f32,
    _input_len: c_ulonglong,
    _stream: CudaRsStream,
    _done_event: CudaRsEvent,
) -> CudaRsResult {
    if input_device_ptr.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }
    CudaRsResult::ErrorNotSupported
}

#[no_mangle]
pub extern "C" fn cudars_trt_get_input_info(
    _handle: CudaRsTrtEngine,
    _index: c_int,
    out_shape: *mut i64,
    out_shape_len: *mut c_int,
    _max_shape_len: c_int,
) -> CudaRsResult {
    if out_shape.is_null() || out_shape_len.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }
    unsafe {
        *out_shape_len = 0;
    }
    CudaRsResult::ErrorNotSupported
}

#[no_mangle]
pub extern "C" fn cudars_trt_get_output_info(
    _handle: CudaRsTrtEngine,
    _index: c_int,
    out_shape: *mut i64,
    out_shape_len: *mut c_int,
    _max_shape_len: c_int,
) -> CudaRsResult {
    if out_shape.is_null() || out_shape_len.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }
    unsafe {
        *out_shape_len = 0;
    }
    CudaRsResult::ErrorNotSupported
}

#[no_mangle]
pub extern "C" fn cudars_trt_destroy(_handle: CudaRsTrtEngine) -> CudaRsResult {
    CudaRsResult::ErrorNotSupported
}

#[no_mangle]
pub extern "C" fn cudars_trt_free_tensors(
    tensors: *mut CudaRsTrtTensor,
    _count: c_ulonglong,
) -> CudaRsResult {
    if tensors.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }
    CudaRsResult::ErrorNotSupported
}
