//! TensorRT C API helper functions.
//!
//! These are thin wrapper functions around TensorRT's C++ API exposed as C functions.
//! In a real build, these would be implemented in a separate C++ file and linked.
//! This file provides the Rust-side extern declarations.

use libc::{c_char, c_int, c_ulonglong, c_void};

// Note: These functions need to be implemented in a C++ wrapper library
// that links against nvinfer and nvonnxparser.
// The implementations below are stubs for compilation purposes.

/// Create a TensorRT logger (stub implementation for linking).
#[no_mangle]
#[cfg(feature = "tensorrt")]
pub extern "C" fn trt_create_logger() -> *mut c_void {
    // In real implementation, this would create nvinfer::ILogger
    std::ptr::null_mut()
}

/// Destroy TensorRT runtime.
#[no_mangle]
#[cfg(feature = "tensorrt")]
pub extern "C" fn trt_destroy_runtime(_runtime: *mut c_void) {}

/// Get number of bindings in engine.
#[no_mangle]
#[cfg(feature = "tensorrt")]
pub extern "C" fn trt_engine_get_nb_bindings(_engine: *mut c_void) -> c_int {
    0
}

/// Get binding name by index.
#[no_mangle]
#[cfg(feature = "tensorrt")]
pub extern "C" fn trt_engine_get_binding_name(
    _engine: *mut c_void,
    _index: c_int,
) -> *const c_char {
    std::ptr::null()
}

/// Check if binding is input.
#[no_mangle]
#[cfg(feature = "tensorrt")]
pub extern "C" fn trt_engine_binding_is_input(_engine: *mut c_void, _index: c_int) -> c_int {
    0
}

/// Get binding dimensions.
#[no_mangle]
#[cfg(feature = "tensorrt")]
pub extern "C" fn trt_engine_get_binding_dimensions(
    _engine: *mut c_void,
    _index: c_int,
    _dims: *mut i64,
    _max_dims: c_int,
) -> c_int {
    0
}

/// Destroy engine.
#[no_mangle]
#[cfg(feature = "tensorrt")]
pub extern "C" fn trt_destroy_engine(_engine: *mut c_void) {}

/// Create execution context from engine.
#[no_mangle]
#[cfg(feature = "tensorrt")]
pub extern "C" fn trt_engine_create_execution_context(_engine: *mut c_void) -> *mut c_void {
    std::ptr::null_mut()
}

/// Enqueue inference (v2 API).
#[no_mangle]
#[cfg(feature = "tensorrt")]
pub extern "C" fn trt_context_enqueue_v2(
    _context: *mut c_void,
    _bindings: *const *mut c_void,
    _stream: *mut c_void,
) -> c_int {
    0
}

/// Destroy execution context.
#[no_mangle]
#[cfg(feature = "tensorrt")]
pub extern "C" fn trt_destroy_execution_context(_context: *mut c_void) {}

/// Deserialize CUDA engine from memory.
#[no_mangle]
#[cfg(feature = "tensorrt")]
pub extern "C" fn trt_runtime_deserialize_cuda_engine(
    _runtime: *mut c_void,
    _data: *const c_void,
    _size: c_ulonglong,
) -> *mut c_void {
    std::ptr::null_mut()
}

/// Serialize engine to memory.
#[no_mangle]
#[cfg(feature = "tensorrt")]
pub extern "C" fn trt_engine_serialize(
    _engine: *mut c_void,
    _size: *mut c_ulonglong,
) -> *mut c_void {
    std::ptr::null_mut()
}

/// Free serialized engine data.
#[no_mangle]
#[cfg(feature = "tensorrt")]
pub extern "C" fn trt_free_serialized(_data: *mut c_void) {}

/// Create network definition from builder.
#[no_mangle]
#[cfg(feature = "tensorrt")]
pub extern "C" fn trt_builder_create_network(
    _builder: *mut c_void,
    _flags: c_int,
) -> *mut c_void {
    std::ptr::null_mut()
}

/// Create builder config.
#[no_mangle]
#[cfg(feature = "tensorrt")]
pub extern "C" fn trt_builder_create_builder_config(_builder: *mut c_void) -> *mut c_void {
    std::ptr::null_mut()
}

/// Build serialized network.
#[no_mangle]
#[cfg(feature = "tensorrt")]
pub extern "C" fn trt_builder_build_serialized_network(
    _builder: *mut c_void,
    _network: *mut c_void,
    _config: *mut c_void,
    _size: *mut c_ulonglong,
) -> *mut c_void {
    std::ptr::null_mut()
}

/// Destroy builder.
#[no_mangle]
#[cfg(feature = "tensorrt")]
pub extern "C" fn trt_destroy_builder(_builder: *mut c_void) {}

/// Destroy network definition.
#[no_mangle]
#[cfg(feature = "tensorrt")]
pub extern "C" fn trt_destroy_network(_network: *mut c_void) {}

/// Destroy builder config.
#[no_mangle]
#[cfg(feature = "tensorrt")]
pub extern "C" fn trt_destroy_builder_config(_config: *mut c_void) {}

/// Set max workspace size.
#[no_mangle]
#[cfg(feature = "tensorrt")]
pub extern "C" fn trt_config_set_max_workspace_size(_config: *mut c_void, _size: c_ulonglong) {}

/// Set builder flag.
#[no_mangle]
#[cfg(feature = "tensorrt")]
pub extern "C" fn trt_config_set_flag(_config: *mut c_void, _flag: c_int) {}

/// Set DLA core.
#[no_mangle]
#[cfg(feature = "tensorrt")]
pub extern "C" fn trt_config_set_dla_core(_config: *mut c_void, _core: c_int) {}

/// Parse ONNX file.
#[no_mangle]
#[cfg(feature = "tensorrt")]
pub extern "C" fn trt_onnx_parser_parse_from_file(
    _parser: *mut c_void,
    _filepath: *const c_char,
    _verbosity: c_int,
) -> c_int {
    0
}

/// Destroy ONNX parser.
#[no_mangle]
#[cfg(feature = "tensorrt")]
pub extern "C" fn trt_destroy_onnx_parser(_parser: *mut c_void) {}
