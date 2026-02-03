//! TensorRT FFI exports (Rust-side implementation).
//!
//! Provides TensorRT engine building, loading, and inference via C FFI.
//! Supports both building engines from ONNX and loading serialized engines.

use crate::CudaRsResult;
use libc::{c_char, c_int, c_ulonglong, c_void};
use std::ffi::{CStr, CString};
use std::ptr;
use std::sync::{Arc, Mutex};

use crate::runtime::{CudaRsEvent, CudaRsStream, EVENTS, STREAMS};

use crate::runtime::HandleManager;

// TensorRT C API bindings (linked at runtime)
#[link(name = "nvinfer")]
extern "C" {
    fn createInferBuilder_INTERNAL(logger: *mut c_void, version: c_int) -> *mut c_void;
    fn createInferRuntime_INTERNAL(logger: *mut c_void, version: c_int) -> *mut c_void;
}

#[link(name = "nvonnxparser")]
extern "C" {
    fn createNvOnnxParser_INTERNAL(
        network: *mut c_void,
        logger: *mut c_void,
        version: c_int,
    ) -> *mut c_void;
}

/// TensorRT API version constant
const NV_TENSORRT_VERSION: c_int = 101501; // TensorRT 10.15.1

/// Opaque handle for TensorRT engine
pub type CudaRsTrtEngine = u64;

/// TensorRT tensor descriptor for I/O
#[repr(C)]
pub struct CudaRsTrtTensor {
    pub data: *mut f32,
    pub data_len: c_ulonglong,
    pub shape: *mut i64,
    pub shape_len: c_ulonglong,
}

/// TensorRT build configuration
#[repr(C)]
pub struct CudaRsTrtBuildConfig {
    pub fp16_enabled: c_int,
    pub int8_enabled: c_int,
    pub max_batch_size: c_int,
    pub workspace_size_mb: c_int,
    pub dla_core: c_int, // -1 = disabled
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

/// Internal TensorRT engine wrapper
struct TrtEngine {
    runtime_ptr: *mut c_void,
    engine_ptr: *mut c_void,
    context_ptr: *mut c_void,
    input_bindings: Vec<TrtBinding>,
    output_bindings: Vec<TrtBinding>,
    device_id: i32,
    stream: cuda_runtime_sys::cudaStream_t,
}

struct TrtBinding {
    name: String,
    shape: Vec<i64>,
    size: usize,
    is_input: bool,
    device_memory: *mut c_void,
}

// TensorRT pointers are thread-safe when used correctly
unsafe impl Send for TrtEngine {}
unsafe impl Sync for TrtEngine {}

impl Drop for TrtEngine {
    fn drop(&mut self) {
        unsafe {
            // Free device memory for bindings
            for binding in self.input_bindings.iter().chain(self.output_bindings.iter()) {
                if !binding.device_memory.is_null() {
                    cuda_runtime_sys::cudaFree(binding.device_memory);
                }
            }

            // Destroy TensorRT objects in reverse order
            if !self.context_ptr.is_null() {
                trt_destroy_execution_context(self.context_ptr);
            }
            if !self.engine_ptr.is_null() {
                trt_destroy_engine(self.engine_ptr);
            }
            if !self.runtime_ptr.is_null() {
                trt_destroy_runtime(self.runtime_ptr);
            }
            if !self.stream.is_null() {
                cuda_runtime_sys::cudaStreamDestroy(self.stream);
            }
        }
    }
}

lazy_static::lazy_static! {
    static ref TRT_ENGINES: Mutex<HandleManager<Arc<Mutex<TrtEngine>>>> = Mutex::new(HandleManager::new());
    static ref TRT_LOGGER: Mutex<TrtLogger> = Mutex::new(TrtLogger::new());
}

/// Simple TensorRT logger implementation
struct TrtLogger {
    logger_ptr: *mut c_void,
}

unsafe impl Send for TrtLogger {}
unsafe impl Sync for TrtLogger {}

impl TrtLogger {
    fn new() -> Self {
        let logger_ptr = unsafe { trt_create_logger() };
        Self { logger_ptr }
    }

    fn ptr(&self) -> *mut c_void {
        self.logger_ptr
    }
}

// Internal TensorRT C API wrappers
extern "C" {
    // Logger
    fn trt_create_logger() -> *mut c_void;

    // Runtime
    fn trt_destroy_runtime(runtime: *mut c_void);

    // Engine
    fn trt_engine_get_nb_bindings(engine: *mut c_void) -> c_int;
    fn trt_engine_get_binding_name(engine: *mut c_void, index: c_int) -> *const c_char;
    fn trt_engine_binding_is_input(engine: *mut c_void, index: c_int) -> c_int;
    fn trt_engine_get_binding_dimensions(
        engine: *mut c_void,
        index: c_int,
        dims: *mut i64,
        max_dims: c_int,
    ) -> c_int;
    fn trt_destroy_engine(engine: *mut c_void);

    // Execution context
    fn trt_engine_create_execution_context(engine: *mut c_void) -> *mut c_void;
    fn trt_context_enqueue_v2(
        context: *mut c_void,
        bindings: *const *mut c_void,
        stream: *mut c_void,
    ) -> c_int;
    fn trt_destroy_execution_context(context: *mut c_void);

    // Serialization
    fn trt_runtime_deserialize_cuda_engine(
        runtime: *mut c_void,
        data: *const c_void,
        size: c_ulonglong,
    ) -> *mut c_void;
    fn trt_engine_serialize(engine: *mut c_void, size: *mut c_ulonglong) -> *mut c_void;
    fn trt_free_serialized(data: *mut c_void);

    // Builder
    fn trt_builder_create_network(builder: *mut c_void, flags: c_int) -> *mut c_void;
    fn trt_builder_create_builder_config(builder: *mut c_void) -> *mut c_void;
    fn trt_builder_build_serialized_network(
        builder: *mut c_void,
        network: *mut c_void,
        config: *mut c_void,
        size: *mut c_ulonglong,
    ) -> *mut c_void;
    fn trt_destroy_builder(builder: *mut c_void);
    fn trt_destroy_network(network: *mut c_void);
    fn trt_destroy_builder_config(config: *mut c_void);

    // Builder config
    fn trt_config_set_max_workspace_size(config: *mut c_void, size: c_ulonglong);
    fn trt_config_set_flag(config: *mut c_void, flag: c_int);
    fn trt_config_set_dla_core(config: *mut c_void, core: c_int);

    // ONNX parser
    fn trt_onnx_parser_parse_from_file(
        parser: *mut c_void,
        filepath: *const c_char,
        verbosity: c_int,
    ) -> c_int;
    fn trt_destroy_onnx_parser(parser: *mut c_void);
}

// TensorRT BuilderFlag enum values
const TRT_BUILDER_FLAG_FP16: c_int = 0;
const TRT_BUILDER_FLAG_INT8: c_int = 1;
const TRT_BUILDER_FLAG_STRICT_TYPES: c_int = 3;

// NetworkDefinitionCreationFlag
const TRT_EXPLICIT_BATCH: c_int = 1;

/// Build TensorRT engine from ONNX model.
#[no_mangle]
pub extern "C" fn cudars_trt_build_engine(
    onnx_path: *const c_char,
    device_id: c_int,
    config: *const CudaRsTrtBuildConfig,
    out_handle: *mut CudaRsTrtEngine,
) -> CudaRsResult {
    if onnx_path.is_null() || out_handle.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let path = unsafe {
        match CStr::from_ptr(onnx_path).to_str() {
            Ok(p) => p,
            Err(_) => return CudaRsResult::ErrorInvalidValue,
        }
    };

    let build_config = if config.is_null() {
        CudaRsTrtBuildConfig::default()
    } else {
        unsafe { ptr::read(config) }
    };

    // Set CUDA device
    unsafe {
        let result = cuda_runtime_sys::cudaSetDevice(device_id);
        if result != cuda_runtime_sys::cudaSuccess {
            return CudaRsResult::ErrorInvalidValue;
        }
    }

    let logger = TRT_LOGGER.lock().unwrap();

    unsafe {
        // Create builder
        let builder = createInferBuilder_INTERNAL(logger.ptr(), NV_TENSORRT_VERSION);
        if builder.is_null() {
            return CudaRsResult::ErrorNotInitialized;
        }

        // Create network with explicit batch
        let network = trt_builder_create_network(builder, TRT_EXPLICIT_BATCH);
        if network.is_null() {
            trt_destroy_builder(builder);
            return CudaRsResult::ErrorNotInitialized;
        }

        // Create ONNX parser
        let parser = createNvOnnxParser_INTERNAL(network, logger.ptr(), NV_TENSORRT_VERSION);
        if parser.is_null() {
            trt_destroy_network(network);
            trt_destroy_builder(builder);
            return CudaRsResult::ErrorNotInitialized;
        }

        // Parse ONNX file
        let path_cstr = match CString::new(path) {
            Ok(s) => s,
            Err(_) => {
                trt_destroy_onnx_parser(parser);
                trt_destroy_network(network);
                trt_destroy_builder(builder);
                return CudaRsResult::ErrorInvalidValue;
            }
        };

        let parse_result = trt_onnx_parser_parse_from_file(parser, path_cstr.as_ptr(), 3);
        if parse_result == 0 {
            trt_destroy_onnx_parser(parser);
            trt_destroy_network(network);
            trt_destroy_builder(builder);
            return CudaRsResult::ErrorInvalidValue;
        }

        // Create builder config
        let builder_config = trt_builder_create_builder_config(builder);
        if builder_config.is_null() {
            trt_destroy_onnx_parser(parser);
            trt_destroy_network(network);
            trt_destroy_builder(builder);
            return CudaRsResult::ErrorNotInitialized;
        }

        // Configure build options
        trt_config_set_max_workspace_size(
            builder_config,
            (build_config.workspace_size_mb as u64) * 1024 * 1024,
        );

        if build_config.fp16_enabled != 0 {
            trt_config_set_flag(builder_config, TRT_BUILDER_FLAG_FP16);
        }

        if build_config.int8_enabled != 0 {
            trt_config_set_flag(builder_config, TRT_BUILDER_FLAG_INT8);
        }

        if build_config.dla_core >= 0 {
            trt_config_set_dla_core(builder_config, build_config.dla_core);
        }

        // Build serialized network
        let mut serialized_size: c_ulonglong = 0;
        let serialized = trt_builder_build_serialized_network(
            builder,
            network,
            builder_config,
            &mut serialized_size,
        );

        // Clean up builder resources
        trt_destroy_builder_config(builder_config);
        trt_destroy_onnx_parser(parser);
        trt_destroy_network(network);
        trt_destroy_builder(builder);

        if serialized.is_null() {
            return CudaRsResult::ErrorUnknown;
        }

        // Deserialize to engine
        let result = deserialize_engine(serialized, serialized_size, device_id, out_handle);

        // Free serialized data
        trt_free_serialized(serialized);

        result
    }
}

/// Load TensorRT engine from serialized file.
#[no_mangle]
pub extern "C" fn cudars_trt_load_engine(
    engine_path: *const c_char,
    device_id: c_int,
    out_handle: *mut CudaRsTrtEngine,
) -> CudaRsResult {
    if engine_path.is_null() || out_handle.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let path = unsafe {
        match CStr::from_ptr(engine_path).to_str() {
            Ok(p) => p,
            Err(_) => return CudaRsResult::ErrorInvalidValue,
        }
    };

    // Read engine file
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(_) => return CudaRsResult::ErrorInvalidValue,
    };

    // Set CUDA device
    unsafe {
        let result = cuda_runtime_sys::cudaSetDevice(device_id);
        if result != cuda_runtime_sys::cudaSuccess {
            return CudaRsResult::ErrorInvalidValue;
        }
    }

    unsafe {
        deserialize_engine(
            data.as_ptr() as *const c_void,
            data.len() as c_ulonglong,
            device_id,
            out_handle,
        )
    }
}

/// Deserialize TensorRT engine from memory.
unsafe fn deserialize_engine(
    data: *const c_void,
    size: c_ulonglong,
    device_id: c_int,
    out_handle: *mut CudaRsTrtEngine,
) -> CudaRsResult {
    let logger = TRT_LOGGER.lock().unwrap();

    // Create runtime
    let runtime = createInferRuntime_INTERNAL(logger.ptr(), NV_TENSORRT_VERSION);
    if runtime.is_null() {
        return CudaRsResult::ErrorNotInitialized;
    }

    // Deserialize engine
    let engine = trt_runtime_deserialize_cuda_engine(runtime, data, size);
    if engine.is_null() {
        trt_destroy_runtime(runtime);
        return CudaRsResult::ErrorInvalidValue;
    }

    // Create execution context
    let context = trt_engine_create_execution_context(engine);
    if context.is_null() {
        trt_destroy_engine(engine);
        trt_destroy_runtime(runtime);
        return CudaRsResult::ErrorNotInitialized;
    }

    let mut stream: cuda_runtime_sys::cudaStream_t = std::ptr::null_mut();
    if cuda_runtime_sys::cudaStreamCreate(&mut stream) != cuda_runtime_sys::cudaSuccess {
        trt_destroy_execution_context(context);
        trt_destroy_engine(engine);
        trt_destroy_runtime(runtime);
        return CudaRsResult::ErrorUnknown;
    }

    // Get bindings info
    let nb_bindings = trt_engine_get_nb_bindings(engine);
    let mut input_bindings: Vec<TrtBinding> = Vec::new();
    let mut output_bindings: Vec<TrtBinding> = Vec::new();

    for i in 0..nb_bindings {
        let name_ptr = trt_engine_get_binding_name(engine, i);
        let name = if name_ptr.is_null() {
            format!("binding_{}", i)
        } else {
            CStr::from_ptr(name_ptr).to_string_lossy().into_owned()
        };

        let is_input = trt_engine_binding_is_input(engine, i) != 0;

        // Get dimensions
        let mut dims = [0i64; 8];
        let ndims = trt_engine_get_binding_dimensions(engine, i, dims.as_mut_ptr(), 8);
        let shape: Vec<i64> = dims[..ndims as usize].to_vec();

        // Calculate size
        let size: usize = shape.iter().map(|&d| d.max(1) as usize).product();
        let byte_size = size * std::mem::size_of::<f32>();

        // Allocate device memory
        let mut device_ptr: *mut c_void = ptr::null_mut();
        let cuda_result = cuda_runtime_sys::cudaMalloc(&mut device_ptr, byte_size);
        if cuda_result != cuda_runtime_sys::cudaSuccess {
            // Clean up already allocated
            for b in input_bindings.iter().chain(output_bindings.iter()) {
                if !b.device_memory.is_null() {
                    cuda_runtime_sys::cudaFree(b.device_memory);
                }
            }
            cuda_runtime_sys::cudaStreamDestroy(stream);
            trt_destroy_execution_context(context);
            trt_destroy_engine(engine);
            trt_destroy_runtime(runtime);
            return CudaRsResult::ErrorOutOfMemory;
        }

        let binding = TrtBinding {
            name,
            shape,
            size,
            is_input,
            device_memory: device_ptr,
        };

        if is_input {
            input_bindings.push(binding);
        } else {
            output_bindings.push(binding);
        }
    }

    let trt_engine = TrtEngine {
        runtime_ptr: runtime,
        engine_ptr: engine,
        context_ptr: context,
        input_bindings,
        output_bindings,
        device_id,
        stream,
    };

    let mut engines = TRT_ENGINES.lock().unwrap();
    let id = engines.insert(Arc::new(Mutex::new(trt_engine)));
    *out_handle = id;

    CudaRsResult::Success
}

/// Save TensorRT engine to file.
#[no_mangle]
pub extern "C" fn cudars_trt_save_engine(
    handle: CudaRsTrtEngine,
    path: *const c_char,
) -> CudaRsResult {
    if path.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let engine = {
        let engines = TRT_ENGINES.lock().unwrap();
        match engines.get(handle) {
            Some(e) => Arc::clone(e),
            None => return CudaRsResult::ErrorInvalidHandle,
        }
    };
    let engine = engine.lock().unwrap();

    let path_str = unsafe {
        match CStr::from_ptr(path).to_str() {
            Ok(p) => p,
            Err(_) => return CudaRsResult::ErrorInvalidValue,
        }
    };

    unsafe {
        let mut size: c_ulonglong = 0;
        let serialized = trt_engine_serialize(engine.engine_ptr, &mut size);
        if serialized.is_null() {
            return CudaRsResult::ErrorUnknown;
        }

        let data = std::slice::from_raw_parts(serialized as *const u8, size as usize);

        let result = match std::fs::write(path_str, data) {
            Ok(_) => CudaRsResult::Success,
            Err(_) => CudaRsResult::ErrorInvalidValue,
        };

        trt_free_serialized(serialized);

        result
    }
}

/// Run inference on TensorRT engine.
#[no_mangle]
pub extern "C" fn cudars_trt_run(
    handle: CudaRsTrtEngine,
    input_ptr: *const f32,
    input_len: c_ulonglong,
    out_tensors: *mut *mut CudaRsTrtTensor,
    out_count: *mut c_ulonglong,
) -> CudaRsResult {
    if input_ptr.is_null() || out_tensors.is_null() || out_count.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let engine = {
        let engines = TRT_ENGINES.lock().unwrap();
        match engines.get(handle) {
            Some(e) => Arc::clone(e),
            None => return CudaRsResult::ErrorInvalidHandle,
        }
    };
    let mut engine = engine.lock().unwrap();

    // Set device
    unsafe {
        cuda_runtime_sys::cudaSetDevice(engine.device_id);
    }

    // Verify input size
    if engine.input_bindings.is_empty() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let expected_size: usize = engine.input_bindings[0].size;
    if input_len as usize != expected_size {
        return CudaRsResult::ErrorInvalidValue;
    }

    unsafe {
        let stream = engine.stream;

        // Copy input to device (async on engine stream)
        let input_binding = &engine.input_bindings[0];
        let copy_result = cuda_runtime_sys::cudaMemcpyAsync(
            input_binding.device_memory,
            input_ptr as *const c_void,
            expected_size * std::mem::size_of::<f32>(),
            cuda_runtime_sys::cudaMemcpyHostToDevice,
            stream,
        );

        if copy_result != cuda_runtime_sys::cudaSuccess {
            return CudaRsResult::ErrorUnknown;
        }

        // Build bindings array (inputs first, then outputs)
        let total_bindings = engine.input_bindings.len() + engine.output_bindings.len();
        let mut bindings: Vec<*mut c_void> = Vec::with_capacity(total_bindings);

        for b in &engine.input_bindings {
            bindings.push(b.device_memory);
        }
        for b in &engine.output_bindings {
            bindings.push(b.device_memory);
        }

        // Execute inference (async on engine stream)
        let exec_result = trt_context_enqueue_v2(
            engine.context_ptr,
            bindings.as_ptr(),
            stream as *mut c_void,
        );

        if exec_result == 0 {
            return CudaRsResult::ErrorUnknown;
        }

        // Copy outputs back to host (async)
        let mut output_tensors: Vec<CudaRsTrtTensor> = Vec::with_capacity(engine.output_bindings.len());

        for binding in &engine.output_bindings {
            let data_vec: Vec<f32> = vec![0.0; binding.size];
            let mut data_box = data_vec.into_boxed_slice();

            let copy_result = cuda_runtime_sys::cudaMemcpyAsync(
                data_box.as_mut_ptr() as *mut c_void,
                binding.device_memory,
                binding.size * std::mem::size_of::<f32>(),
                cuda_runtime_sys::cudaMemcpyDeviceToHost,
                stream,
            );

            if copy_result != cuda_runtime_sys::cudaSuccess {
                return CudaRsResult::ErrorUnknown;
            }

            let shape_vec: Vec<i64> = binding.shape.clone();
            let shape_len = shape_vec.len() as u64;
            let data_len = binding.size as u64;

            let shape_box = shape_vec.into_boxed_slice();
            let shape_ptr = Box::into_raw(shape_box) as *mut i64;
            let data_ptr = Box::into_raw(data_box) as *mut f32;

            output_tensors.push(CudaRsTrtTensor {
                data: data_ptr,
                data_len,
                shape: shape_ptr,
                shape_len,
            });
        }

        if cuda_runtime_sys::cudaStreamSynchronize(stream) != cuda_runtime_sys::cudaSuccess {
            return CudaRsResult::ErrorUnknown;
        }

        let count = output_tensors.len() as u64;
        let boxed = output_tensors.into_boxed_slice();
        let ptr = Box::into_raw(boxed) as *mut CudaRsTrtTensor;

        *out_tensors = ptr;
        *out_count = count;
    }

    CudaRsResult::Success
}

/// Run inference on TensorRT engine using a caller-provided stream.
#[no_mangle]
pub extern "C" fn cudars_trt_run_on_stream(
    handle: CudaRsTrtEngine,
    input_ptr: *const f32,
    input_len: c_ulonglong,
    stream: CudaRsStream,
    out_tensors: *mut *mut CudaRsTrtTensor,
    out_count: *mut c_ulonglong,
) -> CudaRsResult {
    if input_ptr.is_null() || out_tensors.is_null() || out_count.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let stream_raw = {
        let streams = STREAMS.lock().unwrap();
        let s = match streams.get(stream) {
            Some(s) => s,
            None => return CudaRsResult::ErrorInvalidHandle,
        };
        s.as_raw()
    };

    let engine = {
        let engines = TRT_ENGINES.lock().unwrap();
        match engines.get(handle) {
            Some(e) => Arc::clone(e),
            None => return CudaRsResult::ErrorInvalidHandle,
        }
    };
    let mut engine = engine.lock().unwrap();

    // Set device
    unsafe {
        cuda_runtime_sys::cudaSetDevice(engine.device_id);
    }

    if engine.input_bindings.is_empty() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let expected_size: usize = engine.input_bindings[0].size;
    if input_len as usize != expected_size {
        return CudaRsResult::ErrorInvalidValue;
    }

    unsafe {
        // Copy input to device (async on provided stream)
        let input_binding = &engine.input_bindings[0];
        let copy_result = cuda_runtime_sys::cudaMemcpyAsync(
            input_binding.device_memory,
            input_ptr as *const c_void,
            expected_size * std::mem::size_of::<f32>(),
            cuda_runtime_sys::cudaMemcpyHostToDevice,
            stream_raw,
        );

        if copy_result != cuda_runtime_sys::cudaSuccess {
            return CudaRsResult::ErrorUnknown;
        }

        // Build bindings array (inputs first, then outputs)
        let total_bindings = engine.input_bindings.len() + engine.output_bindings.len();
        let mut bindings: Vec<*mut c_void> = Vec::with_capacity(total_bindings);

        for b in &engine.input_bindings {
            bindings.push(b.device_memory);
        }
        for b in &engine.output_bindings {
            bindings.push(b.device_memory);
        }

        // Execute inference (async on provided stream)
        let exec_result = trt_context_enqueue_v2(
            engine.context_ptr,
            bindings.as_ptr(),
            stream_raw as *mut c_void,
        );

        if exec_result == 0 {
            return CudaRsResult::ErrorUnknown;
        }

        // Copy outputs back to host (async)
        let mut output_tensors: Vec<CudaRsTrtTensor> = Vec::with_capacity(engine.output_bindings.len());

        for binding in &engine.output_bindings {
            let data_vec: Vec<f32> = vec![0.0; binding.size];
            let mut data_box = data_vec.into_boxed_slice();

            let copy_result = cuda_runtime_sys::cudaMemcpyAsync(
                data_box.as_mut_ptr() as *mut c_void,
                binding.device_memory,
                binding.size * std::mem::size_of::<f32>(),
                cuda_runtime_sys::cudaMemcpyDeviceToHost,
                stream_raw,
            );

            if copy_result != cuda_runtime_sys::cudaSuccess {
                return CudaRsResult::ErrorUnknown;
            }

            let shape_vec: Vec<i64> = binding.shape.clone();
            let shape_len = shape_vec.len() as u64;
            let data_len = binding.size as u64;

            let shape_box = shape_vec.into_boxed_slice();
            let shape_ptr = Box::into_raw(shape_box) as *mut i64;
            let data_ptr = Box::into_raw(data_box) as *mut f32;

            output_tensors.push(CudaRsTrtTensor {
                data: data_ptr,
                data_len,
                shape: shape_ptr,
                shape_len,
            });
        }

        if cuda_runtime_sys::cudaStreamSynchronize(stream_raw) != cuda_runtime_sys::cudaSuccess {
            return CudaRsResult::ErrorUnknown;
        }

        let count = output_tensors.len() as u64;
        let boxed = output_tensors.into_boxed_slice();
        let ptr = Box::into_raw(boxed) as *mut CudaRsTrtTensor;

        *out_tensors = ptr;
        *out_count = count;
    }

    CudaRsResult::Success
}

/// Get the number of output bindings for a TensorRT engine.
#[no_mangle]
pub extern "C" fn cudars_trt_get_output_count(
    handle: CudaRsTrtEngine,
    out_count: *mut c_int,
) -> CudaRsResult {
    if out_count.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let engine = {
        let engines = TRT_ENGINES.lock().unwrap();
        match engines.get(handle) {
            Some(e) => Arc::clone(e),
            None => return CudaRsResult::ErrorInvalidHandle,
        }
    };
    let engine = engine.lock().unwrap();

    unsafe { *out_count = engine.output_bindings.len() as c_int };
    CudaRsResult::Success
}

/// Get an output binding's device pointer and size in bytes.
#[no_mangle]
pub extern "C" fn cudars_trt_get_output_device_ptr(
    handle: CudaRsTrtEngine,
    index: c_int,
    out_ptr: *mut *mut c_void,
    out_bytes: *mut c_ulonglong,
) -> CudaRsResult {
    if out_ptr.is_null() || out_bytes.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let engine = {
        let engines = TRT_ENGINES.lock().unwrap();
        match engines.get(handle) {
            Some(e) => Arc::clone(e),
            None => return CudaRsResult::ErrorInvalidHandle,
        }
    };
    let engine = engine.lock().unwrap();

    let idx = index as usize;
    if idx >= engine.output_bindings.len() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let binding = &engine.output_bindings[idx];
    unsafe {
        *out_ptr = binding.device_memory;
        *out_bytes = (binding.size * std::mem::size_of::<f32>()) as c_ulonglong;
    }
    CudaRsResult::Success
}

/// Get an input binding's device pointer and size in bytes.
#[no_mangle]
pub extern "C" fn cudars_trt_get_input_device_ptr(
    handle: CudaRsTrtEngine,
    index: c_int,
    out_ptr: *mut *mut c_void,
    out_bytes: *mut c_ulonglong,
) -> CudaRsResult {
    if out_ptr.is_null() || out_bytes.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let engine = {
        let engines = TRT_ENGINES.lock().unwrap();
        match engines.get(handle) {
            Some(e) => Arc::clone(e),
            None => return CudaRsResult::ErrorInvalidHandle,
        }
    };
    let engine = engine.lock().unwrap();

    let idx = index as usize;
    if idx >= engine.input_bindings.len() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let binding = &engine.input_bindings[idx];
    unsafe {
        *out_ptr = binding.device_memory;
        *out_bytes = (binding.size * std::mem::size_of::<f32>()) as c_ulonglong;
    }

    CudaRsResult::Success
}

/// Enqueue inference using a device input pointer on a caller-provided stream (async).
///
/// This does not perform any host<->device copies, does not synchronize, and does not allocate outputs.
/// Callers can copy outputs using `cudars_trt_get_output_device_ptr` + async memcpy APIs.
///
/// If `done_event` is non-zero, it will be recorded on the same stream after the enqueue.
#[no_mangle]
pub extern "C" fn cudars_trt_enqueue_device(
    handle: CudaRsTrtEngine,
    input_device_ptr: *const f32,
    input_len: c_ulonglong,
    stream: CudaRsStream,
    done_event: CudaRsEvent,
) -> CudaRsResult {
    if input_device_ptr.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    // Look up stream raw handle (avoid holding locks across FFI calls).
    let stream_raw = {
        let streams = STREAMS.lock().unwrap();
        let s = match streams.get(stream) {
            Some(s) => s,
            None => return CudaRsResult::ErrorInvalidHandle,
        };
        s.as_raw()
    };

    let event_raw = if done_event != 0 {
        let events = EVENTS.lock().unwrap();
        let e = match events.get(done_event) {
            Some(e) => e,
            None => return CudaRsResult::ErrorInvalidHandle,
        };
        Some(e.as_raw())
    } else {
        None
    };

    let engines = TRT_ENGINES.lock().unwrap();
    let engine = match engines.get(handle) {
        Some(e) => Arc::clone(e),
        None => return CudaRsResult::ErrorInvalidHandle,
    };
    drop(engines);
    let engine = engine.lock().unwrap();

    unsafe {
        cuda_runtime_sys::cudaSetDevice(engine.device_id);
    }

    if engine.input_bindings.is_empty() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let expected_size: usize = engine.input_bindings[0].size;
    if input_len as usize != expected_size {
        return CudaRsResult::ErrorInvalidValue;
    }

    unsafe {
        // Build bindings array (inputs first, then outputs). Replace first input with provided device pointer.
        let total_bindings = engine.input_bindings.len() + engine.output_bindings.len();
        let mut bindings: Vec<*mut c_void> = Vec::with_capacity(total_bindings);

        bindings.push(input_device_ptr as *mut c_void);
        for b in engine.input_bindings.iter().skip(1) {
            bindings.push(b.device_memory);
        }
        for b in &engine.output_bindings {
            bindings.push(b.device_memory);
        }

        let exec_result = trt_context_enqueue_v2(
            engine.context_ptr,
            bindings.as_ptr(),
            stream_raw as *mut c_void,
        );

        if exec_result == 0 {
            return CudaRsResult::ErrorUnknown;
        }

        if let Some(ev) = event_raw {
            let code = cuda_runtime_sys::cudaEventRecord(ev, stream_raw);
            if code != cuda_runtime_sys::cudaSuccess {
                return CudaRsResult::ErrorUnknown;
            }
        }
    }

    CudaRsResult::Success
}

/// Get input tensor info for TensorRT engine.
#[no_mangle]
pub extern "C" fn cudars_trt_get_input_info(
    handle: CudaRsTrtEngine,
    index: c_int,
    out_shape: *mut i64,
    out_shape_len: *mut c_int,
    max_shape_len: c_int,
) -> CudaRsResult {
    if out_shape.is_null() || out_shape_len.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let engine = {
        let engines = TRT_ENGINES.lock().unwrap();
        match engines.get(handle) {
            Some(e) => Arc::clone(e),
            None => return CudaRsResult::ErrorInvalidHandle,
        }
    };
    let engine = engine.lock().unwrap();

    let idx = index as usize;
    if idx >= engine.input_bindings.len() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let binding = &engine.input_bindings[idx];
    let shape_len = binding.shape.len().min(max_shape_len as usize);

    unsafe {
        for i in 0..shape_len {
            *out_shape.add(i) = binding.shape[i];
        }
        *out_shape_len = shape_len as c_int;
    }

    CudaRsResult::Success
}

/// Get output tensor info for TensorRT engine.
#[no_mangle]
pub extern "C" fn cudars_trt_get_output_info(
    handle: CudaRsTrtEngine,
    index: c_int,
    out_shape: *mut i64,
    out_shape_len: *mut c_int,
    max_shape_len: c_int,
) -> CudaRsResult {
    if out_shape.is_null() || out_shape_len.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let engine = {
        let engines = TRT_ENGINES.lock().unwrap();
        match engines.get(handle) {
            Some(e) => Arc::clone(e),
            None => return CudaRsResult::ErrorInvalidHandle,
        }
    };
    let engine = engine.lock().unwrap();

    let idx = index as usize;
    if idx >= engine.output_bindings.len() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let binding = &engine.output_bindings[idx];
    let shape_len = binding.shape.len().min(max_shape_len as usize);

    unsafe {
        for i in 0..shape_len {
            *out_shape.add(i) = binding.shape[i];
        }
        *out_shape_len = shape_len as c_int;
    }

    CudaRsResult::Success
}

/// Destroy TensorRT engine.
#[no_mangle]
pub extern "C" fn cudars_trt_destroy(handle: CudaRsTrtEngine) -> CudaRsResult {
    let mut engines = TRT_ENGINES.lock().unwrap();
    match engines.remove(handle) {
        Some(_) => CudaRsResult::Success,
        None => CudaRsResult::ErrorInvalidHandle,
    }
}

/// Free tensors returned by cudars_trt_run.
#[no_mangle]
pub extern "C" fn cudars_trt_free_tensors(
    tensors: *mut CudaRsTrtTensor,
    count: c_ulonglong,
) -> CudaRsResult {
    if tensors.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let slice = unsafe { std::slice::from_raw_parts_mut(tensors, count as usize) };

    for t in slice.iter_mut() {
        unsafe {
            if !t.data.is_null() {
                let slice = std::slice::from_raw_parts_mut(t.data, t.data_len as usize);
                let _ = Box::from_raw(slice as *mut [f32]);
            }
            if !t.shape.is_null() {
                let slice = std::slice::from_raw_parts_mut(t.shape, t.shape_len as usize);
                let _ = Box::from_raw(slice as *mut [i64]);
            }
            t.data = ptr::null_mut();
            t.shape = ptr::null_mut();
            t.data_len = 0;
            t.shape_len = 0;
        }
    }

    unsafe {
        let _ = Box::from_raw(slice as *mut [CudaRsTrtTensor]);
    }

    CudaRsResult::Success
}
