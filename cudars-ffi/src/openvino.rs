//! OpenVINO FFI exports (Rust-side implementation).
//!
//! Provides OpenVINO model loading and inference via OpenVINO C API.
//! Supports ONNX, OpenVINO IR (.xml/.bin), and other formats.

use crate::CudaRsResult;
use libc::{c_char, c_int, c_ulonglong, c_void, size_t};
use std::ffi::{CStr, CString};
use std::ptr;
use std::sync::Mutex;

use crate::runtime::HandleManager;
use serde_json::Value;

// OpenVINO C API bindings (2.0 API)
#[link(name = "openvino")]
#[link(name = "openvino_c")]
extern "C" {
    // Core
    fn ov_core_create(core: *mut *mut c_void) -> c_int;
    fn ov_core_free(core: *mut c_void);
    fn ov_core_read_model(
        core: *mut c_void,
        model_path: *const c_char,
        weights_path: *const c_char,
        model: *mut *mut c_void,
    ) -> c_int;
    fn ov_core_compile_model(
        core: *mut c_void,
        model: *mut c_void,
        device_name: *const c_char,
        num_properties: size_t,
        compiled_model: *mut *mut c_void,
        ...
    ) -> c_int;
    fn ov_core_get_available_devices(
        core: *mut c_void,
        devices: *mut *mut c_char,
        num_devices: *mut size_t,
    ) -> c_int;

    // Model
    fn ov_model_free(model: *mut c_void);
    #[allow(dead_code)]
    fn ov_model_get_inputs(
        model: *mut c_void,
        inputs: *mut *mut c_void,
        num_inputs: *mut size_t,
    ) -> c_int;
    #[allow(dead_code)]
    fn ov_model_get_outputs(
        model: *mut c_void,
        outputs: *mut *mut c_void,
        num_outputs: *mut size_t,
    ) -> c_int;

    // Compiled Model
    fn ov_compiled_model_free(compiled_model: *mut c_void);
    fn ov_compiled_model_create_infer_request(
        compiled_model: *mut c_void,
        infer_request: *mut *mut c_void,
    ) -> c_int;
    fn ov_compiled_model_input(
        compiled_model: *mut c_void,
        input: *mut *mut c_void,
    ) -> c_int;
    fn ov_compiled_model_output(
        compiled_model: *mut c_void,
        output: *mut *mut c_void,
    ) -> c_int;
    fn ov_compiled_model_inputs_size(
        compiled_model: *mut c_void,
        size: *mut size_t,
    ) -> c_int;
    fn ov_compiled_model_outputs_size(
        compiled_model: *mut c_void,
        size: *mut size_t,
    ) -> c_int;

    // Infer Request
    fn ov_infer_request_free(infer_request: *mut c_void);
    #[allow(dead_code)]
    fn ov_infer_request_set_input_tensor(
        infer_request: *mut c_void,
        tensor: *mut c_void,
    ) -> c_int;
    fn ov_infer_request_set_input_tensor_by_index(
        infer_request: *mut c_void,
        index: size_t,
        tensor: *mut c_void,
    ) -> c_int;
    #[allow(dead_code)]
    fn ov_infer_request_get_output_tensor(
        infer_request: *mut c_void,
        tensor: *mut *mut c_void,
    ) -> c_int;
    fn ov_infer_request_get_output_tensor_by_index(
        infer_request: *mut c_void,
        index: size_t,
        tensor: *mut *mut c_void,
    ) -> c_int;
    fn ov_infer_request_infer(infer_request: *mut c_void) -> c_int;
    fn ov_infer_request_start_async(infer_request: *mut c_void) -> c_int;
    fn ov_infer_request_wait(infer_request: *mut c_void) -> c_int;

    // Tensor
    fn ov_tensor_create_from_host_ptr(
        element_type: c_int,
        shape: OvShape,
        data: *mut c_void,
        tensor: *mut *mut c_void,
    ) -> c_int;
    #[allow(dead_code)]
    fn ov_tensor_create(
        element_type: c_int,
        shape: OvShape,
        tensor: *mut *mut c_void,
    ) -> c_int;
    fn ov_tensor_free(tensor: *mut c_void);
    fn ov_tensor_data(tensor: *mut c_void, data: *mut *mut c_void) -> c_int;
    fn ov_tensor_get_shape(tensor: *mut c_void, shape: *mut OvShape) -> c_int;
    fn ov_tensor_get_size(tensor: *mut c_void, size: *mut size_t) -> c_int;
    #[allow(dead_code)]
    fn ov_tensor_get_element_type(tensor: *mut c_void, element_type: *mut c_int) -> c_int;

    // Port (for input/output info)
    #[allow(dead_code)]
    fn ov_port_get_any_name(port: *mut c_void, name: *mut *mut c_char) -> c_int;
    fn ov_const_port_get_shape(port: *mut c_void, shape: *mut OvShape) -> c_int;
    #[allow(dead_code)]
    fn ov_port_get_element_type(port: *mut c_void, element_type: *mut c_int) -> c_int;

    // Shape
    fn ov_shape_create(rank: i64, dims: *const i64, shape: *mut OvShape) -> c_int;
    fn ov_shape_free(shape: *mut OvShape) -> c_int;

    // Model reshape
    fn ov_model_reshape(
        model: *mut c_void,
        tensor_names: *const *const c_char,
        partial_shapes: *const OvPartialShape,
        size: size_t,
    ) -> c_int;
    fn ov_partial_shape_create_static(rank: i64, dims: *const i64, shape: *mut OvPartialShape) -> c_int;
    fn ov_partial_shape_free(shape: *mut OvPartialShape) -> c_int;

    // Devices string free
    fn ov_free(ptr: *mut c_void);
    fn ov_get_last_err_msg() -> *const c_char;
}

// OpenVINO element types
// ov_element_type_e values from OpenVINO 2025.4 (ov_common.h)
const OV_ELEMENT_TYPE_F32: c_int = 4;
#[allow(dead_code)]
const OV_ELEMENT_TYPE_F16: c_int = 3;
#[allow(dead_code)]
const OV_ELEMENT_TYPE_I32: c_int = 9;
#[allow(dead_code)]
const OV_ELEMENT_TYPE_I64: c_int = 10;
#[allow(dead_code)]
const OV_ELEMENT_TYPE_U8: c_int = 16;

/// OpenVINO status codes
const OV_SUCCESS: c_int = 0;

fn ov_debug_enabled() -> bool {
    std::env::var("CUDARS_OV_DEBUG").as_deref() == Ok("1")
}

fn ov_log_error(step: &str, code: c_int) {
    if ov_debug_enabled() {
        let mut msg = String::new();
        unsafe {
            let ptr = ov_get_last_err_msg();
            if !ptr.is_null() {
                if let Ok(s) = CStr::from_ptr(ptr).to_str() {
                    msg = s.to_string();
                }
            }
        }
        if msg.is_empty() {
            eprintln!("[cudars][openvino] {step} failed: code={code}");
        } else {
            eprintln!("[cudars][openvino] {step} failed: code={code}, msg={msg}");
        }
    }
}

/// Opaque handle for OpenVINO model
pub type CudaRsOvModel = u64;

#[repr(C)]
#[derive(Clone, Copy)]
struct OvShape {
    rank: i64,
    dims: *mut i64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct OvPartialShape {
    rank: i64,
    dims: OvPartialShapeDims,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct OvPartialShapeDims {
    min: *mut i64,
    max: *mut i64,
}

/// OpenVINO tensor descriptor
#[repr(C)]
pub struct CudaRsOvTensor {
    pub data: *mut f32,
    pub data_len: c_ulonglong,
    pub shape: *mut i64,
    pub shape_len: c_ulonglong,
}

/// OpenVINO tensor metadata (for model input/output info)
#[repr(C)]
pub struct CudaRsOvTensorInfo {
    pub name_ptr: *mut c_char,
    pub name_len: c_ulonglong,
    pub shape: *mut i64,
    pub shape_len: c_ulonglong,
    pub element_type: c_int,
}

/// Partial dimension for dynamic shapes (-1 = dynamic)
#[repr(C)]
pub struct CudaRsOvPartialDim {
    pub is_static: c_int,
    pub value: i64,
}

/// Partial shape (array of partial dimensions)
#[repr(C)]
pub struct CudaRsOvPartialShapeArray {
    pub dims: *const CudaRsOvPartialDim,
    pub rank: c_ulonglong,
}

/// OpenVINO device type
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaRsOvDevice {
    Cpu = 0,
    Gpu = 1,
    GpuIndex = 2, // GPU.0, GPU.1, etc.
    Npu = 3,
    Auto = 4,
}

/// OpenVINO inference configuration
#[repr(C)]
pub struct CudaRsOvConfig {
    pub device: CudaRsOvDevice,
    pub device_index: c_int,
    pub num_streams: c_int,      // 0 = auto
    pub enable_profiling: c_int,
    pub properties_json_ptr: *const c_char,
    pub properties_json_len: size_t,
}

/// OpenVINO inference configuration (v2, extends CudaRsOvConfig).
#[repr(C)]
pub struct CudaRsOvConfigV2 {
    pub struct_size: u32,
    pub device: CudaRsOvDevice,
    pub device_index: c_int,
    pub device_name_ptr: *const c_char,
    pub device_name_len: size_t,
    pub num_streams: c_int, // 0 = auto
    pub enable_profiling: c_int,
    pub properties_json_ptr: *const c_char,
    pub properties_json_len: size_t,
}

impl Default for CudaRsOvConfigV2 {
    fn default() -> Self {
        Self {
            struct_size: std::mem::size_of::<CudaRsOvConfigV2>() as u32,
            device: CudaRsOvDevice::Cpu,
            device_index: 0,
            device_name_ptr: ptr::null(),
            device_name_len: 0,
            num_streams: 0,
            enable_profiling: 0,
            properties_json_ptr: ptr::null(),
            properties_json_len: 0,
        }
    }
}

impl Default for CudaRsOvConfig {
    fn default() -> Self {
        Self {
            device: CudaRsOvDevice::Cpu,
            device_index: 0,
            num_streams: 0,
            enable_profiling: 0,
            properties_json_ptr: ptr::null(),
            properties_json_len: 0,
        }
    }
}

/// Internal OpenVINO model wrapper
struct OvModel {
    core_ptr: *mut c_void,
    model_ptr: *mut c_void,
    compiled_model_ptr: *mut c_void,
    infer_request_ptr: *mut c_void,
    async_requests: Vec<OvInferRequestSlot>,
    input_shapes: Vec<Vec<i64>>,
    output_shapes: Vec<Vec<i64>>,
    model_path: String,
    device_name: String,
    properties: Vec<(String, String)>,
    async_request_count: usize,
}

struct OvInferRequestSlot {
    request_ptr: *mut c_void,
    input_tensor_ptr: *mut c_void,
    input_owned: Option<Box<[f32]>>,
    busy: bool,
}

unsafe impl Send for OvModel {}
unsafe impl Sync for OvModel {}

impl Drop for OvModel {
    fn drop(&mut self) {
        unsafe {
            if !self.infer_request_ptr.is_null() {
                ov_infer_request_free(self.infer_request_ptr);
            }
            for slot in self.async_requests.iter_mut() {
                if !slot.input_tensor_ptr.is_null() {
                    ov_tensor_free(slot.input_tensor_ptr);
                    slot.input_tensor_ptr = ptr::null_mut();
                }
                slot.input_owned = None;
                slot.busy = false;
                if !slot.request_ptr.is_null() {
                    ov_infer_request_free(slot.request_ptr);
                    slot.request_ptr = ptr::null_mut();
                }
            }
            if !self.compiled_model_ptr.is_null() {
                ov_compiled_model_free(self.compiled_model_ptr);
            }
            if !self.model_ptr.is_null() {
                ov_model_free(self.model_ptr);
            }
            if !self.core_ptr.is_null() {
                ov_core_free(self.core_ptr);
            }
        }
    }
}

lazy_static::lazy_static! {
    static ref OV_MODELS: Mutex<HandleManager<OvModel>> = Mutex::new(HandleManager::new());
}

const MAX_OV_PROPERTIES: usize = 16;

/// Get device name string from config.
fn get_device_name(config: &CudaRsOvConfig) -> String {
    match config.device {
        CudaRsOvDevice::Cpu => "CPU".to_string(),
        CudaRsOvDevice::Gpu => "GPU".to_string(),
        CudaRsOvDevice::GpuIndex => format!("GPU.{}", config.device_index),
        CudaRsOvDevice::Npu => "NPU".to_string(),
        CudaRsOvDevice::Auto => "AUTO".to_string(),
    }
}

fn get_device_name_v2(config: &CudaRsOvConfigV2) -> Result<String, CudaRsResult> {
    if !config.device_name_ptr.is_null() && config.device_name_len > 0 {
        let bytes = unsafe {
            std::slice::from_raw_parts(config.device_name_ptr as *const u8, config.device_name_len as usize)
        };
        let name = match std::str::from_utf8(bytes) {
            Ok(v) => v.trim(),
            Err(_) => return Err(CudaRsResult::ErrorInvalidValue),
        };
        if name.is_empty() {
            return Err(CudaRsResult::ErrorInvalidValue);
        }
        return Ok(name.to_string());
    }

    let legacy = CudaRsOvConfig {
        device: config.device,
        device_index: config.device_index,
        num_streams: config.num_streams,
        enable_profiling: config.enable_profiling,
        properties_json_ptr: config.properties_json_ptr,
        properties_json_len: config.properties_json_len,
    };
    Ok(get_device_name(&legacy))
}

fn json_value_to_string(value: &Value) -> String {
    match value {
        Value::String(v) => v.clone(),
        Value::Number(v) => v.to_string(),
        Value::Bool(v) => v.to_string(),
        Value::Null => "null".to_string(),
        Value::Array(_) | Value::Object(_) => value.to_string(),
    }
}

fn debug_print_shape(label: &str, shape: &[i64]) {
    if ov_debug_enabled() {
        eprintln!("[cudars][openvino] {label} shape={shape:?}");
    }
}

fn parse_properties_json(properties_json: &str) -> Result<Vec<(String, String)>, CudaRsResult> {
    let value: Value = match serde_json::from_str(properties_json) {
        Ok(v) => v,
        Err(_) => return Err(CudaRsResult::ErrorInvalidValue),
    };

    let obj = match value.as_object() {
        Some(o) => o,
        None => return Err(CudaRsResult::ErrorInvalidValue),
    };

    let mut props = Vec::with_capacity(obj.len());
    for (key, value) in obj {
        props.push((key.clone(), json_value_to_string(value)));
    }

    if props.len() > MAX_OV_PROPERTIES {
        return Err(CudaRsResult::ErrorInvalidValue);
    }

    Ok(props)
}

fn property_key_eq(a: &str, b: &str) -> bool {
    a.eq_ignore_ascii_case(b)
}

fn append_property_if_missing(props: &mut Vec<(String, String)>, key: &str, value: String) {
    if props.iter().any(|(k, _)| property_key_eq(k, key)) {
        return;
    }
    props.push((key.to_string(), value));
}

fn parse_num_property(props: &[(String, String)], key: &str) -> Option<usize> {
    props
        .iter()
        .find(|(k, _)| property_key_eq(k, key))
        .and_then(|(_, v)| v.trim().parse::<i64>().ok())
        .and_then(|v| if v > 0 { Some(v as usize) } else { None })
}

fn read_properties_json(ptr: *const c_char, len: size_t) -> Result<Vec<(String, String)>, CudaRsResult> {
    if ptr.is_null() || len == 0 {
        return Ok(Vec::new());
    }
    let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, len as usize) };
    let json = match std::str::from_utf8(bytes) {
        Ok(v) => v,
        Err(_) => return Err(CudaRsResult::ErrorInvalidValue),
    };
    parse_properties_json(json)
}

fn prepare_properties(
    mut properties: Vec<(String, String)>,
    num_streams: c_int,
    enable_profiling: c_int,
) -> Result<(Vec<(String, String)>, usize), CudaRsResult> {
    if num_streams > 0 {
        append_property_if_missing(&mut properties, "NUM_STREAMS", num_streams.to_string());
    }

    if enable_profiling != 0 {
        append_property_if_missing(&mut properties, "ENABLE_PROFILING", "YES".to_string());
    }

    if properties.len() > MAX_OV_PROPERTIES {
        return Err(CudaRsResult::ErrorInvalidValue);
    }

    let async_request_count =
        parse_num_property(&properties, "NUM_REQUESTS").unwrap_or_else(|| {
            parse_num_property(&properties, "NUM_INFER_REQUESTS").unwrap_or(1)
        });

    let properties = properties
        .into_iter()
        .filter(|(k, _)| {
            !property_key_eq(k, "NUM_REQUESTS") && !property_key_eq(k, "NUM_INFER_REQUESTS")
        })
        .collect::<Vec<_>>();

    Ok((properties, async_request_count))
}

unsafe fn compile_model_with_properties(
    core: *mut c_void,
    model: *mut c_void,
    device_name: *const c_char,
    properties: &[(CString, CString)],
    compiled_model: *mut *mut c_void,
) -> c_int {
    match properties.len() {
        0 => ov_core_compile_model(core, model, device_name, 0, compiled_model),
        1 => ov_core_compile_model(
            core,
            model,
            device_name,
            1,
            compiled_model,
            properties[0].0.as_ptr(),
            properties[0].1.as_ptr(),
        ),
        2 => ov_core_compile_model(
            core,
            model,
            device_name,
            2,
            compiled_model,
            properties[0].0.as_ptr(),
            properties[0].1.as_ptr(),
            properties[1].0.as_ptr(),
            properties[1].1.as_ptr(),
        ),
        3 => ov_core_compile_model(
            core,
            model,
            device_name,
            3,
            compiled_model,
            properties[0].0.as_ptr(),
            properties[0].1.as_ptr(),
            properties[1].0.as_ptr(),
            properties[1].1.as_ptr(),
            properties[2].0.as_ptr(),
            properties[2].1.as_ptr(),
        ),
        4 => ov_core_compile_model(
            core,
            model,
            device_name,
            4,
            compiled_model,
            properties[0].0.as_ptr(),
            properties[0].1.as_ptr(),
            properties[1].0.as_ptr(),
            properties[1].1.as_ptr(),
            properties[2].0.as_ptr(),
            properties[2].1.as_ptr(),
            properties[3].0.as_ptr(),
            properties[3].1.as_ptr(),
        ),
        5 => ov_core_compile_model(
            core,
            model,
            device_name,
            5,
            compiled_model,
            properties[0].0.as_ptr(),
            properties[0].1.as_ptr(),
            properties[1].0.as_ptr(),
            properties[1].1.as_ptr(),
            properties[2].0.as_ptr(),
            properties[2].1.as_ptr(),
            properties[3].0.as_ptr(),
            properties[3].1.as_ptr(),
            properties[4].0.as_ptr(),
            properties[4].1.as_ptr(),
        ),
        6 => ov_core_compile_model(
            core,
            model,
            device_name,
            6,
            compiled_model,
            properties[0].0.as_ptr(),
            properties[0].1.as_ptr(),
            properties[1].0.as_ptr(),
            properties[1].1.as_ptr(),
            properties[2].0.as_ptr(),
            properties[2].1.as_ptr(),
            properties[3].0.as_ptr(),
            properties[3].1.as_ptr(),
            properties[4].0.as_ptr(),
            properties[4].1.as_ptr(),
            properties[5].0.as_ptr(),
            properties[5].1.as_ptr(),
        ),
        7 => ov_core_compile_model(
            core,
            model,
            device_name,
            7,
            compiled_model,
            properties[0].0.as_ptr(),
            properties[0].1.as_ptr(),
            properties[1].0.as_ptr(),
            properties[1].1.as_ptr(),
            properties[2].0.as_ptr(),
            properties[2].1.as_ptr(),
            properties[3].0.as_ptr(),
            properties[3].1.as_ptr(),
            properties[4].0.as_ptr(),
            properties[4].1.as_ptr(),
            properties[5].0.as_ptr(),
            properties[5].1.as_ptr(),
            properties[6].0.as_ptr(),
            properties[6].1.as_ptr(),
        ),
        8 => ov_core_compile_model(
            core,
            model,
            device_name,
            8,
            compiled_model,
            properties[0].0.as_ptr(),
            properties[0].1.as_ptr(),
            properties[1].0.as_ptr(),
            properties[1].1.as_ptr(),
            properties[2].0.as_ptr(),
            properties[2].1.as_ptr(),
            properties[3].0.as_ptr(),
            properties[3].1.as_ptr(),
            properties[4].0.as_ptr(),
            properties[4].1.as_ptr(),
            properties[5].0.as_ptr(),
            properties[5].1.as_ptr(),
            properties[6].0.as_ptr(),
            properties[6].1.as_ptr(),
            properties[7].0.as_ptr(),
            properties[7].1.as_ptr(),
        ),
        9 => ov_core_compile_model(
            core,
            model,
            device_name,
            9,
            compiled_model,
            properties[0].0.as_ptr(),
            properties[0].1.as_ptr(),
            properties[1].0.as_ptr(),
            properties[1].1.as_ptr(),
            properties[2].0.as_ptr(),
            properties[2].1.as_ptr(),
            properties[3].0.as_ptr(),
            properties[3].1.as_ptr(),
            properties[4].0.as_ptr(),
            properties[4].1.as_ptr(),
            properties[5].0.as_ptr(),
            properties[5].1.as_ptr(),
            properties[6].0.as_ptr(),
            properties[6].1.as_ptr(),
            properties[7].0.as_ptr(),
            properties[7].1.as_ptr(),
            properties[8].0.as_ptr(),
            properties[8].1.as_ptr(),
        ),
        10 => ov_core_compile_model(
            core,
            model,
            device_name,
            10,
            compiled_model,
            properties[0].0.as_ptr(),
            properties[0].1.as_ptr(),
            properties[1].0.as_ptr(),
            properties[1].1.as_ptr(),
            properties[2].0.as_ptr(),
            properties[2].1.as_ptr(),
            properties[3].0.as_ptr(),
            properties[3].1.as_ptr(),
            properties[4].0.as_ptr(),
            properties[4].1.as_ptr(),
            properties[5].0.as_ptr(),
            properties[5].1.as_ptr(),
            properties[6].0.as_ptr(),
            properties[6].1.as_ptr(),
            properties[7].0.as_ptr(),
            properties[7].1.as_ptr(),
            properties[8].0.as_ptr(),
            properties[8].1.as_ptr(),
            properties[9].0.as_ptr(),
            properties[9].1.as_ptr(),
        ),
        11 => ov_core_compile_model(
            core,
            model,
            device_name,
            11,
            compiled_model,
            properties[0].0.as_ptr(),
            properties[0].1.as_ptr(),
            properties[1].0.as_ptr(),
            properties[1].1.as_ptr(),
            properties[2].0.as_ptr(),
            properties[2].1.as_ptr(),
            properties[3].0.as_ptr(),
            properties[3].1.as_ptr(),
            properties[4].0.as_ptr(),
            properties[4].1.as_ptr(),
            properties[5].0.as_ptr(),
            properties[5].1.as_ptr(),
            properties[6].0.as_ptr(),
            properties[6].1.as_ptr(),
            properties[7].0.as_ptr(),
            properties[7].1.as_ptr(),
            properties[8].0.as_ptr(),
            properties[8].1.as_ptr(),
            properties[9].0.as_ptr(),
            properties[9].1.as_ptr(),
            properties[10].0.as_ptr(),
            properties[10].1.as_ptr(),
        ),
        12 => ov_core_compile_model(
            core,
            model,
            device_name,
            12,
            compiled_model,
            properties[0].0.as_ptr(),
            properties[0].1.as_ptr(),
            properties[1].0.as_ptr(),
            properties[1].1.as_ptr(),
            properties[2].0.as_ptr(),
            properties[2].1.as_ptr(),
            properties[3].0.as_ptr(),
            properties[3].1.as_ptr(),
            properties[4].0.as_ptr(),
            properties[4].1.as_ptr(),
            properties[5].0.as_ptr(),
            properties[5].1.as_ptr(),
            properties[6].0.as_ptr(),
            properties[6].1.as_ptr(),
            properties[7].0.as_ptr(),
            properties[7].1.as_ptr(),
            properties[8].0.as_ptr(),
            properties[8].1.as_ptr(),
            properties[9].0.as_ptr(),
            properties[9].1.as_ptr(),
            properties[10].0.as_ptr(),
            properties[10].1.as_ptr(),
            properties[11].0.as_ptr(),
            properties[11].1.as_ptr(),
        ),
        13 => ov_core_compile_model(
            core,
            model,
            device_name,
            13,
            compiled_model,
            properties[0].0.as_ptr(),
            properties[0].1.as_ptr(),
            properties[1].0.as_ptr(),
            properties[1].1.as_ptr(),
            properties[2].0.as_ptr(),
            properties[2].1.as_ptr(),
            properties[3].0.as_ptr(),
            properties[3].1.as_ptr(),
            properties[4].0.as_ptr(),
            properties[4].1.as_ptr(),
            properties[5].0.as_ptr(),
            properties[5].1.as_ptr(),
            properties[6].0.as_ptr(),
            properties[6].1.as_ptr(),
            properties[7].0.as_ptr(),
            properties[7].1.as_ptr(),
            properties[8].0.as_ptr(),
            properties[8].1.as_ptr(),
            properties[9].0.as_ptr(),
            properties[9].1.as_ptr(),
            properties[10].0.as_ptr(),
            properties[10].1.as_ptr(),
            properties[11].0.as_ptr(),
            properties[11].1.as_ptr(),
            properties[12].0.as_ptr(),
            properties[12].1.as_ptr(),
        ),
        14 => ov_core_compile_model(
            core,
            model,
            device_name,
            14,
            compiled_model,
            properties[0].0.as_ptr(),
            properties[0].1.as_ptr(),
            properties[1].0.as_ptr(),
            properties[1].1.as_ptr(),
            properties[2].0.as_ptr(),
            properties[2].1.as_ptr(),
            properties[3].0.as_ptr(),
            properties[3].1.as_ptr(),
            properties[4].0.as_ptr(),
            properties[4].1.as_ptr(),
            properties[5].0.as_ptr(),
            properties[5].1.as_ptr(),
            properties[6].0.as_ptr(),
            properties[6].1.as_ptr(),
            properties[7].0.as_ptr(),
            properties[7].1.as_ptr(),
            properties[8].0.as_ptr(),
            properties[8].1.as_ptr(),
            properties[9].0.as_ptr(),
            properties[9].1.as_ptr(),
            properties[10].0.as_ptr(),
            properties[10].1.as_ptr(),
            properties[11].0.as_ptr(),
            properties[11].1.as_ptr(),
            properties[12].0.as_ptr(),
            properties[12].1.as_ptr(),
            properties[13].0.as_ptr(),
            properties[13].1.as_ptr(),
        ),
        15 => ov_core_compile_model(
            core,
            model,
            device_name,
            15,
            compiled_model,
            properties[0].0.as_ptr(),
            properties[0].1.as_ptr(),
            properties[1].0.as_ptr(),
            properties[1].1.as_ptr(),
            properties[2].0.as_ptr(),
            properties[2].1.as_ptr(),
            properties[3].0.as_ptr(),
            properties[3].1.as_ptr(),
            properties[4].0.as_ptr(),
            properties[4].1.as_ptr(),
            properties[5].0.as_ptr(),
            properties[5].1.as_ptr(),
            properties[6].0.as_ptr(),
            properties[6].1.as_ptr(),
            properties[7].0.as_ptr(),
            properties[7].1.as_ptr(),
            properties[8].0.as_ptr(),
            properties[8].1.as_ptr(),
            properties[9].0.as_ptr(),
            properties[9].1.as_ptr(),
            properties[10].0.as_ptr(),
            properties[10].1.as_ptr(),
            properties[11].0.as_ptr(),
            properties[11].1.as_ptr(),
            properties[12].0.as_ptr(),
            properties[12].1.as_ptr(),
            properties[13].0.as_ptr(),
            properties[13].1.as_ptr(),
            properties[14].0.as_ptr(),
            properties[14].1.as_ptr(),
        ),
        16 => ov_core_compile_model(
            core,
            model,
            device_name,
            16,
            compiled_model,
            properties[0].0.as_ptr(),
            properties[0].1.as_ptr(),
            properties[1].0.as_ptr(),
            properties[1].1.as_ptr(),
            properties[2].0.as_ptr(),
            properties[2].1.as_ptr(),
            properties[3].0.as_ptr(),
            properties[3].1.as_ptr(),
            properties[4].0.as_ptr(),
            properties[4].1.as_ptr(),
            properties[5].0.as_ptr(),
            properties[5].1.as_ptr(),
            properties[6].0.as_ptr(),
            properties[6].1.as_ptr(),
            properties[7].0.as_ptr(),
            properties[7].1.as_ptr(),
            properties[8].0.as_ptr(),
            properties[8].1.as_ptr(),
            properties[9].0.as_ptr(),
            properties[9].1.as_ptr(),
            properties[10].0.as_ptr(),
            properties[10].1.as_ptr(),
            properties[11].0.as_ptr(),
            properties[11].1.as_ptr(),
            properties[12].0.as_ptr(),
            properties[12].1.as_ptr(),
            properties[13].0.as_ptr(),
            properties[13].1.as_ptr(),
            properties[14].0.as_ptr(),
            properties[14].1.as_ptr(),
            properties[15].0.as_ptr(),
            properties[15].1.as_ptr(),
        ),
        _ => OV_SUCCESS + 1,
    }
}

unsafe fn create_input_tensor_from_slice(
    shape: &[i64],
    input_ptr: *const f32,
) -> Result<*mut c_void, CudaRsResult> {
    let mut input_tensor: *mut c_void = ptr::null_mut();
    let mut input_shape = OvShape {
        rank: 0,
        dims: ptr::null_mut(),
    };
    let result = ov_shape_create(shape.len() as i64, shape.as_ptr(), &mut input_shape);
    if result != OV_SUCCESS {
        ov_log_error("ov_shape_create(input)", result);
        return Err(CudaRsResult::ErrorInvalidValue);
    }
    let result = ov_tensor_create_from_host_ptr(
        OV_ELEMENT_TYPE_F32,
        input_shape,
        input_ptr as *mut c_void,
        &mut input_tensor,
    );
    let _ = ov_shape_free(&mut input_shape);

    if result != OV_SUCCESS || input_tensor.is_null() {
        ov_log_error("ov_tensor_create_from_host_ptr", result);
        return Err(CudaRsResult::ErrorOutOfMemory);
    }

    Ok(input_tensor)
}

unsafe fn collect_output_tensors(
    compiled_model: *mut c_void,
    infer_request: *mut c_void,
    out_tensors: *mut *mut CudaRsOvTensor,
    out_count: *mut c_ulonglong,
) -> CudaRsResult {
    if out_tensors.is_null() || out_count.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let mut num_outputs: size_t = 0;
    if ov_compiled_model_outputs_size(compiled_model, &mut num_outputs) != OV_SUCCESS {
        ov_log_error("ov_compiled_model_outputs_size", -1);
        return CudaRsResult::ErrorUnknown;
    }

    let mut tensors: Vec<CudaRsOvTensor> = Vec::with_capacity(num_outputs);

    for i in 0..num_outputs {
        let mut output_tensor: *mut c_void = ptr::null_mut();
        let result =
            ov_infer_request_get_output_tensor_by_index(infer_request, i, &mut output_tensor);

        if result != OV_SUCCESS || output_tensor.is_null() {
            if result != OV_SUCCESS {
                ov_log_error("ov_infer_request_get_output_tensor_by_index", result);
            }
            continue;
        }

        let mut data_ptr: *mut c_void = ptr::null_mut();
        if ov_tensor_data(output_tensor, &mut data_ptr) != OV_SUCCESS {
            ov_log_error("ov_tensor_data", -1);
            continue;
        }

        let mut tensor_size: size_t = 0;
        if ov_tensor_get_size(output_tensor, &mut tensor_size) != OV_SUCCESS {
            ov_log_error("ov_tensor_get_size", -1);
            continue;
        }

        let mut out_shape = OvShape {
            rank: 0,
            dims: ptr::null_mut(),
        };
        if ov_tensor_get_shape(output_tensor, &mut out_shape) != OV_SUCCESS {
            ov_log_error("ov_tensor_get_shape", -1);
            continue;
        }

        let shape_len = if out_shape.rank > 0 {
            out_shape.rank as usize
        } else {
            0
        };
        let shape_vec: Vec<i64> = if !out_shape.dims.is_null() && shape_len > 0 {
            std::slice::from_raw_parts(out_shape.dims, shape_len).to_vec()
        } else {
            Vec::new()
        };

        let data_slice = std::slice::from_raw_parts(data_ptr as *const f32, tensor_size);
        let data_vec: Vec<f32> = data_slice.to_vec();

        let _ = ov_shape_free(&mut out_shape);

        let data_len = tensor_size as u64;
        let shape_len = shape_vec.len() as u64;

        tensors.push(CudaRsOvTensor {
            data: Box::into_raw(data_vec.into_boxed_slice()) as *mut f32,
            data_len,
            shape: Box::into_raw(shape_vec.into_boxed_slice()) as *mut i64,
            shape_len,
        });
    }

    let count = tensors.len() as u64;
    let boxed = tensors.into_boxed_slice();
    *out_tensors = Box::into_raw(boxed) as *mut CudaRsOvTensor;
    *out_count = count;

    CudaRsResult::Success
}

/// Get available OpenVINO devices.
#[no_mangle]
pub extern "C" fn cudars_ov_get_devices(
    out_devices: *mut *mut c_char,
    out_count: *mut c_int,
    max_devices: c_int,
) -> CudaRsResult {
    if out_devices.is_null() || out_count.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    unsafe {
        let mut core: *mut c_void = ptr::null_mut();
        if ov_core_create(&mut core) != OV_SUCCESS {
            return CudaRsResult::ErrorNotInitialized;
        }

        let mut devices: *mut c_char = ptr::null_mut();
        let mut num_devices: size_t = 0;

        let result = ov_core_get_available_devices(core, &mut devices, &mut num_devices);

        if result != OV_SUCCESS {
            ov_core_free(core);
            return CudaRsResult::ErrorUnknown;
        }

        // Parse devices string (comma-separated)
        let devices_str = if devices.is_null() {
            String::new()
        } else {
            CStr::from_ptr(devices).to_string_lossy().into_owned()
        };

        let device_list: Vec<&str> = devices_str.split(',').filter(|s| !s.is_empty()).collect();
        let count = device_list.len().min(max_devices as usize);

        for (i, dev) in device_list.iter().take(count).enumerate() {
            let cstr = CString::new(*dev).unwrap_or_default();
            let ptr = libc::malloc(cstr.as_bytes_with_nul().len()) as *mut c_char;
            if !ptr.is_null() {
                libc::strcpy(ptr, cstr.as_ptr());
                *out_devices.add(i) = ptr;
            }
        }

        *out_count = count as c_int;

        if !devices.is_null() {
            ov_free(devices as *mut c_void);
        }
        ov_core_free(core);
    }

    CudaRsResult::Success
}

/// Load OpenVINO model from file.
#[no_mangle]
pub extern "C" fn cudars_ov_load(
    model_path: *const c_char,
    config: *const CudaRsOvConfig,
    out_handle: *mut CudaRsOvModel,
) -> CudaRsResult {
    if model_path.is_null() || out_handle.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let path = unsafe {
        match CStr::from_ptr(model_path).to_str() {
            Ok(p) => p,
            Err(_) => return CudaRsResult::ErrorInvalidValue,
        }
    };

    let ov_config = if config.is_null() {
        CudaRsOvConfig::default()
    } else {
        unsafe { ptr::read(config) }
    };

    let properties = match read_properties_json(ov_config.properties_json_ptr, ov_config.properties_json_len) {
        Ok(v) => v,
        Err(err) => return err,
    };

    let (properties_prepared, async_request_count) =
        match prepare_properties(properties, ov_config.num_streams, ov_config.enable_profiling) {
            Ok(v) => v,
            Err(err) => return err,
        };

    let mut property_cstrings: Vec<(CString, CString)> = Vec::with_capacity(properties_prepared.len());
    for (key, value) in &properties_prepared {
        let key_cstr = match CString::new(key.clone()) {
            Ok(v) => v,
            Err(_) => return CudaRsResult::ErrorInvalidValue,
        };
        let value_cstr = match CString::new(value.clone()) {
            Ok(v) => v,
            Err(_) => return CudaRsResult::ErrorInvalidValue,
        };
        property_cstrings.push((key_cstr, value_cstr));
    }

    let device_name = get_device_name(&ov_config);
    let device_cstr = match CString::new(device_name.clone()) {
        Ok(s) => s,
        Err(_) => return CudaRsResult::ErrorInvalidValue,
    };

    let path_cstr = match CString::new(path) {
        Ok(s) => s,
        Err(_) => return CudaRsResult::ErrorInvalidValue,
    };

    unsafe {
        // Create core
        let mut core: *mut c_void = ptr::null_mut();
        if ov_core_create(&mut core) != OV_SUCCESS {
            return CudaRsResult::ErrorNotInitialized;
        }

        // Read model
        let mut model: *mut c_void = ptr::null_mut();
        let result = ov_core_read_model(core, path_cstr.as_ptr(), ptr::null(), &mut model);
        if result != OV_SUCCESS {
            ov_log_error("ov_core_read_model", result);
            ov_core_free(core);
            return CudaRsResult::ErrorInvalidValue;
        }

        // Compile model for device
        let mut compiled_model: *mut c_void = ptr::null_mut();
        let result = compile_model_with_properties(
            core,
            model,
            device_cstr.as_ptr(),
            &property_cstrings,
            &mut compiled_model,
        );

        if result != OV_SUCCESS {
            ov_log_error("ov_core_compile_model", result);
            ov_model_free(model);
            ov_core_free(core);
            return CudaRsResult::ErrorUnknown;
        }

        // Create infer request (sync)
        let mut infer_request: *mut c_void = ptr::null_mut();
        let result = ov_compiled_model_create_infer_request(compiled_model, &mut infer_request);
        if result != OV_SUCCESS {
            ov_log_error("ov_compiled_model_create_infer_request", result);
            ov_compiled_model_free(compiled_model);
            ov_core_free(core);
            return CudaRsResult::ErrorUnknown;
        }

        // Create async request pool
        let mut async_requests: Vec<OvInferRequestSlot> =
            Vec::with_capacity(async_request_count.max(1));
        for _ in 0..async_request_count.max(1) {
            let mut request_ptr: *mut c_void = ptr::null_mut();
            let result = ov_compiled_model_create_infer_request(compiled_model, &mut request_ptr);
            if result != OV_SUCCESS || request_ptr.is_null() {
                if result != OV_SUCCESS {
                    ov_log_error("ov_compiled_model_create_infer_request_async", result);
                }
                for slot in async_requests {
                    if !slot.request_ptr.is_null() {
                        ov_infer_request_free(slot.request_ptr);
                    }
                }
                ov_infer_request_free(infer_request);
                ov_compiled_model_free(compiled_model);
                ov_core_free(core);
                return CudaRsResult::ErrorUnknown;
            }
            async_requests.push(OvInferRequestSlot {
                request_ptr,
                input_tensor_ptr: ptr::null_mut(),
                input_owned: None,
                busy: false,
            });
        }

        // Query input/output shapes for layout inference and output helpers.
        let input_shapes = get_compiled_model_input_shapes(compiled_model);
        let output_shapes = get_compiled_model_output_shapes(compiled_model);
        if ov_debug_enabled() {
            if let Some(shape) = input_shapes.get(0) {
                debug_print_shape("compiled input", shape);
            } else {
                eprintln!("[cudars][openvino] compiled input shape not available");
            }
            if let Some(shape) = output_shapes.get(0) {
                debug_print_shape("compiled output[0]", shape);
            }
        }

        let ov_model = OvModel {
            core_ptr: core,
            model_ptr: model, // Keep model for reshape
            compiled_model_ptr: compiled_model,
            infer_request_ptr: infer_request,
            async_requests,
            input_shapes,
            output_shapes,
            model_path: path.to_string(),
            device_name,
            properties: properties_prepared,
            async_request_count,
        };

        let mut models = OV_MODELS.lock().unwrap();
        let id = models.insert(ov_model);
        *out_handle = id;
    }

    CudaRsResult::Success
}

/// Common model loading logic
unsafe fn load_and_compile_model(
    path: &str,
    device_name: String,
    properties: Vec<(String, String)>,
    async_request_count: usize,
) -> Result<OvModel, CudaRsResult> {
    let path_cstr = match CString::new(path) {
        Ok(s) => s,
        Err(_) => return Err(CudaRsResult::ErrorInvalidValue),
    };
    
    let device_cstr = match CString::new(device_name.clone()) {
        Ok(s) => s,
        Err(_) => return Err(CudaRsResult::ErrorInvalidValue),
    };

    let mut property_cstrings: Vec<(CString, CString)> = Vec::with_capacity(properties.len());
    for (key, value) in &properties {
        let key_cstr = match CString::new(key.clone()) {
            Ok(v) => v,
            Err(_) => return Err(CudaRsResult::ErrorInvalidValue),
        };
        let value_cstr = match CString::new(value.clone()) {
            Ok(v) => v,
            Err(_) => return Err(CudaRsResult::ErrorInvalidValue),
        };
        property_cstrings.push((key_cstr, value_cstr));
    }

    // Create core
    let mut core: *mut c_void = ptr::null_mut();
    if ov_core_create(&mut core) != OV_SUCCESS {
        return Err(CudaRsResult::ErrorNotInitialized);
    }

    // Read model
    let mut model: *mut c_void = ptr::null_mut();
    let result = ov_core_read_model(core, path_cstr.as_ptr(), ptr::null(), &mut model);
    if result != OV_SUCCESS {
        ov_log_error("ov_core_read_model", result);
        ov_core_free(core);
        return Err(CudaRsResult::ErrorInvalidValue);
    }

    // Compile model for device
    let mut compiled_model: *mut c_void = ptr::null_mut();
    let result = compile_model_with_properties(
        core,
        model,
        device_cstr.as_ptr(),
        &property_cstrings,
        &mut compiled_model,
    );

    if result != OV_SUCCESS {
        ov_log_error("ov_core_compile_model", result);
        ov_model_free(model);
        ov_core_free(core);
        return Err(CudaRsResult::ErrorUnknown);
    }

    // Create infer request (sync)
    let mut infer_request: *mut c_void = ptr::null_mut();
    let result = ov_compiled_model_create_infer_request(compiled_model, &mut infer_request);
    if result != OV_SUCCESS {
        ov_log_error("ov_compiled_model_create_infer_request", result);
        ov_compiled_model_free(compiled_model);
        ov_model_free(model);
        ov_core_free(core);
        return Err(CudaRsResult::ErrorUnknown);
    }

    // Create async request pool
    let mut async_requests: Vec<OvInferRequestSlot> =
        Vec::with_capacity(async_request_count.max(1));
    for _ in 0..async_request_count.max(1) {
        let mut request_ptr: *mut c_void = ptr::null_mut();
        let result = ov_compiled_model_create_infer_request(compiled_model, &mut request_ptr);
        if result != OV_SUCCESS || request_ptr.is_null() {
            if result != OV_SUCCESS {
                ov_log_error("ov_compiled_model_create_infer_request_async", result);
            }
            for slot in async_requests {
                if !slot.request_ptr.is_null() {
                    ov_infer_request_free(slot.request_ptr);
                }
            }
            ov_infer_request_free(infer_request);
            ov_compiled_model_free(compiled_model);
            ov_model_free(model);
            ov_core_free(core);
            return Err(CudaRsResult::ErrorUnknown);
        }
        async_requests.push(OvInferRequestSlot {
            request_ptr,
            input_tensor_ptr: ptr::null_mut(),
            input_owned: None,
            busy: false,
        });
    }

    // Query input/output shapes for layout inference and output helpers.
    let input_shapes = get_compiled_model_input_shapes(compiled_model);
    let output_shapes = get_compiled_model_output_shapes(compiled_model);
    if ov_debug_enabled() {
        if let Some(shape) = input_shapes.get(0) {
            debug_print_shape("compiled input", shape);
        } else {
            eprintln!("[cudars][openvino] compiled input shape not available");
        }
        if let Some(shape) = output_shapes.get(0) {
            debug_print_shape("compiled output[0]", shape);
        }
    }

    Ok(OvModel {
        core_ptr: core,
        model_ptr: model,
        compiled_model_ptr: compiled_model,
        infer_request_ptr: infer_request,
        async_requests,
        input_shapes,
        output_shapes,
        model_path: path.to_string(),
        device_name,
        properties,
        async_request_count,
    })
}

/// Get input shapes from compiled model.
unsafe fn get_compiled_model_input_shapes(compiled_model: *mut c_void) -> Vec<Vec<i64>> {
    let mut shapes = Vec::new();
    let mut num_inputs: size_t = 0;

    if ov_compiled_model_inputs_size(compiled_model, &mut num_inputs) != OV_SUCCESS {
        return shapes;
    }

    for _i in 0..num_inputs {
        // For simplicity, get first input shape
        let mut port: *mut c_void = ptr::null_mut();
        if ov_compiled_model_input(compiled_model, &mut port) == OV_SUCCESS && !port.is_null() {
            let mut shape = OvShape {
                rank: 0,
                dims: ptr::null_mut(),
            };

            if ov_const_port_get_shape(port, &mut shape) == OV_SUCCESS {
                let len = if shape.rank > 0 { shape.rank as usize } else { 0 };
                let shape_vec = if !shape.dims.is_null() && len > 0 {
                    std::slice::from_raw_parts(shape.dims, len).to_vec()
                } else {
                    Vec::new()
                };
                shapes.push(shape_vec);

                let _ = ov_shape_free(&mut shape);
            }
        }
        // Note: In real implementation, would iterate by index
        break;
    }

    shapes
}

/// Get output shapes from compiled model.
unsafe fn get_compiled_model_output_shapes(compiled_model: *mut c_void) -> Vec<Vec<i64>> {
    let mut shapes = Vec::new();
    let mut num_outputs: size_t = 0;

    if ov_compiled_model_outputs_size(compiled_model, &mut num_outputs) != OV_SUCCESS {
        return shapes;
    }

    for _i in 0..num_outputs {
        let mut port: *mut c_void = ptr::null_mut();
        if ov_compiled_model_output(compiled_model, &mut port) == OV_SUCCESS && !port.is_null()
        {
            let mut shape = OvShape {
                rank: 0,
                dims: ptr::null_mut(),
            };

            if ov_const_port_get_shape(port, &mut shape) == OV_SUCCESS {
                let len = if shape.rank > 0 { shape.rank as usize } else { 0 };
                let shape_vec = if !shape.dims.is_null() && len > 0 {
                    std::slice::from_raw_parts(shape.dims, len).to_vec()
                } else {
                    Vec::new()
                };
                shapes.push(shape_vec);

                let _ = ov_shape_free(&mut shape);
            }
        }
        break;
    }

    shapes
}

/// Run inference on OpenVINO model.
#[no_mangle]
pub extern "C" fn cudars_ov_run(
    handle: CudaRsOvModel,
    input_ptr: *const f32,
    input_len: c_ulonglong,
    shape_ptr: *const i64,
    shape_len: c_ulonglong,
    out_tensors: *mut *mut CudaRsOvTensor,
    out_count: *mut c_ulonglong,
) -> CudaRsResult {
    if input_ptr.is_null() || shape_ptr.is_null() || out_tensors.is_null() || out_count.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let models = OV_MODELS.lock().unwrap();
    let model = match models.get(handle) {
        Some(m) => m,
        None => return CudaRsResult::ErrorInvalidHandle,
    };

    let shape = unsafe { std::slice::from_raw_parts(shape_ptr, shape_len as usize) };

    // Verify input size
    let expected_size: i64 = shape.iter().product();
    if expected_size as u64 != input_len {
        return CudaRsResult::ErrorInvalidValue;
    }

    unsafe {
        let input_tensor = match create_input_tensor_from_slice(shape, input_ptr) {
            Ok(t) => t,
            Err(err) => return err,
        };

        let result =
            ov_infer_request_set_input_tensor_by_index(model.infer_request_ptr, 0, input_tensor);
        if result != OV_SUCCESS {
            ov_log_error("ov_infer_request_set_input_tensor_by_index", result);
            ov_tensor_free(input_tensor);
            return CudaRsResult::ErrorUnknown;
        }

        let result = ov_infer_request_infer(model.infer_request_ptr);
        if result != OV_SUCCESS {
            ov_log_error("ov_infer_request_infer", result);
            ov_tensor_free(input_tensor);
            return CudaRsResult::ErrorUnknown;
        }

        let result = collect_output_tensors(
            model.compiled_model_ptr,
            model.infer_request_ptr,
            out_tensors,
            out_count,
        );
        ov_tensor_free(input_tensor);
        return result;
    }
}

/// Run asynchronous inference on OpenVINO model.
#[no_mangle]
pub extern "C" fn cudars_ov_run_async(
    handle: CudaRsOvModel,
    input_ptr: *const f32,
    input_len: c_ulonglong,
    shape_ptr: *const i64,
    shape_len: c_ulonglong,
) -> CudaRsResult {
    if input_ptr.is_null() || shape_ptr.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let mut models = OV_MODELS.lock().unwrap();
    let model = match models.get_mut(handle) {
        Some(m) => m,
        None => return CudaRsResult::ErrorInvalidHandle,
    };

    let shape = unsafe { std::slice::from_raw_parts(shape_ptr, shape_len as usize) };

    let expected_size: i64 = shape.iter().product();
    if expected_size as u64 != input_len {
        return CudaRsResult::ErrorInvalidValue;
    }

    unsafe {
        if model.async_requests.is_empty() {
            return CudaRsResult::ErrorUnknown;
        }

        let slot = &mut model.async_requests[0];
        if slot.busy {
            return CudaRsResult::ErrorUnknown;
        }
        if !slot.input_tensor_ptr.is_null() {
            ov_tensor_free(slot.input_tensor_ptr);
            slot.input_tensor_ptr = ptr::null_mut();
        }
        slot.input_owned = None;

        let input_slice = std::slice::from_raw_parts(input_ptr, input_len as usize);
        let mut owned = vec![0.0f32; input_len as usize];
        owned.copy_from_slice(input_slice);
        let owned = owned.into_boxed_slice();

        let input_tensor = match create_input_tensor_from_slice(shape, owned.as_ptr()) {
            Ok(t) => t,
            Err(err) => return err,
        };

        let result = ov_infer_request_set_input_tensor_by_index(slot.request_ptr, 0, input_tensor);
        if result != OV_SUCCESS {
            ov_log_error("ov_infer_request_set_input_tensor_by_index_async", result);
            ov_tensor_free(input_tensor);
            return CudaRsResult::ErrorUnknown;
        }

        let result = ov_infer_request_start_async(slot.request_ptr);
        if result != OV_SUCCESS {
            ov_log_error("ov_infer_request_start_async", result);
            ov_tensor_free(input_tensor);
            return CudaRsResult::ErrorUnknown;
        }

        slot.input_tensor_ptr = input_tensor;
        slot.input_owned = Some(owned);
        slot.busy = true;
    }

    CudaRsResult::Success
}

/// Load OpenVINO model from file (v2 config).
#[no_mangle]
pub extern "C" fn cudars_ov_load_v2(
    model_path: *const c_char,
    config: *const CudaRsOvConfigV2,
    out_handle: *mut CudaRsOvModel,
) -> CudaRsResult {
    if model_path.is_null() || out_handle.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let path = unsafe {
        match CStr::from_ptr(model_path).to_str() {
            Ok(p) => p,
            Err(_) => return CudaRsResult::ErrorInvalidValue,
        }
    };

    let ov_config = if config.is_null() {
        CudaRsOvConfigV2::default()
    } else {
        unsafe { ptr::read(config) }
    };

    let properties = match read_properties_json(ov_config.properties_json_ptr, ov_config.properties_json_len) {
        Ok(v) => v,
        Err(err) => return err,
    };

    let (properties_prepared, async_request_count) =
        match prepare_properties(properties, ov_config.num_streams, ov_config.enable_profiling) {
            Ok(v) => v,
            Err(err) => return err,
        };

    let mut property_cstrings: Vec<(CString, CString)> = Vec::with_capacity(properties_prepared.len());
    for (key, value) in &properties_prepared {
        let key_cstr = match CString::new(key.clone()) {
            Ok(v) => v,
            Err(_) => return CudaRsResult::ErrorInvalidValue,
        };
        let value_cstr = match CString::new(value.clone()) {
            Ok(v) => v,
            Err(_) => return CudaRsResult::ErrorInvalidValue,
        };
        property_cstrings.push((key_cstr, value_cstr));
    }

    let device_name = match get_device_name_v2(&ov_config) {
        Ok(v) => v,
        Err(err) => return err,
    };
    let device_cstr = match CString::new(device_name.clone()) {
        Ok(s) => s,
        Err(_) => return CudaRsResult::ErrorInvalidValue,
    };

    let path_cstr = match CString::new(path) {
        Ok(s) => s,
        Err(_) => return CudaRsResult::ErrorInvalidValue,
    };

    unsafe {
        // Create core
        let mut core: *mut c_void = ptr::null_mut();
        if ov_core_create(&mut core) != OV_SUCCESS {
            return CudaRsResult::ErrorNotInitialized;
        }

        // Read model
        let mut model: *mut c_void = ptr::null_mut();
        let result = ov_core_read_model(core, path_cstr.as_ptr(), ptr::null(), &mut model);
        if result != OV_SUCCESS {
            ov_log_error("ov_core_read_model", result);
            ov_core_free(core);
            return CudaRsResult::ErrorInvalidValue;
        }

        // Compile model for device
        let mut compiled_model: *mut c_void = ptr::null_mut();
        let result = compile_model_with_properties(
            core,
            model,
            device_cstr.as_ptr(),
            &property_cstrings,
            &mut compiled_model,
        );

        if result != OV_SUCCESS {
            ov_log_error("ov_core_compile_model", result);
            ov_model_free(model);
            ov_core_free(core);
            return CudaRsResult::ErrorUnknown;
        }

        // Create infer request (sync)
        let mut infer_request: *mut c_void = ptr::null_mut();
        let result = ov_compiled_model_create_infer_request(compiled_model, &mut infer_request);
        if result != OV_SUCCESS {
            ov_log_error("ov_compiled_model_create_infer_request", result);
            ov_compiled_model_free(compiled_model);
            ov_core_free(core);
            return CudaRsResult::ErrorUnknown;
        }

        // Create async request pool
        let mut async_requests: Vec<OvInferRequestSlot> =
            Vec::with_capacity(async_request_count.max(1));
        for _ in 0..async_request_count.max(1) {
            let mut request_ptr: *mut c_void = ptr::null_mut();
            let result = ov_compiled_model_create_infer_request(compiled_model, &mut request_ptr);
            if result != OV_SUCCESS || request_ptr.is_null() {
                if result != OV_SUCCESS {
                    ov_log_error("ov_compiled_model_create_infer_request_async", result);
                }
                for slot in async_requests {
                    if !slot.request_ptr.is_null() {
                        ov_infer_request_free(slot.request_ptr);
                    }
                }
                ov_infer_request_free(infer_request);
                ov_compiled_model_free(compiled_model);
                ov_core_free(core);
                return CudaRsResult::ErrorUnknown;
            }
            async_requests.push(OvInferRequestSlot {
                request_ptr,
                input_tensor_ptr: ptr::null_mut(),
                input_owned: None,
                busy: false,
            });
        }

        // Query input/output shapes for layout inference and output helpers.
        let input_shapes = get_compiled_model_input_shapes(compiled_model);
        let output_shapes = get_compiled_model_output_shapes(compiled_model);
        if ov_debug_enabled() {
            if let Some(shape) = input_shapes.get(0) {
                debug_print_shape("compiled input", shape);
            } else {
                eprintln!("[cudars][openvino] compiled input shape not available");
            }
            if let Some(shape) = output_shapes.get(0) {
                debug_print_shape("compiled output[0]", shape);
            }
        }

        let ov_model = OvModel {
            core_ptr: core,
            model_ptr: model, // Keep model for reshape
            compiled_model_ptr: compiled_model,
            infer_request_ptr: infer_request,
            async_requests,
            input_shapes,
            output_shapes,
            model_path: path.to_string(),
            device_name,
            properties: properties_prepared,
            async_request_count,
        };

        let mut models = OV_MODELS.lock().unwrap();
        let id = models.insert(ov_model);
        *out_handle = id;
    }

    CudaRsResult::Success
}

/// Submit input to the async request queue.
#[no_mangle]
pub extern "C" fn cudars_ov_async_queue_submit(
    handle: CudaRsOvModel,
    input_ptr: *const f32,
    input_len: c_ulonglong,
    shape_ptr: *const i64,
    shape_len: c_ulonglong,
    out_request_id: *mut c_int,
) -> CudaRsResult {
    if input_ptr.is_null() || shape_ptr.is_null() || out_request_id.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let mut models = OV_MODELS.lock().unwrap();
    let model = match models.get_mut(handle) {
        Some(m) => m,
        None => return CudaRsResult::ErrorInvalidHandle,
    };

    if model.async_requests.is_empty() {
        return CudaRsResult::ErrorUnknown;
    }

    let shape = unsafe { std::slice::from_raw_parts(shape_ptr, shape_len as usize) };
    let expected_size: i64 = shape.iter().product();
    if expected_size as u64 != input_len {
        return CudaRsResult::ErrorInvalidValue;
    }

    let slot_index = match model
        .async_requests
        .iter()
        .position(|s| !s.busy)
    {
        Some(idx) => idx,
        None => return CudaRsResult::ErrorUnknown,
    };

    unsafe {
        let slot = &mut model.async_requests[slot_index];
        if !slot.input_tensor_ptr.is_null() {
            ov_tensor_free(slot.input_tensor_ptr);
            slot.input_tensor_ptr = ptr::null_mut();
        }
        slot.input_owned = None;

        let input_slice = std::slice::from_raw_parts(input_ptr, input_len as usize);
        let mut owned = vec![0.0f32; input_len as usize];
        owned.copy_from_slice(input_slice);
        let owned = owned.into_boxed_slice();

        let input_tensor = match create_input_tensor_from_slice(shape, owned.as_ptr()) {
            Ok(t) => t,
            Err(err) => return err,
        };

        let result = ov_infer_request_set_input_tensor_by_index(slot.request_ptr, 0, input_tensor);
        if result != OV_SUCCESS {
            ov_log_error("ov_infer_request_set_input_tensor_by_index_async_queue", result);
            ov_tensor_free(input_tensor);
            return CudaRsResult::ErrorUnknown;
        }

        let result = ov_infer_request_start_async(slot.request_ptr);
        if result != OV_SUCCESS {
            ov_log_error("ov_infer_request_start_async_queue", result);
            ov_tensor_free(input_tensor);
            return CudaRsResult::ErrorUnknown;
        }

        slot.input_tensor_ptr = input_tensor;
        slot.input_owned = Some(owned);
        slot.busy = true;
        *out_request_id = slot_index as c_int;
    }

    CudaRsResult::Success
}

/// Wait for async inference to complete and get results.
#[no_mangle]
pub extern "C" fn cudars_ov_wait(
    handle: CudaRsOvModel,
    out_tensors: *mut *mut CudaRsOvTensor,
    out_count: *mut c_ulonglong,
) -> CudaRsResult {
    if out_tensors.is_null() || out_count.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let mut models = OV_MODELS.lock().unwrap();
    let model = match models.get_mut(handle) {
        Some(m) => m,
        None => return CudaRsResult::ErrorInvalidHandle,
    };

    unsafe {
        if model.async_requests.is_empty() {
            return CudaRsResult::ErrorUnknown;
        }

        let slot = &mut model.async_requests[0];
        if !slot.busy {
            return CudaRsResult::ErrorInvalidValue;
        }

        let result = ov_infer_request_wait(slot.request_ptr);
        if result != OV_SUCCESS {
            ov_log_error("ov_infer_request_wait", result);
            return CudaRsResult::ErrorUnknown;
        }

        let result = collect_output_tensors(
            model.compiled_model_ptr,
            slot.request_ptr,
            out_tensors,
            out_count,
        );

        if !slot.input_tensor_ptr.is_null() {
            ov_tensor_free(slot.input_tensor_ptr);
            slot.input_tensor_ptr = ptr::null_mut();
        }
        slot.input_owned = None;
        slot.busy = false;
        return result;
    }
}

/// Wait for async inference request in the queue.
#[no_mangle]
pub extern "C" fn cudars_ov_async_queue_wait(
    handle: CudaRsOvModel,
    request_id: c_int,
    out_tensors: *mut *mut CudaRsOvTensor,
    out_count: *mut c_ulonglong,
) -> CudaRsResult {
    if out_tensors.is_null() || out_count.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let mut models = OV_MODELS.lock().unwrap();
    let model = match models.get_mut(handle) {
        Some(m) => m,
        None => return CudaRsResult::ErrorInvalidHandle,
    };

    let idx = request_id as usize;
    if idx >= model.async_requests.len() {
        return CudaRsResult::ErrorInvalidValue;
    }

    unsafe {
        let slot = &mut model.async_requests[idx];
        if !slot.busy {
            return CudaRsResult::ErrorInvalidValue;
        }

        let result = ov_infer_request_wait(slot.request_ptr);
        if result != OV_SUCCESS {
            ov_log_error("ov_infer_request_wait_queue", result);
            return CudaRsResult::ErrorUnknown;
        }

        let result = collect_output_tensors(
            model.compiled_model_ptr,
            slot.request_ptr,
            out_tensors,
            out_count,
        );

        if !slot.input_tensor_ptr.is_null() {
            ov_tensor_free(slot.input_tensor_ptr);
            slot.input_tensor_ptr = ptr::null_mut();
        }
        slot.input_owned = None;
        slot.busy = false;
        return result;
    }
}


/// Destroy OpenVINO model.
#[no_mangle]
pub extern "C" fn cudars_ov_destroy(handle: CudaRsOvModel) -> CudaRsResult {
    let mut models = OV_MODELS.lock().unwrap();
    match models.remove(handle) {
        Some(_) => CudaRsResult::Success,
        None => CudaRsResult::ErrorInvalidHandle,
    }
}

/// Free tensors returned by cudars_ov_run.
#[no_mangle]
pub extern "C" fn cudars_ov_free_tensors(
    tensors: *mut CudaRsOvTensor,
    count: c_ulonglong,
) -> CudaRsResult {
    if tensors.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let slice = unsafe { std::slice::from_raw_parts_mut(tensors, count as usize) };

    for t in slice.iter_mut() {
        unsafe {
            if !t.data.is_null() {
                let data_slice = std::slice::from_raw_parts_mut(t.data, t.data_len as usize);
                let _ = Box::from_raw(data_slice as *mut [f32]);
            }
            if !t.shape.is_null() {
                let shape_slice = std::slice::from_raw_parts_mut(t.shape, t.shape_len as usize);
                let _ = Box::from_raw(shape_slice as *mut [i64]);
            }
            t.data = ptr::null_mut();
            t.shape = ptr::null_mut();
            t.data_len = 0;
            t.shape_len = 0;
        }
    }

    unsafe {
        let _ = Box::from_raw(slice as *mut [CudaRsOvTensor]);
    }

    CudaRsResult::Success
}

/// Get the number of model inputs.
#[no_mangle]
pub extern "C" fn cudars_ov_get_input_count(
    handle: CudaRsOvModel,
    out_count: *mut c_ulonglong,
) -> CudaRsResult {
    if out_count.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let models = OV_MODELS.lock().unwrap();
    let model = match models.get(handle) {
        Some(m) => m,
        None => return CudaRsResult::ErrorInvalidHandle,
    };

    unsafe {
        let mut size: size_t = 0;
        let result = ov_compiled_model_inputs_size(model.compiled_model_ptr, &mut size);
        if result != OV_SUCCESS {
            ov_log_error("ov_compiled_model_inputs_size", result);
            return CudaRsResult::ErrorUnknown;
        }
        *out_count = size as c_ulonglong;
    }

    CudaRsResult::Success
}

/// Get model input information by index.
#[no_mangle]
pub extern "C" fn cudars_ov_get_input_info(
    handle: CudaRsOvModel,
    index: c_ulonglong,
    out_info: *mut CudaRsOvTensorInfo,
) -> CudaRsResult {
    if out_info.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let models = OV_MODELS.lock().unwrap();
    let model = match models.get(handle) {
        Some(m) => m,
        None => return CudaRsResult::ErrorInvalidHandle,
    };

    unsafe {
        // Initialize output
        (*out_info).name_ptr = ptr::null_mut();
        (*out_info).name_len = 0;
        (*out_info).shape = ptr::null_mut();
        (*out_info).shape_len = 0;
        (*out_info).element_type = 0;

        // Use cached input_shapes
        if index as usize >= model.input_shapes.len() {
            return CudaRsResult::ErrorInvalidValue;
        }
        
        let shape_vec = &model.input_shapes[index as usize];
        let shape_box = shape_vec.clone().into_boxed_slice();
        let shape_len = shape_box.len();
        let shape_ptr = Box::into_raw(shape_box) as *mut i64;
        
        (*out_info).shape = shape_ptr;
        (*out_info).shape_len = shape_len as c_ulonglong;
        (*out_info).element_type = OV_ELEMENT_TYPE_F32; // Default to F32
        
        // Generate default name
        let name = format!("input_{}", index);
        let name_cstring = CString::new(name).unwrap();
        let name_bytes = name_cstring.as_bytes_with_nul();
        let name_len = name_bytes.len() - 1; // Exclude null terminator
        let name_box = name_bytes.to_vec().into_boxed_slice();
        let name_ptr = Box::into_raw(name_box) as *mut c_char;
        
        (*out_info).name_ptr = name_ptr;
        (*out_info).name_len = name_len as c_ulonglong;
    }

    CudaRsResult::Success
}

/// Get the number of model outputs.
#[no_mangle]
pub extern "C" fn cudars_ov_get_output_count(
    handle: CudaRsOvModel,
    out_count: *mut c_ulonglong,
) -> CudaRsResult {
    if out_count.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let models = OV_MODELS.lock().unwrap();
    let model = match models.get(handle) {
        Some(m) => m,
        None => return CudaRsResult::ErrorInvalidHandle,
    };

    unsafe {
        let mut size: size_t = 0;
        let result = ov_compiled_model_outputs_size(model.compiled_model_ptr, &mut size);
        if result != OV_SUCCESS {
            ov_log_error("ov_compiled_model_outputs_size", result);
            return CudaRsResult::ErrorUnknown;
        }
        *out_count = size as c_ulonglong;
    }

    CudaRsResult::Success
}

/// Get model output information by index.
#[no_mangle]
pub extern "C" fn cudars_ov_get_output_info(
    handle: CudaRsOvModel,
    index: c_ulonglong,
    out_info: *mut CudaRsOvTensorInfo,
) -> CudaRsResult {
    if out_info.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let models = OV_MODELS.lock().unwrap();
    let model = match models.get(handle) {
        Some(m) => m,
        None => return CudaRsResult::ErrorInvalidHandle,
    };

    unsafe {
        // Initialize output
        (*out_info).name_ptr = ptr::null_mut();
        (*out_info).name_len = 0;
        (*out_info).shape = ptr::null_mut();
        (*out_info).shape_len = 0;
        (*out_info).element_type = 0;

        // Use cached output_shapes
        if index as usize >= model.output_shapes.len() {
            return CudaRsResult::ErrorInvalidValue;
        }
        
        let shape_vec = &model.output_shapes[index as usize];
        let shape_box = shape_vec.clone().into_boxed_slice();
        let shape_len = shape_box.len();
        let shape_ptr = Box::into_raw(shape_box) as *mut i64;
        
        (*out_info).shape = shape_ptr;
        (*out_info).shape_len = shape_len as c_ulonglong;
        (*out_info).element_type = OV_ELEMENT_TYPE_F32; // Default to F32
        
        // Generate default name
        let name = format!("output_{}", index);
        let name_cstring = CString::new(name).unwrap();
        let name_bytes = name_cstring.as_bytes_with_nul();
        let name_len = name_bytes.len() - 1; // Exclude null terminator
        let name_box = name_bytes.to_vec().into_boxed_slice();
        let name_ptr = Box::into_raw(name_box) as *mut c_char;
        
        (*out_info).name_ptr = name_ptr;
        (*out_info).name_len = name_len as c_ulonglong;
    }

    CudaRsResult::Success
}

/// Free tensor info returned by cudars_ov_get_input_info or cudars_ov_get_output_info.
#[no_mangle]
pub extern "C" fn cudars_ov_free_tensor_info(info: *mut CudaRsOvTensorInfo) -> CudaRsResult {
    if info.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    unsafe {
        let info_ref = &mut *info;
        
        if !info_ref.name_ptr.is_null() && info_ref.name_len > 0 {
            let name_slice = std::slice::from_raw_parts_mut(
                info_ref.name_ptr as *mut u8,
                info_ref.name_len as usize,
            );
            let _ = Box::from_raw(name_slice as *mut [u8]);
            info_ref.name_ptr = ptr::null_mut();
            info_ref.name_len = 0;
        }
        
        if !info_ref.shape.is_null() && info_ref.shape_len > 0 {
            let shape_slice = std::slice::from_raw_parts_mut(
                info_ref.shape,
                info_ref.shape_len as usize,
            );
            let _ = Box::from_raw(shape_slice as *mut [i64]);
            info_ref.shape = ptr::null_mut();
            info_ref.shape_len = 0;
        }
    }

    CudaRsResult::Success
}

/// Reshape OpenVINO model to fixed dimensions.
#[no_mangle]
pub extern "C" fn cudars_ov_reshape_fixed(
    handle: CudaRsOvModel,
    shape_ptr: *const i64,
    shape_len: c_ulonglong,
) -> CudaRsResult {
    if shape_ptr.is_null() || shape_len == 0 {
        return CudaRsResult::ErrorInvalidValue;
    }

    let mut models = OV_MODELS.lock().unwrap();
    let model = match models.get_mut(handle) {
        Some(m) => m,
        None => return CudaRsResult::ErrorInvalidHandle,
    };

    let shape = unsafe { std::slice::from_raw_parts(shape_ptr, shape_len as usize) };

    unsafe {
        // Create partial shape for the first input
        let mut partial_shape = OvPartialShape {
            rank: shape.len() as i64,
            dims: OvPartialShapeDims {
                min: ptr::null_mut(),
                max: ptr::null_mut(),
            },
        };

        let result = ov_partial_shape_create_static(
            shape.len() as i64,
            shape.as_ptr(),
            &mut partial_shape,
        );
        if result != OV_SUCCESS {
            ov_log_error("ov_partial_shape_create_static", result);
            return CudaRsResult::ErrorUnknown;
        }

        // Reshape the model (only first input for now)
        let result = ov_model_reshape(model.model_ptr, ptr::null(), &partial_shape, 1);
        ov_partial_shape_free(&mut partial_shape);

        if result != OV_SUCCESS {
            ov_log_error("ov_model_reshape", result);
            return CudaRsResult::ErrorUnknown;
        }

        // Free old compiled model and inference requests
        if !model.infer_request_ptr.is_null() {
            ov_infer_request_free(model.infer_request_ptr);
            model.infer_request_ptr = ptr::null_mut();
        }
        for slot in model.async_requests.iter_mut() {
            if !slot.input_tensor_ptr.is_null() {
                ov_tensor_free(slot.input_tensor_ptr);
                slot.input_tensor_ptr = ptr::null_mut();
            }
            slot.input_owned = None;
            slot.busy = false;
            if !slot.request_ptr.is_null() {
                ov_infer_request_free(slot.request_ptr);
                slot.request_ptr = ptr::null_mut();
            }
        }
        model.async_requests.clear();
        if !model.compiled_model_ptr.is_null() {
            ov_compiled_model_free(model.compiled_model_ptr);
            model.compiled_model_ptr = ptr::null_mut();
        }

        // Recompile model
        let device_cstr = match CString::new(model.device_name.clone()) {
            Ok(s) => s,
            Err(_) => return CudaRsResult::ErrorInvalidValue,
        };

        let mut property_cstrings: Vec<(CString, CString)> = Vec::with_capacity(model.properties.len());
        for (key, value) in &model.properties {
            let key_cstr = match CString::new(key.clone()) {
                Ok(v) => v,
                Err(_) => return CudaRsResult::ErrorInvalidValue,
            };
            let value_cstr = match CString::new(value.clone()) {
                Ok(v) => v,
                Err(_) => return CudaRsResult::ErrorInvalidValue,
            };
            property_cstrings.push((key_cstr, value_cstr));
        }

        let mut compiled_model: *mut c_void = ptr::null_mut();
        let result = compile_model_with_properties(
            model.core_ptr,
            model.model_ptr,
            device_cstr.as_ptr(),
            &property_cstrings,
            &mut compiled_model,
        );

        if result != OV_SUCCESS {
            ov_log_error("ov_core_compile_model_after_reshape", result);
            return CudaRsResult::ErrorUnknown;
        }

        model.compiled_model_ptr = compiled_model;

        // Recreate infer request
        let mut infer_request: *mut c_void = ptr::null_mut();
        let result = ov_compiled_model_create_infer_request(compiled_model, &mut infer_request);
        if result != OV_SUCCESS {
            ov_log_error("ov_compiled_model_create_infer_request_after_reshape", result);
            return CudaRsResult::ErrorUnknown;
        }
        model.infer_request_ptr = infer_request;

        // Recreate async requests
        let mut async_requests: Vec<OvInferRequestSlot> =
            Vec::with_capacity(model.async_request_count.max(1));
        for _ in 0..model.async_request_count.max(1) {
            let mut request_ptr: *mut c_void = ptr::null_mut();
            let result = ov_compiled_model_create_infer_request(compiled_model, &mut request_ptr);
            if result != OV_SUCCESS || request_ptr.is_null() {
                if result != OV_SUCCESS {
                    ov_log_error("ov_compiled_model_create_infer_request_async_after_reshape", result);
                }
                for slot in async_requests {
                    if !slot.request_ptr.is_null() {
                        ov_infer_request_free(slot.request_ptr);
                    }
                }
                return CudaRsResult::ErrorUnknown;
            }
            async_requests.push(OvInferRequestSlot {
                request_ptr,
                input_tensor_ptr: ptr::null_mut(),
                input_owned: None,
                busy: false,
            });
        }
        model.async_requests = async_requests;

        // Update cached shapes
        model.input_shapes = get_compiled_model_input_shapes(compiled_model);
        model.output_shapes = get_compiled_model_output_shapes(compiled_model);

        if ov_debug_enabled() {
            if let Some(shape) = model.input_shapes.get(0) {
                debug_print_shape("reshaped input", shape);
            }
        }
    }

    CudaRsResult::Success
}

/// Reshape OpenVINO model with partial shape (dynamic dimensions).
#[no_mangle]
pub extern "C" fn cudars_ov_reshape_dynamic(
    handle: CudaRsOvModel,
    partial_shape_ptr: *const CudaRsOvPartialShapeArray,
) -> CudaRsResult {
    if partial_shape_ptr.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let partial_shape = unsafe { &*partial_shape_ptr };
    if partial_shape.dims.is_null() || partial_shape.rank == 0 {
        return CudaRsResult::ErrorInvalidValue;
    }

    let mut models = OV_MODELS.lock().unwrap();
    let model = match models.get_mut(handle) {
        Some(m) => m,
        None => return CudaRsResult::ErrorInvalidHandle,
    };

    let dims_slice = unsafe { std::slice::from_raw_parts(partial_shape.dims, partial_shape.rank as usize) };

    unsafe {
        // Convert to OpenVINO partial shape
        let mut shape_dims: Vec<i64> = Vec::with_capacity(dims_slice.len());
        for dim in dims_slice {
            if dim.is_static != 0 {
                shape_dims.push(dim.value);
            } else {
                // Dynamic dimension: use -1
                shape_dims.push(-1);
            }
        }

        // Create partial shape
        let mut ov_partial_shape = OvPartialShape {
            rank: shape_dims.len() as i64,
            dims: OvPartialShapeDims {
                min: ptr::null_mut(),
                max: ptr::null_mut(),
            },
        };

        let result = ov_partial_shape_create_static(
            shape_dims.len() as i64,
            shape_dims.as_ptr(),
            &mut ov_partial_shape,
        );
        if result != OV_SUCCESS {
            ov_log_error("ov_partial_shape_create_static_dynamic", result);
            return CudaRsResult::ErrorUnknown;
        }

        // Reshape the model
        let result = ov_model_reshape(model.model_ptr, ptr::null(), &ov_partial_shape, 1);
        ov_partial_shape_free(&mut ov_partial_shape);

        if result != OV_SUCCESS {
            ov_log_error("ov_model_reshape_dynamic", result);
            return CudaRsResult::ErrorUnknown;
        }

        // Free old compiled model and inference requests
        if !model.infer_request_ptr.is_null() {
            ov_infer_request_free(model.infer_request_ptr);
            model.infer_request_ptr = ptr::null_mut();
        }
        for slot in model.async_requests.iter_mut() {
            if !slot.input_tensor_ptr.is_null() {
                ov_tensor_free(slot.input_tensor_ptr);
                slot.input_tensor_ptr = ptr::null_mut();
            }
            slot.input_owned = None;
            slot.busy = false;
            if !slot.request_ptr.is_null() {
                ov_infer_request_free(slot.request_ptr);
                slot.request_ptr = ptr::null_mut();
            }
        }
        model.async_requests.clear();
        if !model.compiled_model_ptr.is_null() {
            ov_compiled_model_free(model.compiled_model_ptr);
            model.compiled_model_ptr = ptr::null_mut();
        }

        // Recompile model
        let device_cstr = match CString::new(model.device_name.clone()) {
            Ok(s) => s,
            Err(_) => return CudaRsResult::ErrorInvalidValue,
        };

        let mut property_cstrings: Vec<(CString, CString)> = Vec::with_capacity(model.properties.len());
        for (key, value) in &model.properties {
            let key_cstr = match CString::new(key.clone()) {
                Ok(v) => v,
                Err(_) => return CudaRsResult::ErrorInvalidValue,
            };
            let value_cstr = match CString::new(value.clone()) {
                Ok(v) => v,
                Err(_) => return CudaRsResult::ErrorInvalidValue,
            };
            property_cstrings.push((key_cstr, value_cstr));
        }

        let mut compiled_model: *mut c_void = ptr::null_mut();
        let result = compile_model_with_properties(
            model.core_ptr,
            model.model_ptr,
            device_cstr.as_ptr(),
            &property_cstrings,
            &mut compiled_model,
        );

        if result != OV_SUCCESS {
            ov_log_error("ov_core_compile_model_after_reshape_dynamic", result);
            return CudaRsResult::ErrorUnknown;
        }

        model.compiled_model_ptr = compiled_model;

        // Recreate infer request
        let mut infer_request: *mut c_void = ptr::null_mut();
        let result = ov_compiled_model_create_infer_request(compiled_model, &mut infer_request);
        if result != OV_SUCCESS {
            ov_log_error("ov_compiled_model_create_infer_request_after_reshape_dynamic", result);
            return CudaRsResult::ErrorUnknown;
        }
        model.infer_request_ptr = infer_request;

        // Recreate async requests
        let mut async_requests: Vec<OvInferRequestSlot> =
            Vec::with_capacity(model.async_request_count.max(1));
        for _ in 0..model.async_request_count.max(1) {
            let mut request_ptr: *mut c_void = ptr::null_mut();
            let result = ov_compiled_model_create_infer_request(compiled_model, &mut request_ptr);
            if result != OV_SUCCESS || request_ptr.is_null() {
                if result != OV_SUCCESS {
                    ov_log_error("ov_compiled_model_create_infer_request_async_after_reshape_dynamic", result);
                }
                for slot in async_requests {
                    if !slot.request_ptr.is_null() {
                        ov_infer_request_free(slot.request_ptr);
                    }
                }
                return CudaRsResult::ErrorUnknown;
            }
            async_requests.push(OvInferRequestSlot {
                request_ptr,
                input_tensor_ptr: ptr::null_mut(),
                input_owned: None,
                busy: false,
            });
        }
        model.async_requests = async_requests;

        // Update cached shapes
        model.input_shapes = get_compiled_model_input_shapes(compiled_model);
        model.output_shapes = get_compiled_model_output_shapes(compiled_model);

        if ov_debug_enabled() {
            if let Some(shape) = model.input_shapes.get(0) {
                debug_print_shape("reshaped dynamic input", shape);
            }
        }
    }

    CudaRsResult::Success
}
