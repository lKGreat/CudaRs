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
    fn ov_model_get_inputs(
        model: *mut c_void,
        inputs: *mut *mut c_void,
        num_inputs: *mut size_t,
    ) -> c_int;
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
    fn ov_infer_request_set_input_tensor(
        infer_request: *mut c_void,
        tensor: *mut c_void,
    ) -> c_int;
    fn ov_infer_request_set_input_tensor_by_index(
        infer_request: *mut c_void,
        index: size_t,
        tensor: *mut c_void,
    ) -> c_int;
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
        shape: *const i64,
        shape_size: size_t,
        data: *mut c_void,
        tensor: *mut *mut c_void,
    ) -> c_int;
    fn ov_tensor_create(
        element_type: c_int,
        shape: *const i64,
        shape_size: size_t,
        tensor: *mut *mut c_void,
    ) -> c_int;
    fn ov_tensor_free(tensor: *mut c_void);
    fn ov_tensor_data(tensor: *mut c_void, data: *mut *mut c_void) -> c_int;
    fn ov_tensor_get_shape(
        tensor: *mut c_void,
        shape: *mut *mut i64,
        shape_size: *mut size_t,
    ) -> c_int;
    fn ov_tensor_get_size(tensor: *mut c_void, size: *mut size_t) -> c_int;
    fn ov_tensor_get_element_type(tensor: *mut c_void, element_type: *mut c_int) -> c_int;

    // Port (for input/output info)
    fn ov_port_get_any_name(port: *mut c_void, name: *mut *mut c_char) -> c_int;
    fn ov_port_get_shape(
        port: *mut c_void,
        shape: *mut *mut i64,
        shape_size: *mut size_t,
    ) -> c_int;
    fn ov_port_get_element_type(port: *mut c_void, element_type: *mut c_int) -> c_int;

    // Shape
    fn ov_shape_free(shape: *mut i64);

    // Devices string free
    fn ov_free(ptr: *mut c_void);
}

// OpenVINO element types
const OV_ELEMENT_TYPE_F32: c_int = 10; // ov_element_type_e::F32
const OV_ELEMENT_TYPE_F16: c_int = 9;
const OV_ELEMENT_TYPE_I32: c_int = 6;
const OV_ELEMENT_TYPE_I64: c_int = 8;
const OV_ELEMENT_TYPE_U8: c_int = 3;

/// OpenVINO status codes
const OV_SUCCESS: c_int = 0;

/// Opaque handle for OpenVINO model
pub type CudaRsOvModel = u64;

/// OpenVINO tensor descriptor
#[repr(C)]
pub struct CudaRsOvTensor {
    pub data: *mut f32,
    pub data_len: c_ulonglong,
    pub shape: *mut i64,
    pub shape_len: c_ulonglong,
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
    compiled_model_ptr: *mut c_void,
    infer_request_ptr: *mut c_void,
    input_shapes: Vec<Vec<i64>>,
    output_shapes: Vec<Vec<i64>>,
    device_name: String,
}

unsafe impl Send for OvModel {}
unsafe impl Sync for OvModel {}

impl Drop for OvModel {
    fn drop(&mut self) {
        unsafe {
            if !self.infer_request_ptr.is_null() {
                ov_infer_request_free(self.infer_request_ptr);
            }
            if !self.compiled_model_ptr.is_null() {
                ov_compiled_model_free(self.compiled_model_ptr);
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

fn json_value_to_string(value: &Value) -> String {
    match value {
        Value::String(v) => v.clone(),
        Value::Number(v) => v.to_string(),
        Value::Bool(v) => v.to_string(),
        Value::Null => "null".to_string(),
        Value::Array(_) | Value::Object(_) => value.to_string(),
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

    let properties = if !ov_config.properties_json_ptr.is_null() && ov_config.properties_json_len > 0
    {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                ov_config.properties_json_ptr as *const u8,
                ov_config.properties_json_len as usize,
            )
        };
        let json = match std::str::from_utf8(bytes) {
            Ok(v) => v,
            Err(_) => return CudaRsResult::ErrorInvalidValue,
        };
        match parse_properties_json(json) {
            Ok(v) => v,
            Err(err) => return err,
        }
    } else {
        Vec::new()
    };

    let mut property_cstrings: Vec<(CString, CString)> = Vec::with_capacity(properties.len());
    for (key, value) in properties {
        let key_cstr = match CString::new(key) {
            Ok(v) => v,
            Err(_) => return CudaRsResult::ErrorInvalidValue,
        };
        let value_cstr = match CString::new(value) {
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

        // Free the model (compiled model has its own copy)
        ov_model_free(model);

        if result != OV_SUCCESS {
            ov_core_free(core);
            return CudaRsResult::ErrorUnknown;
        }

        // Create infer request
        let mut infer_request: *mut c_void = ptr::null_mut();
        let result = ov_compiled_model_create_infer_request(compiled_model, &mut infer_request);
        if result != OV_SUCCESS {
            ov_compiled_model_free(compiled_model);
            ov_core_free(core);
            return CudaRsResult::ErrorUnknown;
        }

        // Avoid querying port shapes during load to reduce OpenVINO API surface.
        // Pipelines that need layout info can fall back to NCHW.
        let input_shapes = Vec::new();
        let output_shapes = Vec::new();

        let ov_model = OvModel {
            core_ptr: core,
            compiled_model_ptr: compiled_model,
            infer_request_ptr: infer_request,
            input_shapes,
            output_shapes,
            device_name,
        };

        let mut models = OV_MODELS.lock().unwrap();
        let id = models.insert(ov_model);
        *out_handle = id;
    }

    CudaRsResult::Success
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
            let mut shape_ptr: *mut i64 = ptr::null_mut();
            let mut shape_size: size_t = 0;

            if ov_port_get_shape(port, &mut shape_ptr, &mut shape_size) == OV_SUCCESS {
                let shape = std::slice::from_raw_parts(shape_ptr, shape_size).to_vec();
                shapes.push(shape);

                if !shape_ptr.is_null() {
                    ov_shape_free(shape_ptr);
                }
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
            let mut shape_ptr: *mut i64 = ptr::null_mut();
            let mut shape_size: size_t = 0;

            if ov_port_get_shape(port, &mut shape_ptr, &mut shape_size) == OV_SUCCESS {
                let shape = std::slice::from_raw_parts(shape_ptr, shape_size).to_vec();
                shapes.push(shape);

                if !shape_ptr.is_null() {
                    ov_shape_free(shape_ptr);
                }
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
        // Create input tensor from host data
        let mut input_tensor: *mut c_void = ptr::null_mut();
        let result = ov_tensor_create_from_host_ptr(
            OV_ELEMENT_TYPE_F32,
            shape.as_ptr(),
            shape.len(),
            input_ptr as *mut c_void,
            &mut input_tensor,
        );

        if result != OV_SUCCESS {
            return CudaRsResult::ErrorOutOfMemory;
        }

        // Set input tensor
        let result = ov_infer_request_set_input_tensor(model.infer_request_ptr, input_tensor);
        if result != OV_SUCCESS {
            ov_tensor_free(input_tensor);
            return CudaRsResult::ErrorUnknown;
        }

        // Run inference
        let result = ov_infer_request_infer(model.infer_request_ptr);
        if result != OV_SUCCESS {
            ov_tensor_free(input_tensor);
            return CudaRsResult::ErrorUnknown;
        }

        // Get output tensors
        let mut num_outputs: size_t = 0;
        if ov_compiled_model_outputs_size(model.compiled_model_ptr, &mut num_outputs) != OV_SUCCESS
        {
            ov_tensor_free(input_tensor);
            return CudaRsResult::ErrorUnknown;
        }

        let mut tensors: Vec<CudaRsOvTensor> = Vec::with_capacity(num_outputs);

        for i in 0..num_outputs {
            let mut output_tensor: *mut c_void = ptr::null_mut();
            let result = ov_infer_request_get_output_tensor_by_index(
                model.infer_request_ptr,
                i,
                &mut output_tensor,
            );

            if result != OV_SUCCESS || output_tensor.is_null() {
                continue;
            }

            // Get tensor data
            let mut data_ptr: *mut c_void = ptr::null_mut();
            if ov_tensor_data(output_tensor, &mut data_ptr) != OV_SUCCESS {
                continue;
            }

            // Get tensor size
            let mut tensor_size: size_t = 0;
            if ov_tensor_get_size(output_tensor, &mut tensor_size) != OV_SUCCESS {
                continue;
            }

            // Get tensor shape
            let mut out_shape_ptr: *mut i64 = ptr::null_mut();
            let mut out_shape_len: size_t = 0;
            if ov_tensor_get_shape(output_tensor, &mut out_shape_ptr, &mut out_shape_len)
                != OV_SUCCESS
            {
                continue;
            }

            // Copy shape
            let shape_vec: Vec<i64> =
                std::slice::from_raw_parts(out_shape_ptr, out_shape_len).to_vec();

            // Copy data
            let data_slice = std::slice::from_raw_parts(data_ptr as *const f32, tensor_size);
            let data_vec: Vec<f32> = data_slice.to_vec();

            // Free OpenVINO shape
            if !out_shape_ptr.is_null() {
                ov_shape_free(out_shape_ptr);
            }

            // Package output
            let data_len = tensor_size as u64;
            let shape_len = shape_vec.len() as u64;

            let shape_box = shape_vec.into_boxed_slice();
            let data_box = data_vec.into_boxed_slice();

            tensors.push(CudaRsOvTensor {
                data: Box::into_raw(data_box) as *mut f32,
                data_len,
                shape: Box::into_raw(shape_box) as *mut i64,
                shape_len,
            });
        }

        // Clean up input tensor
        ov_tensor_free(input_tensor);

        // Return results
        let count = tensors.len() as u64;
        let boxed = tensors.into_boxed_slice();
        *out_tensors = Box::into_raw(boxed) as *mut CudaRsOvTensor;
        *out_count = count;
    }

    CudaRsResult::Success
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

    let models = OV_MODELS.lock().unwrap();
    let model = match models.get(handle) {
        Some(m) => m,
        None => return CudaRsResult::ErrorInvalidHandle,
    };

    let shape = unsafe { std::slice::from_raw_parts(shape_ptr, shape_len as usize) };

    let expected_size: i64 = shape.iter().product();
    if expected_size as u64 != input_len {
        return CudaRsResult::ErrorInvalidValue;
    }

    unsafe {
        let mut input_tensor: *mut c_void = ptr::null_mut();
        let result = ov_tensor_create_from_host_ptr(
            OV_ELEMENT_TYPE_F32,
            shape.as_ptr(),
            shape.len(),
            input_ptr as *mut c_void,
            &mut input_tensor,
        );

        if result != OV_SUCCESS {
            return CudaRsResult::ErrorOutOfMemory;
        }

        let result = ov_infer_request_set_input_tensor(model.infer_request_ptr, input_tensor);
        if result != OV_SUCCESS {
            ov_tensor_free(input_tensor);
            return CudaRsResult::ErrorUnknown;
        }

        // Start async inference
        let result = ov_infer_request_start_async(model.infer_request_ptr);
        if result != OV_SUCCESS {
            ov_tensor_free(input_tensor);
            return CudaRsResult::ErrorUnknown;
        }

        // Note: Input tensor should be kept alive until inference completes
        // In a real implementation, we'd track this in the model state
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

    let models = OV_MODELS.lock().unwrap();
    let model = match models.get(handle) {
        Some(m) => m,
        None => return CudaRsResult::ErrorInvalidHandle,
    };

    unsafe {
        // Wait for inference to complete
        let result = ov_infer_request_wait(model.infer_request_ptr);
        if result != OV_SUCCESS {
            return CudaRsResult::ErrorUnknown;
        }

        // Get outputs (same logic as synchronous)
        let mut num_outputs: size_t = 0;
        if ov_compiled_model_outputs_size(model.compiled_model_ptr, &mut num_outputs) != OV_SUCCESS
        {
            return CudaRsResult::ErrorUnknown;
        }

        let mut tensors: Vec<CudaRsOvTensor> = Vec::with_capacity(num_outputs);

        for i in 0..num_outputs {
            let mut output_tensor: *mut c_void = ptr::null_mut();
            let result = ov_infer_request_get_output_tensor_by_index(
                model.infer_request_ptr,
                i,
                &mut output_tensor,
            );

            if result != OV_SUCCESS || output_tensor.is_null() {
                continue;
            }

            let mut data_ptr: *mut c_void = ptr::null_mut();
            if ov_tensor_data(output_tensor, &mut data_ptr) != OV_SUCCESS {
                continue;
            }

            let mut tensor_size: size_t = 0;
            if ov_tensor_get_size(output_tensor, &mut tensor_size) != OV_SUCCESS {
                continue;
            }

            let mut out_shape_ptr: *mut i64 = ptr::null_mut();
            let mut out_shape_len: size_t = 0;
            if ov_tensor_get_shape(output_tensor, &mut out_shape_ptr, &mut out_shape_len)
                != OV_SUCCESS
            {
                continue;
            }

            let shape_vec: Vec<i64> =
                std::slice::from_raw_parts(out_shape_ptr, out_shape_len).to_vec();
            let data_slice = std::slice::from_raw_parts(data_ptr as *const f32, tensor_size);
            let data_vec: Vec<f32> = data_slice.to_vec();

            if !out_shape_ptr.is_null() {
                ov_shape_free(out_shape_ptr);
            }

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
        *out_tensors = Box::into_raw(tensors.into_boxed_slice()) as *mut CudaRsOvTensor;
        *out_count = count;
    }

    CudaRsResult::Success
}

/// Get input info for OpenVINO model.
#[no_mangle]
pub extern "C" fn cudars_ov_get_input_info(
    handle: CudaRsOvModel,
    index: c_int,
    out_shape: *mut i64,
    out_shape_len: *mut c_int,
    max_shape_len: c_int,
) -> CudaRsResult {
    if out_shape.is_null() || out_shape_len.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let models = OV_MODELS.lock().unwrap();
    let model = match models.get(handle) {
        Some(m) => m,
        None => return CudaRsResult::ErrorInvalidHandle,
    };

    let idx = index as usize;
    if idx >= model.input_shapes.len() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let shape = &model.input_shapes[idx];
    let len = shape.len().min(max_shape_len as usize);

    unsafe {
        for i in 0..len {
            *out_shape.add(i) = shape[i];
        }
        *out_shape_len = len as c_int;
    }

    CudaRsResult::Success
}

/// Get output info for OpenVINO model.
#[no_mangle]
pub extern "C" fn cudars_ov_get_output_info(
    handle: CudaRsOvModel,
    index: c_int,
    out_shape: *mut i64,
    out_shape_len: *mut c_int,
    max_shape_len: c_int,
) -> CudaRsResult {
    if out_shape.is_null() || out_shape_len.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let models = OV_MODELS.lock().unwrap();
    let model = match models.get(handle) {
        Some(m) => m,
        None => return CudaRsResult::ErrorInvalidHandle,
    };

    let idx = index as usize;
    if idx >= model.output_shapes.len() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let shape = &model.output_shapes[idx];
    let len = shape.len().min(max_shape_len as usize);

    unsafe {
        for i in 0..len {
            *out_shape.add(i) = shape[i];
        }
        *out_shape_len = len as c_int;
    }

    CudaRsResult::Success
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
