//! ONNX Runtime FFI exports (Rust-side implementation).

use crate::CudaRsResult;
use libc::{c_char, c_int, c_ulonglong};
use std::ffi::CStr;
use std::sync::Mutex;

use ndarray::IxDyn;
use onnxruntime::{environment::Environment, session::Session, GraphOptimizationLevel};

use crate::runtime::HandleManager;

/// Opaque handle for ONNX Runtime session.
pub type CudaRsOnnxSession = u64;

struct OrtSession(Session<'static>);

unsafe impl Send for OrtSession {}

#[repr(C)]
pub struct CudaRsTensor {
    pub data: *mut f32,
    pub data_len: c_ulonglong,
    pub shape: *mut i64,
    pub shape_len: c_ulonglong,
}

lazy_static::lazy_static! {
    static ref ORT_ENV: Environment = Environment::builder()
        .with_name("cudars_ort")
        .with_log_level(onnxruntime::LoggingLevel::Warning)
        .build()
        .expect("Failed to create ONNX Runtime environment");
    static ref ORT_SESSIONS: Mutex<HandleManager<OrtSession>> = Mutex::new(HandleManager::new());
}

fn make_session(model_path: &str, _device_id: i32) -> Result<Session<'static>, CudaRsResult> {
    let session = ORT_ENV
        .new_session_builder()
        .map_err(|_| CudaRsResult::ErrorUnknown)?
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(|_| CudaRsResult::ErrorUnknown)?
        .with_number_threads(1)
        .map_err(|_| CudaRsResult::ErrorUnknown)?
        .with_model_from_file(model_path)
        .map_err(|_| CudaRsResult::ErrorUnknown)?;

    // Safety: session lifetime is tied to ORT_ENV which is static.
    let session_static: Session<'static> = unsafe { std::mem::transmute::<Session<'_>, Session<'static>>(session) };
    Ok(session_static)
}

/// Create an ONNX Runtime session.
#[no_mangle]
pub extern "C" fn cudars_onnx_create(
    model_path: *const c_char,
    device_id: c_int,
    out_handle: *mut CudaRsOnnxSession,
) -> CudaRsResult {
    if model_path.is_null() || out_handle.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let path = unsafe {
        match CStr::from_ptr(model_path).to_str() {
            Ok(p) => p.to_string(),
            Err(_) => return CudaRsResult::ErrorInvalidValue,
        }
    };

    match make_session(&path, device_id) {
        Ok(session) => {
            let mut sessions = ORT_SESSIONS.lock().unwrap();
            let id = sessions.insert(OrtSession(session));
            unsafe { *out_handle = id; }
            CudaRsResult::Success
        }
        Err(code) => code,
    }
}

/// Destroy an ONNX Runtime session.
#[no_mangle]
pub extern "C" fn cudars_onnx_destroy(handle: CudaRsOnnxSession) -> CudaRsResult {
    let mut sessions = ORT_SESSIONS.lock().unwrap();
    match sessions.remove(handle) {
        Some(_) => CudaRsResult::Success,
        None => CudaRsResult::ErrorInvalidHandle,
    }
}

/// Run an ONNX Runtime session.
#[no_mangle]
pub extern "C" fn cudars_onnx_run(
    handle: CudaRsOnnxSession,
    input_ptr: *const f32,
    input_len: c_ulonglong,
    shape_ptr: *const i64,
    shape_len: c_ulonglong,
    out_tensors: *mut *mut CudaRsTensor,
    out_count: *mut c_ulonglong,
) -> CudaRsResult {
    if input_ptr.is_null() || shape_ptr.is_null() || out_tensors.is_null() || out_count.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let mut sessions = ORT_SESSIONS.lock().unwrap();
    let session = match sessions.get_mut(handle) {
        Some(s) => s,
        None => return CudaRsResult::ErrorInvalidHandle,
    };

    let shape = unsafe { std::slice::from_raw_parts(shape_ptr, shape_len as usize) };
    let input = unsafe { std::slice::from_raw_parts(input_ptr, input_len as usize) };

    let shape_usize: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
    let array = ndarray::Array::from_shape_vec(IxDyn(&shape_usize), input.to_vec())
        .map_err(|_| CudaRsResult::ErrorInvalidValue);

    let input_tensor = match array {
        Ok(t) => t,
        Err(code) => return code,
    };

    let outputs: Vec<onnxruntime::tensor::OrtOwnedTensor<f32, IxDyn>> =
        match session.0.run(vec![input_tensor]) {
            Ok(o) => o,
            Err(_) => return CudaRsResult::ErrorUnknown,
        };

    let mut tensors: Vec<CudaRsTensor> = Vec::with_capacity(outputs.len());

    for output in outputs {
        let shape_vec: Vec<i64> = output.shape().iter().map(|d| *d as i64).collect();
        let data_vec: Vec<f32> = output.iter().copied().collect();

        let shape_len = shape_vec.len() as u64;
        let data_len = data_vec.len() as u64;

        let shape_box = shape_vec.into_boxed_slice();
        let data_box = data_vec.into_boxed_slice();

        let shape_ptr = Box::into_raw(shape_box) as *mut i64;
        let data_ptr = Box::into_raw(data_box) as *mut f32;

        tensors.push(CudaRsTensor {
            data: data_ptr,
            data_len,
            shape: shape_ptr,
            shape_len,
        });
    }

    let count = tensors.len() as u64;
    let boxed = tensors.into_boxed_slice();
    let ptr = Box::into_raw(boxed) as *mut CudaRsTensor;

    unsafe {
        *out_tensors = ptr;
        *out_count = count;
    }

    CudaRsResult::Success
}

/// Free tensors returned by cudars_onnx_run.
#[no_mangle]
pub extern "C" fn cudars_onnx_free_tensors(tensors: *mut CudaRsTensor, count: c_ulonglong) -> CudaRsResult {
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
            t.data = std::ptr::null_mut();
            t.shape = std::ptr::null_mut();
            t.data_len = 0;
            t.shape_len = 0;
        }
    }

    unsafe { let _ = Box::from_raw(slice as *mut [CudaRsTensor]); }

    CudaRsResult::Success
}

// shape length is stored in CudaRsTensor
