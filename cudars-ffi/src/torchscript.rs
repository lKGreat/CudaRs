//! TorchScript FFI exports (Rust-side implementation).
//!
//! Provides TorchScript model loading and inference via libtorch C++ API.
//! Supports .pt and .torchscript model formats.

use crate::CudaRsResult;
use libc::{c_char, c_int, c_ulonglong, c_void};
use std::ffi::{CStr, CString};
use std::ptr;
use std::sync::Mutex;

use crate::runtime::HandleManager;

// LibTorch C ABI bindings
#[link(name = "torch")]
#[link(name = "torch_cpu")]
#[link(name = "c10")]
extern "C" {
    // Module loading
    fn torch_jit_load(path: *const c_char, device_type: c_int, device_index: c_int) -> *mut c_void;
    fn torch_jit_module_destroy(module: *mut c_void);

    // Forward pass
    fn torch_jit_module_forward(
        module: *mut c_void,
        inputs: *const TorchTensorHandle,
        num_inputs: c_int,
        outputs: *mut *mut TorchTensorHandle,
        num_outputs: *mut c_int,
    ) -> c_int;

    // Tensor operations
    fn torch_tensor_from_blob(
        data: *const f32,
        dims: *const i64,
        ndim: c_int,
        device_type: c_int,
        device_index: c_int,
    ) -> *mut c_void;
    fn torch_tensor_data_ptr(tensor: *mut c_void) -> *const f32;
    fn torch_tensor_numel(tensor: *mut c_void) -> c_ulonglong;
    fn torch_tensor_dim(tensor: *mut c_void) -> c_int;
    fn torch_tensor_size(tensor: *mut c_void, dim: c_int) -> i64;
    fn torch_tensor_to_cpu(tensor: *mut c_void) -> *mut c_void;
    fn torch_tensor_destroy(tensor: *mut c_void);

    // Device management
    fn torch_cuda_is_available() -> c_int;
    fn torch_cuda_device_count() -> c_int;
}

/// Device type constants
const TORCH_DEVICE_CPU: c_int = 0;
const TORCH_DEVICE_CUDA: c_int = 1;

/// Opaque handle for TorchScript model
pub type CudaRsTorchModel = u64;

/// Torch tensor handle (opaque)
#[repr(C)]
pub struct TorchTensorHandle {
    ptr: *mut c_void,
}

/// Output tensor descriptor
#[repr(C)]
pub struct CudaRsTorchTensor {
    pub data: *mut f32,
    pub data_len: c_ulonglong,
    pub shape: *mut i64,
    pub shape_len: c_ulonglong,
}

/// Internal TorchScript model wrapper
struct TorchModel {
    module_ptr: *mut c_void,
    device_type: c_int,
    device_index: c_int,
    input_shapes: Vec<Vec<i64>>, // Cached from first inference
}

unsafe impl Send for TorchModel {}
unsafe impl Sync for TorchModel {}

impl Drop for TorchModel {
    fn drop(&mut self) {
        if !self.module_ptr.is_null() {
            unsafe {
                torch_jit_module_destroy(self.module_ptr);
            }
        }
    }
}

lazy_static::lazy_static! {
    static ref TORCH_MODELS: Mutex<HandleManager<TorchModel>> = Mutex::new(HandleManager::new());
}

/// Check if CUDA is available for PyTorch.
#[no_mangle]
pub extern "C" fn cudars_torch_cuda_available() -> c_int {
    unsafe { torch_cuda_is_available() }
}

/// Get CUDA device count for PyTorch.
#[no_mangle]
pub extern "C" fn cudars_torch_cuda_device_count() -> c_int {
    unsafe { torch_cuda_device_count() }
}

/// Load TorchScript model from file.
#[no_mangle]
pub extern "C" fn cudars_torch_load(
    model_path: *const c_char,
    device_id: c_int,
    out_handle: *mut CudaRsTorchModel,
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

    // Determine device
    let (device_type, device_index) = if device_id < 0 {
        (TORCH_DEVICE_CPU, 0)
    } else {
        let cuda_available = unsafe { torch_cuda_is_available() };
        if cuda_available != 0 {
            let device_count = unsafe { torch_cuda_device_count() };
            if device_id < device_count {
                (TORCH_DEVICE_CUDA, device_id)
            } else {
                return CudaRsResult::ErrorInvalidValue;
            }
        } else {
            // Fall back to CPU if CUDA unavailable
            (TORCH_DEVICE_CPU, 0)
        }
    };

    let path_cstr = match CString::new(path) {
        Ok(s) => s,
        Err(_) => return CudaRsResult::ErrorInvalidValue,
    };

    let module_ptr = unsafe { torch_jit_load(path_cstr.as_ptr(), device_type, device_index) };

    if module_ptr.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let model = TorchModel {
        module_ptr,
        device_type,
        device_index,
        input_shapes: Vec::new(),
    };

    let mut models = TORCH_MODELS.lock().unwrap();
    let id = models.insert(model);

    unsafe {
        *out_handle = id;
    }

    CudaRsResult::Success
}

/// Run inference on TorchScript model.
#[no_mangle]
pub extern "C" fn cudars_torch_run(
    handle: CudaRsTorchModel,
    input_ptr: *const f32,
    input_len: c_ulonglong,
    shape_ptr: *const i64,
    shape_len: c_ulonglong,
    out_tensors: *mut *mut CudaRsTorchTensor,
    out_count: *mut c_ulonglong,
) -> CudaRsResult {
    if input_ptr.is_null() || shape_ptr.is_null() || out_tensors.is_null() || out_count.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let models = TORCH_MODELS.lock().unwrap();
    let model = match models.get(handle) {
        Some(m) => m,
        None => return CudaRsResult::ErrorInvalidHandle,
    };

    let shape = unsafe { std::slice::from_raw_parts(shape_ptr, shape_len as usize) };

    // Verify input size matches shape
    let expected_size: i64 = shape.iter().product();
    if expected_size as u64 != input_len {
        return CudaRsResult::ErrorInvalidValue;
    }

    unsafe {
        // Create input tensor
        let input_tensor = torch_tensor_from_blob(
            input_ptr,
            shape.as_ptr(),
            shape.len() as c_int,
            model.device_type,
            model.device_index,
        );

        if input_tensor.is_null() {
            return CudaRsResult::ErrorOutOfMemory;
        }

        // Prepare input handles
        let input_handle = TorchTensorHandle { ptr: input_tensor };
        let inputs = [input_handle];

        // Run forward pass
        let mut output_handles: *mut TorchTensorHandle = ptr::null_mut();
        let mut num_outputs: c_int = 0;

        let result = torch_jit_module_forward(
            model.module_ptr,
            inputs.as_ptr(),
            1,
            &mut output_handles,
            &mut num_outputs,
        );

        // Clean up input tensor
        torch_tensor_destroy(input_tensor);

        if result == 0 || output_handles.is_null() {
            return CudaRsResult::ErrorUnknown;
        }

        // Process outputs
        let output_slice = std::slice::from_raw_parts(output_handles, num_outputs as usize);
        let mut tensors: Vec<CudaRsTorchTensor> = Vec::with_capacity(num_outputs as usize);

        for output_handle in output_slice {
            let tensor_ptr = output_handle.ptr;

            // Move to CPU if on GPU
            let cpu_tensor = if model.device_type == TORCH_DEVICE_CUDA {
                torch_tensor_to_cpu(tensor_ptr)
            } else {
                tensor_ptr
            };

            // Get tensor info
            let numel = torch_tensor_numel(cpu_tensor) as usize;
            let ndim = torch_tensor_dim(cpu_tensor) as usize;

            // Get shape
            let mut shape_vec: Vec<i64> = Vec::with_capacity(ndim);
            for d in 0..ndim {
                shape_vec.push(torch_tensor_size(cpu_tensor, d as c_int));
            }

            // Copy data
            let data_ptr = torch_tensor_data_ptr(cpu_tensor);
            let data_slice = std::slice::from_raw_parts(data_ptr, numel);
            let data_vec: Vec<f32> = data_slice.to_vec();

            // Clean up tensors
            if cpu_tensor != tensor_ptr {
                torch_tensor_destroy(cpu_tensor);
            }
            torch_tensor_destroy(tensor_ptr);

            // Package output
            let data_len = numel as u64;
            let shape_len = ndim as u64;

            let shape_box = shape_vec.into_boxed_slice();
            let data_box = data_vec.into_boxed_slice();

            let shape_ptr = Box::into_raw(shape_box) as *mut i64;
            let data_ptr = Box::into_raw(data_box) as *mut f32;

            tensors.push(CudaRsTorchTensor {
                data: data_ptr,
                data_len,
                shape: shape_ptr,
                shape_len,
            });
        }

        // Free output handles array (allocated by libtorch)
        libc::free(output_handles as *mut c_void);

        // Return results
        let count = tensors.len() as u64;
        let boxed = tensors.into_boxed_slice();
        let ptr = Box::into_raw(boxed) as *mut CudaRsTorchTensor;

        *out_tensors = ptr;
        *out_count = count;
    }

    CudaRsResult::Success
}

/// Run inference with multiple inputs.
#[no_mangle]
pub extern "C" fn cudars_torch_run_multi(
    handle: CudaRsTorchModel,
    inputs: *const CudaRsTorchInputDesc,
    num_inputs: c_int,
    out_tensors: *mut *mut CudaRsTorchTensor,
    out_count: *mut c_ulonglong,
) -> CudaRsResult {
    if inputs.is_null() || out_tensors.is_null() || out_count.is_null() || num_inputs <= 0 {
        return CudaRsResult::ErrorInvalidValue;
    }

    let models = TORCH_MODELS.lock().unwrap();
    let model = match models.get(handle) {
        Some(m) => m,
        None => return CudaRsResult::ErrorInvalidHandle,
    };

    let input_descs =
        unsafe { std::slice::from_raw_parts(inputs, num_inputs as usize) };

    unsafe {
        // Create input tensors
        let mut input_tensors: Vec<*mut c_void> = Vec::with_capacity(num_inputs as usize);
        let mut input_handles: Vec<TorchTensorHandle> = Vec::with_capacity(num_inputs as usize);

        for desc in input_descs {
            let shape = std::slice::from_raw_parts(desc.shape, desc.shape_len as usize);
            let tensor = torch_tensor_from_blob(
                desc.data,
                shape.as_ptr(),
                shape.len() as c_int,
                model.device_type,
                model.device_index,
            );

            if tensor.is_null() {
                // Clean up already created tensors
                for t in &input_tensors {
                    torch_tensor_destroy(*t);
                }
                return CudaRsResult::ErrorOutOfMemory;
            }

            input_tensors.push(tensor);
            input_handles.push(TorchTensorHandle { ptr: tensor });
        }

        // Run forward pass
        let mut output_handles: *mut TorchTensorHandle = ptr::null_mut();
        let mut num_outputs: c_int = 0;

        let result = torch_jit_module_forward(
            model.module_ptr,
            input_handles.as_ptr(),
            num_inputs,
            &mut output_handles,
            &mut num_outputs,
        );

        // Clean up input tensors
        for t in &input_tensors {
            torch_tensor_destroy(*t);
        }

        if result == 0 || output_handles.is_null() {
            return CudaRsResult::ErrorUnknown;
        }

        // Process outputs (same as single input)
        let output_slice = std::slice::from_raw_parts(output_handles, num_outputs as usize);
        let mut tensors: Vec<CudaRsTorchTensor> = Vec::with_capacity(num_outputs as usize);

        for output_handle in output_slice {
            let tensor_ptr = output_handle.ptr;

            let cpu_tensor = if model.device_type == TORCH_DEVICE_CUDA {
                torch_tensor_to_cpu(tensor_ptr)
            } else {
                tensor_ptr
            };

            let numel = torch_tensor_numel(cpu_tensor) as usize;
            let ndim = torch_tensor_dim(cpu_tensor) as usize;

            let mut shape_vec: Vec<i64> = Vec::with_capacity(ndim);
            for d in 0..ndim {
                shape_vec.push(torch_tensor_size(cpu_tensor, d as c_int));
            }

            let data_ptr = torch_tensor_data_ptr(cpu_tensor);
            let data_slice = std::slice::from_raw_parts(data_ptr, numel);
            let data_vec: Vec<f32> = data_slice.to_vec();

            if cpu_tensor != tensor_ptr {
                torch_tensor_destroy(cpu_tensor);
            }
            torch_tensor_destroy(tensor_ptr);

            let data_len = numel as u64;
            let shape_len = ndim as u64;

            let shape_box = shape_vec.into_boxed_slice();
            let data_box = data_vec.into_boxed_slice();

            tensors.push(CudaRsTorchTensor {
                data: Box::into_raw(data_box) as *mut f32,
                data_len,
                shape: Box::into_raw(shape_box) as *mut i64,
                shape_len,
            });
        }

        libc::free(output_handles as *mut c_void);

        let count = tensors.len() as u64;
        let boxed = tensors.into_boxed_slice();
        *out_tensors = Box::into_raw(boxed) as *mut CudaRsTorchTensor;
        *out_count = count;
    }

    CudaRsResult::Success
}

/// Input descriptor for multi-input forward pass
#[repr(C)]
pub struct CudaRsTorchInputDesc {
    pub data: *const f32,
    pub data_len: c_ulonglong,
    pub shape: *const i64,
    pub shape_len: c_ulonglong,
}

/// Set model to evaluation mode.
#[no_mangle]
pub extern "C" fn cudars_torch_eval(handle: CudaRsTorchModel) -> CudaRsResult {
    let models = TORCH_MODELS.lock().unwrap();
    let model = match models.get(handle) {
        Some(m) => m,
        None => return CudaRsResult::ErrorInvalidHandle,
    };

    // LibTorch should have an eval function
    extern "C" {
        fn torch_jit_module_eval(module: *mut c_void);
    }

    unsafe {
        torch_jit_module_eval(model.module_ptr);
    }

    CudaRsResult::Success
}

/// Destroy TorchScript model.
#[no_mangle]
pub extern "C" fn cudars_torch_destroy(handle: CudaRsTorchModel) -> CudaRsResult {
    let mut models = TORCH_MODELS.lock().unwrap();
    match models.remove(handle) {
        Some(_) => CudaRsResult::Success,
        None => CudaRsResult::ErrorInvalidHandle,
    }
}

/// Free tensors returned by cudars_torch_run.
#[no_mangle]
pub extern "C" fn cudars_torch_free_tensors(
    tensors: *mut CudaRsTorchTensor,
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
        let _ = Box::from_raw(slice as *mut [CudaRsTorchTensor]);
    }

    CudaRsResult::Success
}
