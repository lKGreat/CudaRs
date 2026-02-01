//! cuBLAS FFI exports.

use super::CudaRsResult;
use super::runtime::{CudaRsStream, STREAMS};
use cublas::CublasHandle;
use libc::c_int;
use std::collections::HashMap;
use std::sync::Mutex;

lazy_static::lazy_static! {
    static ref CUBLAS_HANDLES: Mutex<HandleManager<CublasHandle>> = Mutex::new(HandleManager::new());
}

struct HandleManager<T> {
    handles: HashMap<u64, T>,
    next_id: u64,
}

impl<T> HandleManager<T> {
    fn new() -> Self {
        Self {
            handles: HashMap::new(),
            next_id: 1,
        }
    }

    fn insert(&mut self, value: T) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.handles.insert(id, value);
        id
    }

    fn get(&self, id: u64) -> Option<&T> {
        self.handles.get(&id)
    }

    fn remove(&mut self, id: u64) -> Option<T> {
        self.handles.remove(&id)
    }
}

pub type CudaRsCublasHandle = u64;

/// Create a cuBLAS handle.
#[no_mangle]
pub extern "C" fn cudars_cublas_create(handle: *mut CudaRsCublasHandle) -> CudaRsResult {
    if handle.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }
    match CublasHandle::new() {
        Ok(h) => {
            let mut handles = CUBLAS_HANDLES.lock().unwrap();
            let id = handles.insert(h);
            unsafe { *handle = id };
            CudaRsResult::Success
        }
        Err(_) => CudaRsResult::ErrorUnknown,
    }
}

/// Destroy a cuBLAS handle.
#[no_mangle]
pub extern "C" fn cudars_cublas_destroy(handle: CudaRsCublasHandle) -> CudaRsResult {
    let mut handles = CUBLAS_HANDLES.lock().unwrap();
    match handles.remove(handle) {
        Some(_) => CudaRsResult::Success,
        None => CudaRsResult::ErrorInvalidHandle,
    }
}

/// Set the stream for a cuBLAS handle.
#[no_mangle]
pub extern "C" fn cudars_cublas_set_stream(
    handle: CudaRsCublasHandle,
    stream: CudaRsStream,
) -> CudaRsResult {
    let handles = CUBLAS_HANDLES.lock().unwrap();
    let streams = STREAMS.lock().unwrap();
    
    let h = match handles.get(handle) {
        Some(h) => h,
        None => return CudaRsResult::ErrorInvalidHandle,
    };
    
    let s = match streams.get(stream) {
        Some(s) => s,
        None => return CudaRsResult::ErrorInvalidHandle,
    };
    
    match h.set_stream(s) {
        Ok(()) => CudaRsResult::Success,
        Err(_) => CudaRsResult::ErrorUnknown,
    }
}

/// Get cuBLAS version.
#[no_mangle]
pub extern "C" fn cudars_cublas_get_version(
    handle: CudaRsCublasHandle,
    version: *mut c_int,
) -> CudaRsResult {
    if version.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }
    
    let handles = CUBLAS_HANDLES.lock().unwrap();
    match handles.get(handle) {
        Some(h) => match h.version() {
            Ok(v) => {
                unsafe { *version = v };
                CudaRsResult::Success
            }
            Err(_) => CudaRsResult::ErrorUnknown,
        },
        None => CudaRsResult::ErrorInvalidHandle,
    }
}

// Note: Full BLAS operations would require device buffer integration
// which is complex due to the type-safety requirements.
// A complete implementation would include sgemm, dgemm, etc.
