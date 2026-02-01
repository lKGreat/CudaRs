//! cuDNN FFI exports.

use super::CudaRsResult;
use cudnn::Handle;
use libc::size_t;
use std::collections::HashMap;
use std::sync::Mutex;

lazy_static::lazy_static! {
    static ref CUDNN_HANDLES: Mutex<HandleManager<Handle>> = Mutex::new(HandleManager::new());
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

    fn remove(&mut self, id: u64) -> Option<T> {
        self.handles.remove(&id)
    }
}

pub type CudaRsCudnnHandle = u64;

/// Create a cuDNN handle.
#[no_mangle]
pub extern "C" fn cudars_cudnn_create(handle: *mut CudaRsCudnnHandle) -> CudaRsResult {
    if handle.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }
    match Handle::new() {
        Ok(h) => {
            let mut handles = CUDNN_HANDLES.lock().unwrap();
            let id = handles.insert(h);
            unsafe { *handle = id };
            CudaRsResult::Success
        }
        Err(_) => CudaRsResult::ErrorUnknown,
    }
}

/// Destroy a cuDNN handle.
#[no_mangle]
pub extern "C" fn cudars_cudnn_destroy(handle: CudaRsCudnnHandle) -> CudaRsResult {
    let mut handles = CUDNN_HANDLES.lock().unwrap();
    match handles.remove(handle) {
        Some(_) => CudaRsResult::Success,
        None => CudaRsResult::ErrorInvalidHandle,
    }
}

/// Get cuDNN version.
#[no_mangle]
pub extern "C" fn cudars_cudnn_get_version() -> size_t {
    Handle::version()
}
