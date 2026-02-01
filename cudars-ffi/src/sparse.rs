//! cuSPARSE FFI exports.

use super::CudaRsResult;
use cusparse::SparseHandle;
use libc::c_int;
use std::collections::HashMap;
use std::sync::Mutex;

lazy_static::lazy_static! {
    static ref SPARSE_HANDLES: Mutex<HandleManager<SparseHandle>> = Mutex::new(HandleManager::new());
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

pub type CudaRsSparseHandle = u64;

/// Create a cuSPARSE handle.
#[no_mangle]
pub extern "C" fn cudars_sparse_create(handle: *mut CudaRsSparseHandle) -> CudaRsResult {
    if handle.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }
    match SparseHandle::new() {
        Ok(h) => {
            let mut handles = SPARSE_HANDLES.lock().unwrap();
            let id = handles.insert(h);
            unsafe { *handle = id };
            CudaRsResult::Success
        }
        Err(_) => CudaRsResult::ErrorUnknown,
    }
}

/// Destroy a cuSPARSE handle.
#[no_mangle]
pub extern "C" fn cudars_sparse_destroy(handle: CudaRsSparseHandle) -> CudaRsResult {
    let mut handles = SPARSE_HANDLES.lock().unwrap();
    match handles.remove(handle) {
        Some(_) => CudaRsResult::Success,
        None => CudaRsResult::ErrorInvalidHandle,
    }
}

/// Get cuSPARSE version.
#[no_mangle]
pub extern "C" fn cudars_sparse_get_version(
    handle: CudaRsSparseHandle,
    version: *mut c_int,
) -> CudaRsResult {
    if version.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }
    
    let handles = SPARSE_HANDLES.lock().unwrap();
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
