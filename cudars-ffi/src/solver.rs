//! cuSOLVER FFI exports.

use super::CudaRsResult;
use cusolver::DnHandle;
use std::collections::HashMap;
use std::sync::Mutex;

lazy_static::lazy_static! {
    static ref SOLVER_HANDLES: Mutex<HandleManager<DnHandle>> = Mutex::new(HandleManager::new());
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

pub type CudaRsSolverHandle = u64;

/// Create a cuSOLVER dense handle.
#[no_mangle]
pub extern "C" fn cudars_solver_dn_create(handle: *mut CudaRsSolverHandle) -> CudaRsResult {
    if handle.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }
    match DnHandle::new() {
        Ok(h) => {
            let mut handles = SOLVER_HANDLES.lock().unwrap();
            let id = handles.insert(h);
            unsafe { *handle = id };
            CudaRsResult::Success
        }
        Err(_) => CudaRsResult::ErrorUnknown,
    }
}

/// Destroy a cuSOLVER dense handle.
#[no_mangle]
pub extern "C" fn cudars_solver_dn_destroy(handle: CudaRsSolverHandle) -> CudaRsResult {
    let mut handles = SOLVER_HANDLES.lock().unwrap();
    match handles.remove(handle) {
        Some(_) => CudaRsResult::Success,
        None => CudaRsResult::ErrorInvalidHandle,
    }
}
