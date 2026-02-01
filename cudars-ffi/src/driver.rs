//! CUDA Driver API FFI exports.

use super::CudaRsResult;
use cuda_driver::{self, Context, Device, Module, Stream};
use libc::{c_int, size_t};
use std::collections::HashMap;
use std::sync::Mutex;

lazy_static::lazy_static! {
    static ref CONTEXTS: Mutex<HandleManager<Context>> = Mutex::new(HandleManager::new());
    static ref MODULES: Mutex<HandleManager<Module>> = Mutex::new(HandleManager::new());
    static ref DRIVER_STREAMS: Mutex<HandleManager<Stream>> = Mutex::new(HandleManager::new());
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

pub type CudaRsContext = u64;
pub type CudaRsModule = u64;
pub type CudaRsDriverStream = u64;

/// Initialize the CUDA driver.
#[no_mangle]
pub extern "C" fn cudars_driver_init() -> CudaRsResult {
    match cuda_driver::init() {
        Ok(()) => CudaRsResult::Success,
        Err(_) => CudaRsResult::ErrorNotInitialized,
    }
}

/// Get CUDA driver version.
#[no_mangle]
pub extern "C" fn cudars_driver_get_version(version: *mut c_int) -> CudaRsResult {
    if version.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }
    match cuda_driver::driver_version() {
        Ok(v) => {
            unsafe { *version = v };
            CudaRsResult::Success
        }
        Err(_) => CudaRsResult::ErrorUnknown,
    }
}

/// Get CUDA device count (driver API).
#[no_mangle]
pub extern "C" fn cudars_driver_device_get_count(count: *mut c_int) -> CudaRsResult {
    if count.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }
    match cuda_driver::device_count() {
        Ok(c) => {
            unsafe { *count = c };
            CudaRsResult::Success
        }
        Err(_) => CudaRsResult::ErrorUnknown,
    }
}

/// Create a CUDA context.
#[no_mangle]
pub extern "C" fn cudars_context_create(ctx: *mut CudaRsContext, device: c_int) -> CudaRsResult {
    if ctx.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }
    
    let dev = match Device::get(device) {
        Ok(d) => d,
        Err(_) => return CudaRsResult::ErrorInvalidValue,
    };
    
    match Context::new(dev) {
        Ok(c) => {
            let mut contexts = CONTEXTS.lock().unwrap();
            let id = contexts.insert(c);
            unsafe { *ctx = id };
            CudaRsResult::Success
        }
        Err(_) => CudaRsResult::ErrorUnknown,
    }
}

/// Destroy a CUDA context.
#[no_mangle]
pub extern "C" fn cudars_context_destroy(ctx: CudaRsContext) -> CudaRsResult {
    let mut contexts = CONTEXTS.lock().unwrap();
    match contexts.remove(ctx) {
        Some(_) => CudaRsResult::Success,
        None => CudaRsResult::ErrorInvalidHandle,
    }
}

/// Synchronize a CUDA context.
#[no_mangle]
pub extern "C" fn cudars_context_synchronize(ctx: CudaRsContext) -> CudaRsResult {
    let contexts = CONTEXTS.lock().unwrap();
    match contexts.get(ctx) {
        Some(c) => match c.synchronize() {
            Ok(()) => CudaRsResult::Success,
            Err(_) => CudaRsResult::ErrorUnknown,
        },
        None => CudaRsResult::ErrorInvalidHandle,
    }
}

/// Load a module from PTX data.
#[no_mangle]
pub extern "C" fn cudars_module_load_data(
    module: *mut CudaRsModule,
    data: *const u8,
    size: size_t,
) -> CudaRsResult {
    if module.is_null() || data.is_null() || size == 0 {
        return CudaRsResult::ErrorInvalidValue;
    }
    
    let ptx = unsafe { std::slice::from_raw_parts(data, size) };
    match Module::load_data(ptx) {
        Ok(m) => {
            let mut modules = MODULES.lock().unwrap();
            let id = modules.insert(m);
            unsafe { *module = id };
            CudaRsResult::Success
        }
        Err(_) => CudaRsResult::ErrorUnknown,
    }
}

/// Unload a module.
#[no_mangle]
pub extern "C" fn cudars_module_unload(module: CudaRsModule) -> CudaRsResult {
    let mut modules = MODULES.lock().unwrap();
    match modules.remove(module) {
        Some(_) => CudaRsResult::Success,
        None => CudaRsResult::ErrorInvalidHandle,
    }
}

/// Allocate device memory (driver API).
#[no_mangle]
pub extern "C" fn cudars_driver_mem_alloc(dptr: *mut u64, size: size_t) -> CudaRsResult {
    if dptr.is_null() || size == 0 {
        return CudaRsResult::ErrorInvalidValue;
    }
    match cuda_driver::mem_alloc(size) {
        Ok(ptr) => {
            unsafe { *dptr = ptr as u64 };
            CudaRsResult::Success
        }
        Err(_) => CudaRsResult::ErrorOutOfMemory,
    }
}

/// Free device memory (driver API).
#[no_mangle]
pub extern "C" fn cudars_driver_mem_free(dptr: u64) -> CudaRsResult {
    let ptr = match u32::try_from(dptr) {
        Ok(p) => p,
        Err(_) => return CudaRsResult::ErrorInvalidValue,
    };
    match cuda_driver::mem_free(ptr) {
        Ok(()) => CudaRsResult::Success,
        Err(_) => CudaRsResult::ErrorUnknown,
    }
}
