//! CUDA Runtime API FFI exports.

use super::CudaRsResult;
use cuda_runtime::{self, DeviceBuffer, Event, Stream};
use cuda_runtime_sys::{cudaGetDeviceCount, cudaErrorNoDevice, cudaSuccess};
use libc::{c_int, c_void, size_t};
use std::collections::HashMap;
use std::sync::Mutex;

// Global handle storage
lazy_static::lazy_static! {
    pub(crate) static ref STREAMS: Mutex<HandleManager<Stream>> = Mutex::new(HandleManager::new());
    pub(crate) static ref EVENTS: Mutex<HandleManager<Event>> = Mutex::new(HandleManager::new());
    pub(crate) static ref BUFFERS: Mutex<HandleManager<DeviceBuffer<u8>>> = Mutex::new(HandleManager::new());
}

/// Generic handle manager for resource tracking.
pub(crate) struct HandleManager<T> {
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

    pub(crate) fn insert(&mut self, value: T) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.handles.insert(id, value);
        id
    }

    pub(crate) fn get(&self, id: u64) -> Option<&T> {
        self.handles.get(&id)
    }

    pub(crate) fn get_mut(&mut self, id: u64) -> Option<&mut T> {
        self.handles.get_mut(&id)
    }

    pub(crate) fn remove(&mut self, id: u64) -> Option<T> {
        self.handles.remove(&id)
    }
}

/// Opaque handle types for C interop.
pub type CudaRsStream = u64;
pub type CudaRsEvent = u64;
pub type CudaRsBuffer = u64;

// ============================================================================
// Device Management
// ============================================================================

/// Get the number of CUDA devices.
#[no_mangle]
pub extern "C" fn cudars_device_get_count(count: *mut c_int) -> CudaRsResult {
    if count.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }
    unsafe {
        let mut c = 0;
        let code = cudaGetDeviceCount(&mut c);
        if code == cudaSuccess {
            *count = c;
            CudaRsResult::Success
        } else if code == cudaErrorNoDevice {
            *count = 0;
            CudaRsResult::Success
        } else {
            CudaRsResult::ErrorUnknown
        }
    }
}

/// Set the current CUDA device.
#[no_mangle]
pub extern "C" fn cudars_device_set(device: c_int) -> CudaRsResult {
    match cuda_runtime::set_device(device) {
        Ok(()) => CudaRsResult::Success,
        Err(_) => CudaRsResult::ErrorInvalidValue,
    }
}

/// Get the current CUDA device.
#[no_mangle]
pub extern "C" fn cudars_device_get(device: *mut c_int) -> CudaRsResult {
    if device.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }
    match cuda_runtime::get_device() {
        Ok(d) => {
            unsafe { *device = d };
            CudaRsResult::Success
        }
        Err(_) => CudaRsResult::ErrorUnknown,
    }
}

/// Synchronize the current device.
#[no_mangle]
pub extern "C" fn cudars_device_synchronize() -> CudaRsResult {
    match cuda_runtime::device_synchronize() {
        Ok(()) => CudaRsResult::Success,
        Err(_) => CudaRsResult::ErrorUnknown,
    }
}

/// Reset the current device.
#[no_mangle]
pub extern "C" fn cudars_device_reset() -> CudaRsResult {
    match cuda_runtime::device_reset() {
        Ok(()) => CudaRsResult::Success,
        Err(_) => CudaRsResult::ErrorUnknown,
    }
}

// ============================================================================
// Stream Management
// ============================================================================

/// Create a new CUDA stream.
#[no_mangle]
pub extern "C" fn cudars_stream_create(stream: *mut CudaRsStream) -> CudaRsResult {
    if stream.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }
    match Stream::new() {
        Ok(s) => {
            let mut streams = STREAMS.lock().unwrap();
            let id = streams.insert(s);
            unsafe { *stream = id };
            CudaRsResult::Success
        }
        Err(_) => CudaRsResult::ErrorUnknown,
    }
}

/// Destroy a CUDA stream.
#[no_mangle]
pub extern "C" fn cudars_stream_destroy(stream: CudaRsStream) -> CudaRsResult {
    let mut streams = STREAMS.lock().unwrap();
    match streams.remove(stream) {
        Some(_) => CudaRsResult::Success,
        None => CudaRsResult::ErrorInvalidHandle,
    }
}

/// Synchronize a CUDA stream.
#[no_mangle]
pub extern "C" fn cudars_stream_synchronize(stream: CudaRsStream) -> CudaRsResult {
    let streams = STREAMS.lock().unwrap();
    match streams.get(stream) {
        Some(s) => match s.synchronize() {
            Ok(()) => CudaRsResult::Success,
            Err(_) => CudaRsResult::ErrorUnknown,
        },
        None => CudaRsResult::ErrorInvalidHandle,
    }
}

// ============================================================================
// Event Management
// ============================================================================

/// Create a new CUDA event.
#[no_mangle]
pub extern "C" fn cudars_event_create(event: *mut CudaRsEvent) -> CudaRsResult {
    if event.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }
    match Event::new() {
        Ok(e) => {
            let mut events = EVENTS.lock().unwrap();
            let id = events.insert(e);
            unsafe { *event = id };
            CudaRsResult::Success
        }
        Err(_) => CudaRsResult::ErrorUnknown,
    }
}

/// Destroy a CUDA event.
#[no_mangle]
pub extern "C" fn cudars_event_destroy(event: CudaRsEvent) -> CudaRsResult {
    let mut events = EVENTS.lock().unwrap();
    match events.remove(event) {
        Some(_) => CudaRsResult::Success,
        None => CudaRsResult::ErrorInvalidHandle,
    }
}

/// Record an event on a stream.
#[no_mangle]
pub extern "C" fn cudars_event_record(event: CudaRsEvent, stream: CudaRsStream) -> CudaRsResult {
    let events = EVENTS.lock().unwrap();
    let streams = STREAMS.lock().unwrap();
    
    let e = match events.get(event) {
        Some(e) => e,
        None => return CudaRsResult::ErrorInvalidHandle,
    };
    
    let s = match streams.get(stream) {
        Some(s) => s,
        None => return CudaRsResult::ErrorInvalidHandle,
    };
    
    match e.record(s) {
        Ok(()) => CudaRsResult::Success,
        Err(_) => CudaRsResult::ErrorUnknown,
    }
}

/// Synchronize an event.
#[no_mangle]
pub extern "C" fn cudars_event_synchronize(event: CudaRsEvent) -> CudaRsResult {
    let events = EVENTS.lock().unwrap();
    match events.get(event) {
        Some(e) => match e.synchronize() {
            Ok(()) => CudaRsResult::Success,
            Err(_) => CudaRsResult::ErrorUnknown,
        },
        None => CudaRsResult::ErrorInvalidHandle,
    }
}

/// Get elapsed time between two events.
#[no_mangle]
pub extern "C" fn cudars_event_elapsed_time(
    ms: *mut f32,
    start: CudaRsEvent,
    end: CudaRsEvent,
) -> CudaRsResult {
    if ms.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }
    
    let events = EVENTS.lock().unwrap();
    let start_event = match events.get(start) {
        Some(e) => e,
        None => return CudaRsResult::ErrorInvalidHandle,
    };
    let end_event = match events.get(end) {
        Some(e) => e,
        None => return CudaRsResult::ErrorInvalidHandle,
    };
    
    match end_event.elapsed_time(start_event) {
        Ok(t) => {
            unsafe { *ms = t };
            CudaRsResult::Success
        }
        Err(_) => CudaRsResult::ErrorUnknown,
    }
}

// ============================================================================
// Memory Management
// ============================================================================

/// Allocate device memory.
#[no_mangle]
pub extern "C" fn cudars_malloc(buffer: *mut CudaRsBuffer, size: size_t) -> CudaRsResult {
    if buffer.is_null() || size == 0 {
        return CudaRsResult::ErrorInvalidValue;
    }
    match DeviceBuffer::<u8>::new(size) {
        Ok(b) => {
            let mut buffers = BUFFERS.lock().unwrap();
            let id = buffers.insert(b);
            unsafe { *buffer = id };
            CudaRsResult::Success
        }
        Err(_) => CudaRsResult::ErrorOutOfMemory,
    }
}

/// Free device memory.
#[no_mangle]
pub extern "C" fn cudars_free(buffer: CudaRsBuffer) -> CudaRsResult {
    let mut buffers = BUFFERS.lock().unwrap();
    match buffers.remove(buffer) {
        Some(_) => CudaRsResult::Success,
        None => CudaRsResult::ErrorInvalidHandle,
    }
}

/// Copy memory from host to device.
#[no_mangle]
pub extern "C" fn cudars_memcpy_htod(
    buffer: CudaRsBuffer,
    src: *const c_void,
    size: size_t,
) -> CudaRsResult {
    if src.is_null() || size == 0 {
        return CudaRsResult::ErrorInvalidValue;
    }
    
    let mut buffers = BUFFERS.lock().unwrap();
    match buffers.get_mut(buffer) {
        Some(b) => {
            let data = unsafe { std::slice::from_raw_parts(src as *const u8, size) };
            match b.copy_from_host(data) {
                Ok(()) => CudaRsResult::Success,
                Err(_) => CudaRsResult::ErrorUnknown,
            }
        }
        None => CudaRsResult::ErrorInvalidHandle,
    }
}

/// Copy memory from device to host.
#[no_mangle]
pub extern "C" fn cudars_memcpy_dtoh(
    dst: *mut c_void,
    buffer: CudaRsBuffer,
    size: size_t,
) -> CudaRsResult {
    if dst.is_null() || size == 0 {
        return CudaRsResult::ErrorInvalidValue;
    }
    
    let buffers = BUFFERS.lock().unwrap();
    match buffers.get(buffer) {
        Some(b) => {
            let data = unsafe { std::slice::from_raw_parts_mut(dst as *mut u8, size) };
            match b.copy_to_host(data) {
                Ok(()) => CudaRsResult::Success,
                Err(_) => CudaRsResult::ErrorUnknown,
            }
        }
        None => CudaRsResult::ErrorInvalidHandle,
    }
}

/// Set device memory to a value.
#[no_mangle]
pub extern "C" fn cudars_memset(buffer: CudaRsBuffer, value: c_int) -> CudaRsResult {
    let mut buffers = BUFFERS.lock().unwrap();
    match buffers.get_mut(buffer) {
        Some(b) => match b.memset(value) {
            Ok(()) => CudaRsResult::Success,
            Err(_) => CudaRsResult::ErrorUnknown,
        },
        None => CudaRsResult::ErrorInvalidHandle,
    }
}
