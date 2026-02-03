//! Safe Rust wrapper for CUDA Runtime API.

use cuda_runtime_sys::*;
use std::ffi::CStr;
use std::ptr;
use thiserror::Error;

#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
#[error("CUDA Runtime Error: {0} ({1})")]
pub struct CudaError(pub i32, pub &'static str);

impl CudaError {
    pub fn from_code(code: cudaError_t) -> Self {
        let msg = unsafe {
            let ptr = cudaGetErrorString(code);
            if ptr.is_null() {
                "Unknown error"
            } else {
                CStr::from_ptr(ptr).to_str().unwrap_or("Unknown error")
            }
        };
        CudaError(code, msg)
    }
}

pub type Result<T> = std::result::Result<T, CudaError>;

#[inline]
fn check(code: cudaError_t) -> Result<()> {
    if code == cudaSuccess {
        Ok(())
    } else {
        Err(CudaError::from_code(code))
    }
}

/// CUDA Stream wrapper with automatic resource management.
pub struct Stream {
    handle: cudaStream_t,
    owned: bool,
}

impl Stream {
    /// Create a new CUDA stream.
    pub fn new() -> Result<Self> {
        let mut handle = ptr::null_mut();
        unsafe { check(cudaStreamCreate(&mut handle))? };
        Ok(Self { handle, owned: true })
    }

    /// Create a new CUDA stream with flags.
    pub fn with_flags(flags: u32) -> Result<Self> {
        let mut handle = ptr::null_mut();
        unsafe { check(cudaStreamCreateWithFlags(&mut handle, flags))? };
        Ok(Self { handle, owned: true })
    }

    /// Get the default stream (non-owning).
    pub fn default_stream() -> Self {
        Self {
            handle: ptr::null_mut(),
            owned: false,
        }
    }

    /// Synchronize the stream.
    pub fn synchronize(&self) -> Result<()> {
        unsafe { check(cudaStreamSynchronize(self.handle)) }
    }

    /// Get the raw stream handle.
    pub fn as_raw(&self) -> cudaStream_t {
        self.handle
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if self.owned && !self.handle.is_null() {
            unsafe { cudaStreamDestroy(self.handle) };
        }
    }
}

unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

/// CUDA Event wrapper with automatic resource management.
pub struct Event {
    handle: cudaEvent_t,
}

impl Event {
    /// Create a new CUDA event.
    pub fn new() -> Result<Self> {
        let mut handle = ptr::null_mut();
        unsafe { check(cudaEventCreate(&mut handle))? };
        Ok(Self { handle })
    }

    /// Create a new CUDA event with flags.
    pub fn with_flags(flags: u32) -> Result<Self> {
        let mut handle = ptr::null_mut();
        unsafe { check(cudaEventCreateWithFlags(&mut handle, flags))? };
        Ok(Self { handle })
    }

    /// Record the event on a stream.
    pub fn record(&self, stream: &Stream) -> Result<()> {
        unsafe { check(cudaEventRecord(self.handle, stream.as_raw())) }
    }

    /// Get the raw event handle.
    pub fn as_raw(&self) -> cudaEvent_t {
        self.handle
    }

    /// Synchronize the event.
    pub fn synchronize(&self) -> Result<()> {
        unsafe { check(cudaEventSynchronize(self.handle)) }
    }

    /// Get elapsed time between two events in milliseconds.
    pub fn elapsed_time(&self, start: &Event) -> Result<f32> {
        let mut ms = 0.0f32;
        unsafe { check(cudaEventElapsedTime(&mut ms, start.handle, self.handle))? };
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { cudaEventDestroy(self.handle) };
        }
    }
}

unsafe impl Send for Event {}
unsafe impl Sync for Event {}

/// Device memory buffer with automatic deallocation.
pub struct DeviceBuffer<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> DeviceBuffer<T> {
    /// Allocate device memory for `len` elements.
    pub fn new(len: usize) -> Result<Self> {
        let size = len * std::mem::size_of::<T>();
        let mut ptr = ptr::null_mut();
        unsafe { check(cudaMalloc(&mut ptr, size))? };
        Ok(Self { ptr: ptr as *mut T, len })
    }

    /// Get the length of the buffer in elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the size of the buffer in bytes.
    pub fn size(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// Get the raw device pointer.
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Get the raw mutable device pointer.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Copy data from host to device.
    pub fn copy_from_host(&mut self, data: &[T]) -> Result<()> {
        assert!(data.len() <= self.len, "Source data exceeds buffer size");
        let size = data.len() * std::mem::size_of::<T>();
        unsafe {
            check(cudaMemcpy(
                self.ptr as *mut _,
                data.as_ptr() as *const _,
                size,
                cudaMemcpyHostToDevice,
            ))
        }
    }

    /// Copy data from device to host.
    pub fn copy_to_host(&self, data: &mut [T]) -> Result<()> {
        assert!(data.len() <= self.len, "Destination buffer too small");
        let size = data.len() * std::mem::size_of::<T>();
        unsafe {
            check(cudaMemcpy(
                data.as_mut_ptr() as *mut _,
                self.ptr as *const _,
                size,
                cudaMemcpyDeviceToHost,
            ))
        }
    }

    /// Async copy from host to device.
    pub fn copy_from_host_async(&mut self, data: &[T], stream: &Stream) -> Result<()> {
        assert!(data.len() <= self.len, "Source data exceeds buffer size");
        let size = data.len() * std::mem::size_of::<T>();
        unsafe {
            check(cudaMemcpyAsync(
                self.ptr as *mut _,
                data.as_ptr() as *const _,
                size,
                cudaMemcpyHostToDevice,
                stream.as_raw(),
            ))
        }
    }

    /// Async copy from device to host.
    pub fn copy_to_host_async(&self, data: &mut [T], stream: &Stream) -> Result<()> {
        assert!(data.len() <= self.len, "Destination buffer too small");
        let size = data.len() * std::mem::size_of::<T>();
        unsafe {
            check(cudaMemcpyAsync(
                data.as_mut_ptr() as *mut _,
                self.ptr as *const _,
                size,
                cudaMemcpyDeviceToHost,
                stream.as_raw(),
            ))
        }
    }

    /// Set memory to a value.
    pub fn memset(&mut self, value: i32) -> Result<()> {
        unsafe { check(cudaMemset(self.ptr as *mut _, value, self.size())) }
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { cudaFree(self.ptr as *mut _) };
        }
    }
}

unsafe impl<T: Send> Send for DeviceBuffer<T> {}
unsafe impl<T: Sync> Sync for DeviceBuffer<T> {}

/// Get the number of CUDA devices.
pub fn device_count() -> Result<i32> {
    let mut count = 0;
    unsafe { check(cudaGetDeviceCount(&mut count))? };
    Ok(count)
}

/// Set the current CUDA device.
pub fn set_device(device: i32) -> Result<()> {
    unsafe { check(cudaSetDevice(device)) }
}

/// Get the current CUDA device.
pub fn get_device() -> Result<i32> {
    let mut device = 0;
    unsafe { check(cudaGetDevice(&mut device))? };
    Ok(device)
}

/// Synchronize the current device.
pub fn device_synchronize() -> Result<()> {
    unsafe { check(cudaDeviceSynchronize()) }
}

/// Reset the current device.
pub fn device_reset() -> Result<()> {
    unsafe { check(cudaDeviceReset()) }
}

/// Get the last CUDA error without resetting it.
pub fn peek_last_error() -> Result<()> {
    unsafe { check(cudaPeekAtLastError()) }
}

/// Get the last CUDA error and reset it.
pub fn get_last_error() -> Result<()> {
    unsafe { check(cudaGetLastError()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_count() {
        // This will succeed even if no CUDA device is available
        let _ = device_count();
    }
}
