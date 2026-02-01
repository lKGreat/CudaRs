//! Safe Rust wrapper for CUDA Driver API.

use cuda_driver_sys::*;
use std::ffi::{CStr, CString};
use std::ptr;
use thiserror::Error;

#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
#[error("CUDA Driver Error: {0} ({1})")]
pub struct CuError(pub i32, pub &'static str);

impl CuError {
    pub fn from_code(code: CUresult) -> Self {
        let msg = unsafe {
            let mut ptr = ptr::null();
            if cuGetErrorString(code, &mut ptr) == CUDA_SUCCESS && !ptr.is_null() {
                CStr::from_ptr(ptr).to_str().unwrap_or("Unknown error")
            } else {
                "Unknown error"
            }
        };
        CuError(code, msg)
    }
}

pub type Result<T> = std::result::Result<T, CuError>;

#[inline]
fn check(code: CUresult) -> Result<()> {
    if code == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(CuError::from_code(code))
    }
}

/// Initialize the CUDA driver.
pub fn init() -> Result<()> {
    unsafe { check(cuInit(0)) }
}

/// Get CUDA driver version.
pub fn driver_version() -> Result<i32> {
    let mut version = 0;
    unsafe { check(cuDriverGetVersion(&mut version))? };
    Ok(version)
}

/// Get the number of CUDA devices.
pub fn device_count() -> Result<i32> {
    let mut count = 0;
    unsafe { check(cuDeviceGetCount(&mut count))? };
    Ok(count)
}

/// CUDA Device wrapper.
#[derive(Debug, Clone, Copy)]
pub struct Device(CUdevice);

impl Device {
    /// Get a device by index.
    pub fn get(ordinal: i32) -> Result<Self> {
        let mut device = 0;
        unsafe { check(cuDeviceGet(&mut device, ordinal))? };
        Ok(Device(device))
    }

    /// Get the device name.
    pub fn name(&self) -> Result<String> {
        let mut name = [0i8; 256];
        unsafe { check(cuDeviceGetName(name.as_mut_ptr(), 256, self.0))? };
        Ok(unsafe { CStr::from_ptr(name.as_ptr()) }
            .to_string_lossy()
            .into_owned())
    }

    /// Get total memory in bytes.
    pub fn total_memory(&self) -> Result<usize> {
        let mut bytes = 0;
        unsafe { check(cuDeviceTotalMem(&mut bytes, self.0))? };
        Ok(bytes)
    }

    /// Get compute capability.
    pub fn compute_capability(&self) -> Result<(i32, i32)> {
        let mut major = 0;
        let mut minor = 0;
        unsafe {
            check(cuDeviceGetAttribute(&mut major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, self.0))?;
            check(cuDeviceGetAttribute(&mut minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, self.0))?;
        }
        Ok((major, minor))
    }

    /// Get the raw device handle.
    pub fn as_raw(&self) -> CUdevice {
        self.0
    }
}

/// CUDA Context wrapper with automatic resource management.
pub struct Context {
    handle: CUcontext,
}

impl Context {
    /// Create a new context on the given device.
    pub fn new(device: Device) -> Result<Self> {
        let mut handle = ptr::null_mut();
        unsafe { check(cuCtxCreate(&mut handle, 0, device.0))? };
        Ok(Self { handle })
    }

    /// Create a new context with flags.
    pub fn with_flags(device: Device, flags: u32) -> Result<Self> {
        let mut handle = ptr::null_mut();
        unsafe { check(cuCtxCreate(&mut handle, flags, device.0))? };
        Ok(Self { handle })
    }

    /// Push this context to the current thread's context stack.
    pub fn push(&self) -> Result<()> {
        unsafe { check(cuCtxPushCurrent(self.handle)) }
    }

    /// Pop the current context from the thread's context stack.
    pub fn pop() -> Result<CUcontext> {
        let mut ctx = ptr::null_mut();
        unsafe { check(cuCtxPopCurrent(&mut ctx))? };
        Ok(ctx)
    }

    /// Synchronize the context.
    pub fn synchronize(&self) -> Result<()> {
        unsafe { check(cuCtxSynchronize()) }
    }

    /// Get the raw context handle.
    pub fn as_raw(&self) -> CUcontext {
        self.handle
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { cuCtxDestroy(self.handle) };
        }
    }
}

unsafe impl Send for Context {}
unsafe impl Sync for Context {}

/// CUDA Module wrapper with automatic resource management.
pub struct Module {
    handle: CUmodule,
}

impl Module {
    /// Load a module from a file.
    pub fn load(filename: &str) -> Result<Self> {
        let filename = CString::new(filename).expect("Invalid filename");
        let mut handle = ptr::null_mut();
        unsafe { check(cuModuleLoad(&mut handle, filename.as_ptr()))? };
        Ok(Self { handle })
    }

    /// Load a module from PTX data.
    pub fn load_data(data: &[u8]) -> Result<Self> {
        let mut handle = ptr::null_mut();
        unsafe { check(cuModuleLoadData(&mut handle, data.as_ptr() as *const _))? };
        Ok(Self { handle })
    }

    /// Get a function from the module.
    pub fn get_function(&self, name: &str) -> Result<Function> {
        let name = CString::new(name).expect("Invalid function name");
        let mut func = ptr::null_mut();
        unsafe { check(cuModuleGetFunction(&mut func, self.handle, name.as_ptr()))? };
        Ok(Function { handle: func })
    }

    /// Get a global variable from the module.
    pub fn get_global(&self, name: &str) -> Result<(CUdeviceptr, usize)> {
        let name = CString::new(name).expect("Invalid global name");
        let mut dptr = 0;
        let mut size = 0;
        unsafe { check(cuModuleGetGlobal(&mut dptr, &mut size, self.handle, name.as_ptr()))? };
        Ok((dptr, size))
    }

    /// Get the raw module handle.
    pub fn as_raw(&self) -> CUmodule {
        self.handle
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { cuModuleUnload(self.handle) };
        }
    }
}

unsafe impl Send for Module {}
unsafe impl Sync for Module {}

/// CUDA Function wrapper.
pub struct Function {
    handle: CUfunction,
}

impl Function {
    /// Launch the kernel.
    pub fn launch(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        stream: CUstream,
        params: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        unsafe {
            check(cuLaunchKernel(
                self.handle,
                grid.0, grid.1, grid.2,
                block.0, block.1, block.2,
                shared_mem,
                stream,
                params.as_mut_ptr(),
                ptr::null_mut(),
            ))
        }
    }

    /// Get the raw function handle.
    pub fn as_raw(&self) -> CUfunction {
        self.handle
    }
}

unsafe impl Send for Function {}
unsafe impl Sync for Function {}

/// CUDA Stream wrapper with automatic resource management.
pub struct Stream {
    handle: CUstream,
    owned: bool,
}

impl Stream {
    /// Create a new stream.
    pub fn new() -> Result<Self> {
        let mut handle = ptr::null_mut();
        unsafe { check(cuStreamCreate(&mut handle, 0))? };
        Ok(Self { handle, owned: true })
    }

    /// Create a new stream with flags.
    pub fn with_flags(flags: u32) -> Result<Self> {
        let mut handle = ptr::null_mut();
        unsafe { check(cuStreamCreate(&mut handle, flags))? };
        Ok(Self { handle, owned: true })
    }

    /// Get the default (null) stream.
    pub fn default_stream() -> Self {
        Self {
            handle: ptr::null_mut(),
            owned: false,
        }
    }

    /// Synchronize the stream.
    pub fn synchronize(&self) -> Result<()> {
        unsafe { check(cuStreamSynchronize(self.handle)) }
    }

    /// Get the raw stream handle.
    pub fn as_raw(&self) -> CUstream {
        self.handle
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if self.owned && !self.handle.is_null() {
            unsafe { cuStreamDestroy(self.handle) };
        }
    }
}

unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

/// Allocate device memory.
pub fn mem_alloc(size: usize) -> Result<CUdeviceptr> {
    let mut dptr = 0;
    unsafe { check(cuMemAlloc(&mut dptr, size))? };
    Ok(dptr)
}

/// Free device memory.
pub fn mem_free(dptr: CUdeviceptr) -> Result<()> {
    unsafe { check(cuMemFree(dptr)) }
}

/// Copy memory from host to device.
pub fn memcpy_htod(dst: CUdeviceptr, src: &[u8]) -> Result<()> {
    unsafe { check(cuMemcpyHtoD(dst, src.as_ptr() as *const _, src.len())) }
}

/// Copy memory from device to host.
pub fn memcpy_dtoh(dst: &mut [u8], src: CUdeviceptr) -> Result<()> {
    unsafe { check(cuMemcpyDtoH(dst.as_mut_ptr() as *mut _, src, dst.len())) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        // This will fail if no CUDA driver is installed, which is expected
        let _ = init();
    }
}
