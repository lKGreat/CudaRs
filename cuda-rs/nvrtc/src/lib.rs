//! Safe Rust wrapper for NVRTC (NVIDIA Runtime Compilation).

use nvrtc_sys::*;
use std::ffi::{CStr, CString};
use std::ptr;
use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq, Eq)]
#[error("NVRTC Error: {0} ({1})")]
pub struct NvrtcError(pub i32, pub String);

impl NvrtcError {
    pub fn from_code(code: nvrtcResult) -> Self {
        let msg = unsafe {
            let ptr = nvrtcGetErrorString(code);
            if ptr.is_null() {
                "Unknown error".to_string()
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        NvrtcError(code, msg)
    }
}

pub type Result<T> = std::result::Result<T, NvrtcError>;

#[inline]
fn check(code: nvrtcResult) -> Result<()> {
    if code == NVRTC_SUCCESS {
        Ok(())
    } else {
        Err(NvrtcError::from_code(code))
    }
}

/// Get NVRTC version.
pub fn version() -> Result<(i32, i32)> {
    let mut major = 0;
    let mut minor = 0;
    unsafe { check(nvrtcVersion(&mut major, &mut minor))? };
    Ok((major, minor))
}

/// Get supported compute architectures.
pub fn get_supported_archs() -> Result<Vec<i32>> {
    let mut num_archs = 0;
    unsafe { check(nvrtcGetNumSupportedArchs(&mut num_archs))? };
    
    let mut archs = vec![0; num_archs as usize];
    unsafe { check(nvrtcGetSupportedArchs(archs.as_mut_ptr()))? };
    
    Ok(archs)
}

/// NVRTC Program wrapper with automatic resource management.
pub struct Program {
    handle: nvrtcProgram,
}

impl Program {
    /// Create a new program from source code.
    pub fn new(src: &str, name: &str) -> Result<Self> {
        let src = CString::new(src).expect("Invalid source");
        let name = CString::new(name).expect("Invalid name");
        let mut handle = ptr::null_mut();
        unsafe {
            check(nvrtcCreateProgram(
                &mut handle,
                src.as_ptr(),
                name.as_ptr(),
                0,
                ptr::null(),
                ptr::null(),
            ))?;
        }
        Ok(Self { handle })
    }

    /// Create a new program with headers.
    pub fn with_headers(
        src: &str,
        name: &str,
        headers: &[(&str, &str)],
    ) -> Result<Self> {
        let src = CString::new(src).expect("Invalid source");
        let name = CString::new(name).expect("Invalid name");
        
        let header_srcs: Vec<CString> = headers
            .iter()
            .map(|(src, _)| CString::new(*src).expect("Invalid header source"))
            .collect();
        let header_names: Vec<CString> = headers
            .iter()
            .map(|(_, name)| CString::new(*name).expect("Invalid header name"))
            .collect();
        
        let header_src_ptrs: Vec<*const i8> = header_srcs.iter().map(|s| s.as_ptr()).collect();
        let header_name_ptrs: Vec<*const i8> = header_names.iter().map(|s| s.as_ptr()).collect();
        
        let mut handle = ptr::null_mut();
        unsafe {
            check(nvrtcCreateProgram(
                &mut handle,
                src.as_ptr(),
                name.as_ptr(),
                headers.len() as i32,
                header_src_ptrs.as_ptr(),
                header_name_ptrs.as_ptr(),
            ))?;
        }
        Ok(Self { handle })
    }

    /// Compile the program.
    pub fn compile(&self, options: &[&str]) -> Result<()> {
        let options: Vec<CString> = options
            .iter()
            .map(|s| CString::new(*s).expect("Invalid option"))
            .collect();
        let option_ptrs: Vec<*const i8> = options.iter().map(|s| s.as_ptr()).collect();
        
        unsafe {
            check(nvrtcCompileProgram(
                self.handle,
                options.len() as i32,
                option_ptrs.as_ptr(),
            ))
        }
    }

    /// Get the compilation log.
    pub fn get_log(&self) -> Result<String> {
        let mut log_size = 0;
        unsafe { check(nvrtcGetProgramLogSize(self.handle, &mut log_size))? };
        
        let mut log = vec![0i8; log_size];
        unsafe { check(nvrtcGetProgramLog(self.handle, log.as_mut_ptr()))? };
        
        Ok(unsafe { CStr::from_ptr(log.as_ptr()) }
            .to_string_lossy()
            .into_owned())
    }

    /// Get the PTX output.
    pub fn get_ptx(&self) -> Result<Vec<u8>> {
        let mut ptx_size = 0;
        unsafe { check(nvrtcGetPTXSize(self.handle, &mut ptx_size))? };
        
        let mut ptx = vec![0i8; ptx_size];
        unsafe { check(nvrtcGetPTX(self.handle, ptx.as_mut_ptr()))? };
        
        Ok(ptx.into_iter().map(|c| c as u8).collect())
    }

    /// Get the CUBIN output (CUDA 12+).
    #[cfg(feature = "cuda-12")]
    pub fn get_cubin(&self) -> Result<Vec<u8>> {
        let mut cubin_size = 0;
        unsafe { check(nvrtcGetCUBINSize(self.handle, &mut cubin_size))? };
        
        let mut cubin = vec![0i8; cubin_size];
        unsafe { check(nvrtcGetCUBIN(self.handle, cubin.as_mut_ptr()))? };
        
        Ok(cubin.into_iter().map(|c| c as u8).collect())
    }

    /// Add a name expression for lowered name lookup.
    pub fn add_name_expression(&self, name: &str) -> Result<()> {
        let name = CString::new(name).expect("Invalid name");
        unsafe { check(nvrtcAddNameExpression(self.handle, name.as_ptr())) }
    }

    /// Get the lowered name for a name expression.
    pub fn get_lowered_name(&self, name: &str) -> Result<String> {
        let name = CString::new(name).expect("Invalid name");
        let mut lowered = ptr::null();
        unsafe { check(nvrtcGetLoweredName(self.handle, name.as_ptr(), &mut lowered))? };
        
        Ok(unsafe { CStr::from_ptr(lowered) }
            .to_string_lossy()
            .into_owned())
    }

    /// Get the raw program handle.
    pub fn as_raw(&self) -> nvrtcProgram {
        self.handle
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { nvrtcDestroyProgram(&mut self.handle) };
        }
    }
}

unsafe impl Send for Program {}
unsafe impl Sync for Program {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let result = version();
        assert!(result.is_ok());
    }

    #[test]
    fn test_create_program() {
        let src = r#"
            extern "C" __global__ void hello() {}
        "#;
        let prog = Program::new(src, "hello.cu");
        assert!(prog.is_ok());
    }
}
