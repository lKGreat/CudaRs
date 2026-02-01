//! NVRTC FFI exports.

use super::CudaRsResult;
use nvrtc::Program;
use libc::{c_char, c_int, size_t};
use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::Mutex;

lazy_static::lazy_static! {
    static ref PROGRAMS: Mutex<HandleManager<Program>> = Mutex::new(HandleManager::new());
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

pub type CudaRsProgram = u64;

/// Get NVRTC version.
#[no_mangle]
pub extern "C" fn cudars_nvrtc_version(major: *mut c_int, minor: *mut c_int) -> CudaRsResult {
    if major.is_null() || minor.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }
    match nvrtc::version() {
        Ok((maj, min)) => {
            unsafe {
                *major = maj;
                *minor = min;
            }
            CudaRsResult::Success
        }
        Err(_) => CudaRsResult::ErrorUnknown,
    }
}

/// Create a program from CUDA C source.
#[no_mangle]
pub extern "C" fn cudars_program_create(
    program: *mut CudaRsProgram,
    src: *const c_char,
    name: *const c_char,
) -> CudaRsResult {
    if program.is_null() || src.is_null() || name.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }
    
    let src_str = unsafe { CStr::from_ptr(src) }.to_str().unwrap_or("");
    let name_str = unsafe { CStr::from_ptr(name) }.to_str().unwrap_or("program.cu");
    
    match Program::new(src_str, name_str) {
        Ok(p) => {
            let mut programs = PROGRAMS.lock().unwrap();
            let id = programs.insert(p);
            unsafe { *program = id };
            CudaRsResult::Success
        }
        Err(_) => CudaRsResult::ErrorUnknown,
    }
}

/// Destroy a program.
#[no_mangle]
pub extern "C" fn cudars_program_destroy(program: CudaRsProgram) -> CudaRsResult {
    let mut programs = PROGRAMS.lock().unwrap();
    match programs.remove(program) {
        Some(_) => CudaRsResult::Success,
        None => CudaRsResult::ErrorInvalidHandle,
    }
}

/// Compile a program.
#[no_mangle]
pub extern "C" fn cudars_program_compile(
    program: CudaRsProgram,
    options: *const *const c_char,
    num_options: c_int,
) -> CudaRsResult {
    let programs = PROGRAMS.lock().unwrap();
    match programs.get(program) {
        Some(p) => {
            let opts: Vec<&str> = if options.is_null() || num_options <= 0 {
                vec![]
            } else {
                unsafe {
                    std::slice::from_raw_parts(options, num_options as usize)
                        .iter()
                        .filter_map(|&opt| {
                            if opt.is_null() {
                                None
                            } else {
                                CStr::from_ptr(opt).to_str().ok()
                            }
                        })
                        .collect()
                }
            };
            
            match p.compile(&opts) {
                Ok(()) => CudaRsResult::Success,
                Err(_) => CudaRsResult::ErrorUnknown,
            }
        }
        None => CudaRsResult::ErrorInvalidHandle,
    }
}

/// Get PTX size.
#[no_mangle]
pub extern "C" fn cudars_program_get_ptx_size(
    program: CudaRsProgram,
    size: *mut size_t,
) -> CudaRsResult {
    if size.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }
    
    let programs = PROGRAMS.lock().unwrap();
    match programs.get(program) {
        Some(p) => match p.get_ptx() {
            Ok(ptx) => {
                unsafe { *size = ptx.len() };
                CudaRsResult::Success
            }
            Err(_) => CudaRsResult::ErrorUnknown,
        },
        None => CudaRsResult::ErrorInvalidHandle,
    }
}
