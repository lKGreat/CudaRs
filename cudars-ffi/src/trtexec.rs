//! TensorRT trtexec wrapper (spawn external process).

use crate::CudaRsResult;
use libc::{c_char, c_int, c_ulonglong};
use std::ffi::CStr;
use std::process::{Command, Stdio};

/// Run trtexec with custom arguments.
///
/// - exe_path: full path to trtexec.exe
/// - args: array of utf-8 strings (argv style)
/// - workdir: optional working directory (nullable)
/// - exit_code: receives process exit code (nullable)
#[no_mangle]
pub extern "C" fn cudars_trtexec_run(
    exe_path: *const c_char,
    args: *const *const c_char,
    args_len: c_ulonglong,
    workdir: *const c_char,
    exit_code: *mut c_int,
) -> CudaRsResult {
    if exe_path.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let exe = unsafe {
        match CStr::from_ptr(exe_path).to_str() {
            Ok(p) if !p.is_empty() => p,
            _ => return CudaRsResult::ErrorInvalidValue,
        }
    };

    let mut cmd = Command::new(exe);
    cmd.stdout(Stdio::inherit()).stderr(Stdio::inherit());

    if !workdir.is_null() {
        let wd = unsafe { CStr::from_ptr(workdir) }.to_str().unwrap_or("");
        if !wd.is_empty() {
            cmd.current_dir(wd);
        }
    }

    if !args.is_null() && args_len > 0 {
        let slice = unsafe { std::slice::from_raw_parts(args, args_len as usize) };
        for &arg_ptr in slice.iter() {
            if arg_ptr.is_null() {
                return CudaRsResult::ErrorInvalidValue;
            }
            let arg = unsafe { CStr::from_ptr(arg_ptr) }.to_str().map_err(|_| CudaRsResult::ErrorInvalidValue);
            match arg {
                Ok(a) => cmd.arg(a),
                Err(code) => return code,
            };
        }
    }

    match cmd.status() {
        Ok(status) => {
            if !exit_code.is_null() {
                unsafe {
                    *exit_code = status.code().unwrap_or(-1);
                }
            }
            CudaRsResult::Success
        }
        Err(_) => CudaRsResult::ErrorUnknown,
    }
}
