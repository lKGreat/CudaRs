use std::cell::RefCell;
use std::panic::{catch_unwind, AssertUnwindSafe};

use libc::c_char;

use crate::CudaRsResult;
use cudars_core::SdkErr;

thread_local! {
    static LAST_ERROR: RefCell<Vec<u8>> = RefCell::new(Vec::new());
}

const EMPTY_ERROR: &[u8] = b"";

pub fn clear_last_error() {
    LAST_ERROR.with(|msg| msg.borrow_mut().clear());
}

pub fn set_last_error(message: &str) {
    LAST_ERROR.with(|msg| {
        let mut buffer = msg.as_bytes().to_vec();
        buffer.retain(|b| *b != 0);
        *msg.borrow_mut() = buffer;
    });
}

pub fn map_cudars_result(result: CudaRsResult) -> SdkErr {
    match result {
        CudaRsResult::Success => SdkErr::Ok,
        CudaRsResult::ErrorInvalidValue => SdkErr::InvalidArg,
        CudaRsResult::ErrorOutOfMemory => SdkErr::OutOfMemory,
        CudaRsResult::ErrorNotInitialized => SdkErr::BadState,
        CudaRsResult::ErrorInvalidHandle => SdkErr::InvalidArg,
        CudaRsResult::ErrorNotSupported => SdkErr::Unsupported,
        CudaRsResult::ErrorUnknown => SdkErr::Runtime,
    }
}

pub fn handle_cudars_result(result: CudaRsResult, context: &str) -> SdkErr {
    let err = map_cudars_result(result);
    if err == SdkErr::Ok {
        clear_last_error();
    } else {
        set_last_error(&format!("{context}: {result:?}"));
    }
    err
}

pub fn with_panic_boundary_err<F>(context: &str, f: F) -> SdkErr
where
    F: FnOnce() -> SdkErr,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(_) => {
            set_last_error(&format!("panic in {context}"));
            SdkErr::Runtime
        }
    }
}

pub fn with_panic_boundary_u32<F>(context: &str, fallback: u32, f: F) -> u32
where
    F: FnOnce() -> u32,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(_) => {
            set_last_error(&format!("panic in {context}"));
            fallback
        }
    }
}

pub fn with_panic_boundary_ptr<T, F>(context: &str, fallback: *const T, f: F) -> *const T
where
    F: FnOnce() -> *const T,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(_) => {
            set_last_error(&format!("panic in {context}"));
            fallback
        }
    }
}

#[no_mangle]
pub extern "C" fn sdk_last_error_message_utf8(out_ptr: *mut *const c_char, out_len: *mut usize) -> SdkErr {
    with_panic_boundary_err("sdk_last_error_message_utf8", || {
        if out_ptr.is_null() || out_len.is_null() {
            return SdkErr::InvalidArg;
        }

        LAST_ERROR.with(|msg| {
            let buffer = msg.borrow();
            if buffer.is_empty() {
                unsafe {
                    *out_ptr = EMPTY_ERROR.as_ptr() as *const c_char;
                    *out_len = 0;
                }
                return;
            }

            unsafe {
                *out_ptr = buffer.as_ptr() as *const c_char;
                *out_len = buffer.len();
            }
        });

        SdkErr::Ok
    })
}
