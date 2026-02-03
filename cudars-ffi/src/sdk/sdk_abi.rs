use libc::c_char;

use cudars_core::SdkErr;

use super::sdk_error::{clear_last_error, set_last_error, with_panic_boundary_err, with_panic_boundary_ptr, with_panic_boundary_u32};

const SDK_ABI_VERSION: u32 = 1;
const SDK_VERSION: &[u8] = b"0.1.0\0";

#[no_mangle]
pub extern "C" fn sdk_abi_version() -> u32 {
    with_panic_boundary_u32("sdk_abi_version", 0, || SDK_ABI_VERSION)
}

#[no_mangle]
pub extern "C" fn sdk_version_string() -> *const c_char {
    with_panic_boundary_ptr("sdk_version_string", std::ptr::null(), || SDK_VERSION.as_ptr() as *const c_char)
}

#[no_mangle]
pub extern "C" fn sdk_version_string_len(out_len: *mut usize) -> SdkErr {
    with_panic_boundary_err("sdk_version_string_len", || {
        if out_len.is_null() {
            set_last_error("out_len is null");
            return SdkErr::InvalidArg;
        }

        unsafe {
            *out_len = SDK_VERSION.len().saturating_sub(1);
        }
        clear_last_error();
        SdkErr::Ok
    })
}

#[no_mangle]
pub extern "C" fn sdk_version_string_write(buf: *mut c_char, cap: usize, out_written: *mut usize) -> SdkErr {
    with_panic_boundary_err("sdk_version_string_write", || {
        if buf.is_null() || out_written.is_null() {
            set_last_error("buffer or out_written is null");
            return SdkErr::InvalidArg;
        }

        let src = &SDK_VERSION[..SDK_VERSION.len().saturating_sub(1)];
        let to_copy = src.len().min(cap);

        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), buf as *mut u8, to_copy);
            *out_written = to_copy;
        }

        clear_last_error();
        if to_copy < src.len() {
            set_last_error("buffer too small");
            return SdkErr::InvalidArg;
        }

        SdkErr::Ok
    })
}
