use libc::c_char;

use cudars_core::SdkErr;

use super::sdk_error::{clear_last_error, set_last_error, with_panic_boundary_err, with_panic_boundary_ptr, with_panic_boundary_u32};

const SDK_ABI_VERSION: u32 = 2;
const SDK_VERSION: &[u8] = b"0.1.0\0";

// Feature flag bitmask (kept in sync with C#/docs)
pub const SDK_F_TENSORRT: u64 = 1 << 0;
pub const SDK_F_ONNX: u64 = 1 << 1;
pub const SDK_F_OV_CPU: u64 = 1 << 2;
pub const SDK_F_OV_GPU: u64 = 1 << 3;
pub const SDK_F_PADDLE: u64 = 1 << 4;
pub const SDK_F_MULTI_STREAM: u64 = 1 << 5;

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
pub extern "C" fn sdk_feature_flags(out_flags: *mut u64) -> SdkErr {
    with_panic_boundary_err("sdk_feature_flags", || {
        if out_flags.is_null() {
            set_last_error("out_flags is null");
            return SdkErr::InvalidArg;
        }

        #[allow(unused_mut)]
        let mut flags: u64 = 0;
        #[cfg(feature = "tensorrt")]
        {
            flags |= SDK_F_TENSORRT | SDK_F_MULTI_STREAM;
        }
        #[cfg(feature = "onnxruntime")]
        {
            flags |= SDK_F_ONNX;
        }
        #[cfg(feature = "openvino")]
        {
            // OpenVINO 可同时 CPU/GPU，具体设备在运行时再判定
            flags |= SDK_F_OV_CPU | SDK_F_OV_GPU | SDK_F_MULTI_STREAM;
        }
        #[cfg(feature = "paddleocr")]
        {
            flags |= SDK_F_PADDLE;
        }

        unsafe {
            *out_flags = flags;
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
