use libc::c_char;

use cudars_core::SdkErr;

use super::sdk_error::set_last_error;

pub fn read_utf8(ptr: *const c_char, len: usize, field: &str) -> Result<String, SdkErr> {
    if len == 0 {
        return Ok(String::new());
    }
    if ptr.is_null() {
        set_last_error(&format!("{field} pointer is null"));
        return Err(SdkErr::InvalidArg);
    }

    let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) };
    match std::str::from_utf8(bytes) {
        Ok(s) => Ok(s.to_string()),
        Err(_) => {
            set_last_error(&format!("{field} is not valid UTF-8"));
            Err(SdkErr::InvalidArg)
        }
    }
}
