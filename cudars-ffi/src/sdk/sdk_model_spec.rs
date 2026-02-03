use libc::c_char;

use cudars_core::ModelKind;

#[repr(C)]
pub struct SdkModelSpec {
    pub id_ptr: *const c_char,
    pub id_len: usize,
    pub kind: ModelKind,
    pub config_json_ptr: *const c_char,
    pub config_json_len: usize,
}
