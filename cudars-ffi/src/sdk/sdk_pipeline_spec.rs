use libc::c_char;

use cudars_core::PipelineKind;

#[repr(C)]
pub struct SdkPipelineSpec {
    pub id_ptr: *const c_char,
    pub id_len: usize,
    pub kind: PipelineKind,
    pub config_json_ptr: *const c_char,
    pub config_json_len: usize,
}
