//! Raw FFI bindings to NVRTC (NVIDIA Runtime Compilation).

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use libc::{c_char, c_int, size_t};

pub type nvrtcResult = c_int;
pub const NVRTC_SUCCESS: nvrtcResult = 0;
pub const NVRTC_ERROR_OUT_OF_MEMORY: nvrtcResult = 1;
pub const NVRTC_ERROR_PROGRAM_CREATION_FAILURE: nvrtcResult = 2;
pub const NVRTC_ERROR_INVALID_INPUT: nvrtcResult = 3;
pub const NVRTC_ERROR_INVALID_PROGRAM: nvrtcResult = 4;
pub const NVRTC_ERROR_INVALID_OPTION: nvrtcResult = 5;
pub const NVRTC_ERROR_COMPILATION: nvrtcResult = 6;
pub const NVRTC_ERROR_BUILTIN_OPERATION_FAILURE: nvrtcResult = 7;
pub const NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION: nvrtcResult = 8;
pub const NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION: nvrtcResult = 9;
pub const NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID: nvrtcResult = 10;
pub const NVRTC_ERROR_INTERNAL_ERROR: nvrtcResult = 11;

#[repr(C)]
pub struct _nvrtcProgram { _unused: [u8; 0] }
pub type nvrtcProgram = *mut _nvrtcProgram;

extern "C" {
    pub fn nvrtcVersion(major: *mut c_int, minor: *mut c_int) -> nvrtcResult;
    pub fn nvrtcGetErrorString(result: nvrtcResult) -> *const c_char;

    pub fn nvrtcCreateProgram(
        prog: *mut nvrtcProgram,
        src: *const c_char,
        name: *const c_char,
        numHeaders: c_int,
        headers: *const *const c_char,
        includeNames: *const *const c_char,
    ) -> nvrtcResult;

    pub fn nvrtcDestroyProgram(prog: *mut nvrtcProgram) -> nvrtcResult;

    pub fn nvrtcCompileProgram(
        prog: nvrtcProgram,
        numOptions: c_int,
        options: *const *const c_char,
    ) -> nvrtcResult;

    pub fn nvrtcGetPTXSize(prog: nvrtcProgram, ptxSizeRet: *mut size_t) -> nvrtcResult;
    pub fn nvrtcGetPTX(prog: nvrtcProgram, ptx: *mut c_char) -> nvrtcResult;

    pub fn nvrtcGetCUBINSize(prog: nvrtcProgram, cubinSizeRet: *mut size_t) -> nvrtcResult;
    pub fn nvrtcGetCUBIN(prog: nvrtcProgram, cubin: *mut c_char) -> nvrtcResult;

    pub fn nvrtcGetProgramLogSize(prog: nvrtcProgram, logSizeRet: *mut size_t) -> nvrtcResult;
    pub fn nvrtcGetProgramLog(prog: nvrtcProgram, log: *mut c_char) -> nvrtcResult;

    pub fn nvrtcAddNameExpression(prog: nvrtcProgram, name_expression: *const c_char) -> nvrtcResult;
    pub fn nvrtcGetLoweredName(
        prog: nvrtcProgram,
        name_expression: *const c_char,
        lowered_name: *mut *const c_char,
    ) -> nvrtcResult;

    #[cfg(feature = "cuda-12")]
    pub fn nvrtcGetNVVMSize(prog: nvrtcProgram, nvvmSizeRet: *mut size_t) -> nvrtcResult;
    #[cfg(feature = "cuda-12")]
    pub fn nvrtcGetNVVM(prog: nvrtcProgram, nvvm: *mut c_char) -> nvrtcResult;

    pub fn nvrtcGetNumSupportedArchs(numArchs: *mut c_int) -> nvrtcResult;
    pub fn nvrtcGetSupportedArchs(supportedArchs: *mut c_int) -> nvrtcResult;
}
