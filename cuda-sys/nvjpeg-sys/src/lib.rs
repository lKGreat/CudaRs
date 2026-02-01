//! Raw FFI bindings to nvJPEG.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use cuda_runtime_sys::cudaStream_t;
use libc::{c_int, c_uchar, size_t};

pub type nvjpegStatus_t = c_int;
pub const NVJPEG_STATUS_SUCCESS: nvjpegStatus_t = 0;
pub const NVJPEG_STATUS_NOT_INITIALIZED: nvjpegStatus_t = 1;
pub const NVJPEG_STATUS_INVALID_PARAMETER: nvjpegStatus_t = 2;
pub const NVJPEG_STATUS_BAD_JPEG: nvjpegStatus_t = 3;
pub const NVJPEG_STATUS_JPEG_NOT_SUPPORTED: nvjpegStatus_t = 4;
pub const NVJPEG_STATUS_ALLOCATOR_FAILURE: nvjpegStatus_t = 5;
pub const NVJPEG_STATUS_EXECUTION_FAILED: nvjpegStatus_t = 6;
pub const NVJPEG_STATUS_ARCH_MISMATCH: nvjpegStatus_t = 7;
pub const NVJPEG_STATUS_INTERNAL_ERROR: nvjpegStatus_t = 8;
pub const NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED: nvjpegStatus_t = 9;

pub type nvjpegOutputFormat_t = c_int;
pub const NVJPEG_OUTPUT_UNCHANGED: nvjpegOutputFormat_t = 0;
pub const NVJPEG_OUTPUT_YUV: nvjpegOutputFormat_t = 1;
pub const NVJPEG_OUTPUT_Y: nvjpegOutputFormat_t = 2;
pub const NVJPEG_OUTPUT_RGB: nvjpegOutputFormat_t = 3;
pub const NVJPEG_OUTPUT_BGR: nvjpegOutputFormat_t = 4;
pub const NVJPEG_OUTPUT_RGBI: nvjpegOutputFormat_t = 5;
pub const NVJPEG_OUTPUT_BGRI: nvjpegOutputFormat_t = 6;

pub type nvjpegInputFormat_t = c_int;
pub const NVJPEG_INPUT_RGB: nvjpegInputFormat_t = 3;
pub const NVJPEG_INPUT_BGR: nvjpegInputFormat_t = 4;
pub const NVJPEG_INPUT_RGBI: nvjpegInputFormat_t = 5;
pub const NVJPEG_INPUT_BGRI: nvjpegInputFormat_t = 6;

pub type nvjpegBackend_t = c_int;
pub const NVJPEG_BACKEND_DEFAULT: nvjpegBackend_t = 0;
pub const NVJPEG_BACKEND_HYBRID: nvjpegBackend_t = 1;
pub const NVJPEG_BACKEND_GPU_HYBRID: nvjpegBackend_t = 2;
pub const NVJPEG_BACKEND_HARDWARE: nvjpegBackend_t = 3;

pub type nvjpegChromaSubsampling_t = c_int;
pub const NVJPEG_CSS_444: nvjpegChromaSubsampling_t = 0;
pub const NVJPEG_CSS_422: nvjpegChromaSubsampling_t = 1;
pub const NVJPEG_CSS_420: nvjpegChromaSubsampling_t = 2;
pub const NVJPEG_CSS_440: nvjpegChromaSubsampling_t = 3;
pub const NVJPEG_CSS_411: nvjpegChromaSubsampling_t = 4;
pub const NVJPEG_CSS_410: nvjpegChromaSubsampling_t = 5;
pub const NVJPEG_CSS_GRAY: nvjpegChromaSubsampling_t = 6;
pub const NVJPEG_CSS_UNKNOWN: nvjpegChromaSubsampling_t = -1;

#[repr(C)]
pub struct nvjpegHandle { _unused: [u8; 0] }
pub type nvjpegHandle_t = *mut nvjpegHandle;

#[repr(C)]
pub struct nvjpegJpegState { _unused: [u8; 0] }
pub type nvjpegJpegState_t = *mut nvjpegJpegState;

#[repr(C)]
pub struct nvjpegEncoderState { _unused: [u8; 0] }
pub type nvjpegEncoderState_t = *mut nvjpegEncoderState;

#[repr(C)]
pub struct nvjpegEncoderParams { _unused: [u8; 0] }
pub type nvjpegEncoderParams_t = *mut nvjpegEncoderParams;

#[repr(C)]
pub struct nvjpegBufferPinned { _unused: [u8; 0] }
pub type nvjpegBufferPinned_t = *mut nvjpegBufferPinned;

#[repr(C)]
pub struct nvjpegBufferDevice { _unused: [u8; 0] }
pub type nvjpegBufferDevice_t = *mut nvjpegBufferDevice;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct nvjpegImage_t {
    pub channel: [*mut c_uchar; 4],
    pub pitch: [c_int; 4],
}

extern "C" {
    pub fn nvjpegGetProperty(type_: c_int, value: *mut c_int) -> nvjpegStatus_t;
    pub fn nvjpegGetCudartProperty(type_: c_int, value: *mut c_int) -> nvjpegStatus_t;

    pub fn nvjpegCreate(backend: nvjpegBackend_t, handle: *mut nvjpegHandle_t) -> nvjpegStatus_t;
    pub fn nvjpegDestroy(handle: nvjpegHandle_t) -> nvjpegStatus_t;

    pub fn nvjpegJpegStateCreate(handle: nvjpegHandle_t, jpeg_handle: *mut nvjpegJpegState_t) -> nvjpegStatus_t;
    pub fn nvjpegJpegStateDestroy(jpeg_handle: nvjpegJpegState_t) -> nvjpegStatus_t;

    pub fn nvjpegGetImageInfo(
        handle: nvjpegHandle_t,
        data: *const c_uchar,
        length: size_t,
        nComponents: *mut c_int,
        subsampling: *mut nvjpegChromaSubsampling_t,
        widths: *mut c_int,
        heights: *mut c_int,
    ) -> nvjpegStatus_t;

    pub fn nvjpegDecode(
        handle: nvjpegHandle_t,
        jpeg_handle: nvjpegJpegState_t,
        data: *const c_uchar,
        length: size_t,
        output_format: nvjpegOutputFormat_t,
        destination: *mut nvjpegImage_t,
        stream: cudaStream_t,
    ) -> nvjpegStatus_t;

    // Encoder
    pub fn nvjpegEncoderStateCreate(
        handle: nvjpegHandle_t,
        enc_state: *mut nvjpegEncoderState_t,
        stream: cudaStream_t,
    ) -> nvjpegStatus_t;
    pub fn nvjpegEncoderStateDestroy(enc_state: nvjpegEncoderState_t) -> nvjpegStatus_t;

    pub fn nvjpegEncoderParamsCreate(
        handle: nvjpegHandle_t,
        encoder_params: *mut nvjpegEncoderParams_t,
        stream: cudaStream_t,
    ) -> nvjpegStatus_t;
    pub fn nvjpegEncoderParamsDestroy(encoder_params: nvjpegEncoderParams_t) -> nvjpegStatus_t;

    pub fn nvjpegEncoderParamsSetQuality(
        encoder_params: nvjpegEncoderParams_t,
        quality: c_int,
        stream: cudaStream_t,
    ) -> nvjpegStatus_t;

    pub fn nvjpegEncodeImage(
        handle: nvjpegHandle_t,
        enc_state: nvjpegEncoderState_t,
        encoder_params: nvjpegEncoderParams_t,
        source: *const nvjpegImage_t,
        input_format: nvjpegInputFormat_t,
        image_width: c_int,
        image_height: c_int,
        stream: cudaStream_t,
    ) -> nvjpegStatus_t;

    pub fn nvjpegEncodeRetrieveBitstream(
        handle: nvjpegHandle_t,
        enc_state: nvjpegEncoderState_t,
        data: *mut c_uchar,
        length: *mut size_t,
        stream: cudaStream_t,
    ) -> nvjpegStatus_t;
}
