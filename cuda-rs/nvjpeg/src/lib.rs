//! Safe Rust wrapper for nvJPEG.

use cuda_runtime::Stream;
use nvjpeg_sys::*;
use std::ptr;
use thiserror::Error;

#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
#[error("nvJPEG Error: {0}")]
pub struct NvjpegError(pub i32);

pub type Result<T> = std::result::Result<T, NvjpegError>;

#[inline]
fn check(code: nvjpegStatus_t) -> Result<()> {
    if code == NVJPEG_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(NvjpegError(code))
    }
}

/// Output format for decoded images.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Unchanged,
    Yuv,
    Y,
    Rgb,
    Bgr,
    Rgbi,
    Bgri,
}

#[allow(dead_code)]
impl OutputFormat {
    fn to_nvjpeg(self) -> nvjpegOutputFormat_t {
        match self {
            OutputFormat::Unchanged => NVJPEG_OUTPUT_UNCHANGED,
            OutputFormat::Yuv => NVJPEG_OUTPUT_YUV,
            OutputFormat::Y => NVJPEG_OUTPUT_Y,
            OutputFormat::Rgb => NVJPEG_OUTPUT_RGB,
            OutputFormat::Bgr => NVJPEG_OUTPUT_BGR,
            OutputFormat::Rgbi => NVJPEG_OUTPUT_RGBI,
            OutputFormat::Bgri => NVJPEG_OUTPUT_BGRI,
        }
    }
}

/// Backend for nvJPEG.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Default,
    Hybrid,
    GpuHybrid,
    Hardware,
}

impl Backend {
    fn to_nvjpeg(self) -> nvjpegBackend_t {
        match self {
            Backend::Default => NVJPEG_BACKEND_DEFAULT,
            Backend::Hybrid => NVJPEG_BACKEND_HYBRID,
            Backend::GpuHybrid => NVJPEG_BACKEND_GPU_HYBRID,
            Backend::Hardware => NVJPEG_BACKEND_HARDWARE,
        }
    }
}

/// Chroma subsampling type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaSubsampling {
    Css444,
    Css422,
    Css420,
    Css440,
    Css411,
    Css410,
    Gray,
    Unknown,
}

impl From<nvjpegChromaSubsampling_t> for ChromaSubsampling {
    fn from(value: nvjpegChromaSubsampling_t) -> Self {
        match value {
            NVJPEG_CSS_444 => ChromaSubsampling::Css444,
            NVJPEG_CSS_422 => ChromaSubsampling::Css422,
            NVJPEG_CSS_420 => ChromaSubsampling::Css420,
            NVJPEG_CSS_440 => ChromaSubsampling::Css440,
            NVJPEG_CSS_411 => ChromaSubsampling::Css411,
            NVJPEG_CSS_410 => ChromaSubsampling::Css410,
            NVJPEG_CSS_GRAY => ChromaSubsampling::Gray,
            _ => ChromaSubsampling::Unknown,
        }
    }
}

/// nvJPEG Handle wrapper with automatic resource management.
pub struct Handle {
    handle: nvjpegHandle_t,
}

impl Handle {
    /// Create a new nvJPEG handle.
    pub fn new() -> Result<Self> {
        Self::with_backend(Backend::Default)
    }

    /// Create a new nvJPEG handle with a specific backend.
    pub fn with_backend(backend: Backend) -> Result<Self> {
        let mut handle = ptr::null_mut();
        unsafe { check(nvjpegCreate(backend.to_nvjpeg(), ptr::null_mut(), &mut handle))? };
        Ok(Self { handle })
    }

    /// Get image information from JPEG data.
    pub fn get_image_info(&self, data: &[u8]) -> Result<ImageInfo> {
        let mut n_components = 0;
        let mut subsampling = 0;
        let mut widths = [0i32; 4];
        let mut heights = [0i32; 4];
        
        unsafe {
            check(nvjpegGetImageInfo(
                self.handle,
                data.as_ptr(),
                data.len(),
                &mut n_components,
                &mut subsampling,
                widths.as_mut_ptr(),
                heights.as_mut_ptr(),
            ))?;
        }
        
        Ok(ImageInfo {
            n_components,
            subsampling: ChromaSubsampling::from(subsampling),
            width: widths[0],
            height: heights[0],
        })
    }

    /// Get the raw handle.
    pub fn as_raw(&self) -> nvjpegHandle_t {
        self.handle
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { nvjpegDestroy(self.handle) };
        }
    }
}

unsafe impl Send for Handle {}
unsafe impl Sync for Handle {}

/// JPEG image information.
#[derive(Debug, Clone, Copy)]
pub struct ImageInfo {
    pub n_components: i32,
    pub subsampling: ChromaSubsampling,
    pub width: i32,
    pub height: i32,
}

/// JPEG State for decoding.
pub struct JpegState {
    handle: nvjpegJpegState_t,
}

impl JpegState {
    /// Create a new JPEG state.
    pub fn new(nvjpeg: &Handle) -> Result<Self> {
        let mut handle = ptr::null_mut();
        unsafe { check(nvjpegJpegStateCreate(nvjpeg.as_raw(), &mut handle))? };
        Ok(Self { handle })
    }

    /// Get the raw handle.
    pub fn as_raw(&self) -> nvjpegJpegState_t {
        self.handle
    }
}

impl Drop for JpegState {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { nvjpegJpegStateDestroy(self.handle) };
        }
    }
}

/// Encoder state.
pub struct EncoderState {
    handle: nvjpegEncoderState_t,
}

impl EncoderState {
    /// Create a new encoder state.
    pub fn new(nvjpeg: &Handle, stream: &Stream) -> Result<Self> {
        let mut handle = ptr::null_mut();
        unsafe { check(nvjpegEncoderStateCreate(nvjpeg.as_raw(), &mut handle, stream.as_raw()))? };
        Ok(Self { handle })
    }

    /// Get the raw handle.
    pub fn as_raw(&self) -> nvjpegEncoderState_t {
        self.handle
    }
}

impl Drop for EncoderState {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { nvjpegEncoderStateDestroy(self.handle) };
        }
    }
}

/// Encoder parameters.
pub struct EncoderParams {
    handle: nvjpegEncoderParams_t,
}

impl EncoderParams {
    /// Create new encoder parameters.
    pub fn new(nvjpeg: &Handle, stream: &Stream) -> Result<Self> {
        let mut handle = ptr::null_mut();
        unsafe { check(nvjpegEncoderParamsCreate(nvjpeg.as_raw(), &mut handle, stream.as_raw()))? };
        Ok(Self { handle })
    }

    /// Set the quality (1-100).
    pub fn set_quality(&self, quality: i32, stream: &Stream) -> Result<()> {
        unsafe { check(nvjpegEncoderParamsSetQuality(self.handle, quality, stream.as_raw())) }
    }

    /// Get the raw handle.
    pub fn as_raw(&self) -> nvjpegEncoderParams_t {
        self.handle
    }
}

impl Drop for EncoderParams {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { nvjpegEncoderParamsDestroy(self.handle) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_handle() {
        let _ = Handle::new();
    }
}
