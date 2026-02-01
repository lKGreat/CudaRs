//! Safe Rust wrapper for NPP (NVIDIA Performance Primitives).

use cuda_runtime::Stream;
use npp_sys::*;
use thiserror::Error;

#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
#[error("NPP Error: {0}")]
pub struct NppError(pub i32);

pub type Result<T> = std::result::Result<T, NppError>;

#[inline]
fn check(code: NppStatus) -> Result<()> {
    if code >= NPP_SUCCESS {
        Ok(())
    } else {
        Err(NppError(code))
    }
}

/// Interpolation mode for resize operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMode {
    NearestNeighbor,
    Linear,
    Cubic,
    Super,
    Lanczos,
}

impl InterpolationMode {
    fn to_npp(self) -> NppiInterpolationMode {
        match self {
            InterpolationMode::NearestNeighbor => NPPI_INTER_NN,
            InterpolationMode::Linear => NPPI_INTER_LINEAR,
            InterpolationMode::Cubic => NPPI_INTER_CUBIC,
            InterpolationMode::Super => NPPI_INTER_SUPER,
            InterpolationMode::Lanczos => NPPI_INTER_LANCZOS,
        }
    }
}

/// 2D size structure.
#[derive(Debug, Clone, Copy, Default)]
pub struct Size {
    pub width: i32,
    pub height: i32,
}

impl Size {
    pub fn new(width: i32, height: i32) -> Self {
        Self { width, height }
    }

    fn to_npp(self) -> NppiSize {
        NppiSize {
            width: self.width,
            height: self.height,
        }
    }
}

/// 2D point structure.
#[derive(Debug, Clone, Copy, Default)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

impl Point {
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

/// Rectangle structure.
#[derive(Debug, Clone, Copy, Default)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

impl Rect {
    pub fn new(x: i32, y: i32, width: i32, height: i32) -> Self {
        Self { x, y, width, height }
    }

    pub fn from_size(size: Size) -> Self {
        Self {
            x: 0,
            y: 0,
            width: size.width,
            height: size.height,
        }
    }

    fn to_npp(self) -> NppiRect {
        NppiRect {
            x: self.x,
            y: self.y,
            width: self.width,
            height: self.height,
        }
    }
}

/// Set the NPP stream.
pub fn set_stream(stream: &Stream) -> Result<()> {
    unsafe { check(nppSetStream(stream.as_raw())) }
}

/// Get the current NPP stream.
pub fn get_stream() -> cuda_runtime_sys::cudaStream_t {
    unsafe { nppGetStream() }
}

/// Allocate 2D image memory (8-bit, 1 channel).
pub fn malloc_8u_c1(width: i32, height: i32) -> Result<(*mut u8, i32)> {
    let mut step = 0;
    let ptr = unsafe { nppiMalloc_8u_C1(width, height, &mut step) };
    if ptr.is_null() {
        Err(NppError(NPP_MEMORY_ALLOCATION_ERR))
    } else {
        Ok((ptr, step))
    }
}

/// Allocate 2D image memory (8-bit, 3 channels).
pub fn malloc_8u_c3(width: i32, height: i32) -> Result<(*mut u8, i32)> {
    let mut step = 0;
    let ptr = unsafe { nppiMalloc_8u_C3(width, height, &mut step) };
    if ptr.is_null() {
        Err(NppError(NPP_MEMORY_ALLOCATION_ERR))
    } else {
        Ok((ptr, step))
    }
}

/// Allocate 2D image memory (8-bit, 4 channels).
pub fn malloc_8u_c4(width: i32, height: i32) -> Result<(*mut u8, i32)> {
    let mut step = 0;
    let ptr = unsafe { nppiMalloc_8u_C4(width, height, &mut step) };
    if ptr.is_null() {
        Err(NppError(NPP_MEMORY_ALLOCATION_ERR))
    } else {
        Ok((ptr, step))
    }
}

/// Allocate 2D image memory (32-bit float, 1 channel).
pub fn malloc_32f_c1(width: i32, height: i32) -> Result<(*mut f32, i32)> {
    let mut step = 0;
    let ptr = unsafe { nppiMalloc_32f_C1(width, height, &mut step) };
    if ptr.is_null() {
        Err(NppError(NPP_MEMORY_ALLOCATION_ERR))
    } else {
        Ok((ptr, step))
    }
}

/// Free NPP allocated memory.
pub fn free(ptr: *mut libc::c_void) {
    unsafe { nppiFree(ptr) };
}

/// Resize 8-bit, 1 channel image.
pub fn resize_8u_c1(
    src: *const u8,
    src_step: i32,
    src_size: Size,
    src_roi: Rect,
    dst: *mut u8,
    dst_step: i32,
    dst_size: Size,
    dst_roi: Rect,
    interpolation: InterpolationMode,
) -> Result<()> {
    unsafe {
        check(nppiResize_8u_C1R(
            src,
            src_step,
            src_size.to_npp(),
            src_roi.to_npp(),
            dst,
            dst_step,
            dst_size.to_npp(),
            dst_roi.to_npp(),
            interpolation.to_npp(),
        ))
    }
}

/// Resize 8-bit, 3 channel image.
pub fn resize_8u_c3(
    src: *const u8,
    src_step: i32,
    src_size: Size,
    src_roi: Rect,
    dst: *mut u8,
    dst_step: i32,
    dst_size: Size,
    dst_roi: Rect,
    interpolation: InterpolationMode,
) -> Result<()> {
    unsafe {
        check(nppiResize_8u_C3R(
            src,
            src_step,
            src_size.to_npp(),
            src_roi.to_npp(),
            dst,
            dst_step,
            dst_size.to_npp(),
            dst_roi.to_npp(),
            interpolation.to_npp(),
        ))
    }
}

/// Convert RGB to grayscale (8-bit, 3 channel to 1 channel).
pub fn rgb_to_gray_8u(
    src: *const u8,
    src_step: i32,
    dst: *mut u8,
    dst_step: i32,
    roi_size: Size,
) -> Result<()> {
    unsafe {
        check(nppiRGBToGray_8u_C3C1R(
            src,
            src_step,
            dst,
            dst_step,
            roi_size.to_npp(),
        ))
    }
}

/// Convert BGR to grayscale (8-bit, 3 channel to 1 channel).
pub fn bgr_to_gray_8u(
    src: *const u8,
    src_step: i32,
    dst: *mut u8,
    dst_step: i32,
    roi_size: Size,
) -> Result<()> {
    unsafe {
        check(nppiBGRToGray_8u_C3C1R(
            src,
            src_step,
            dst,
            dst_step,
            roi_size.to_npp(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size() {
        let size = Size::new(640, 480);
        assert_eq!(size.width, 640);
        assert_eq!(size.height, 480);
    }
}
