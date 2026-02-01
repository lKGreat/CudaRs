//! Safe Rust wrapper for cuFFT.

use cufft_sys::*;
use cuda_runtime::Stream;
use thiserror::Error;

#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
#[error("cuFFT Error: {0}")]
pub struct CufftError(pub i32);

pub type Result<T> = std::result::Result<T, CufftError>;

#[inline]
fn check(code: cufftResult) -> Result<()> {
    if code == CUFFT_SUCCESS {
        Ok(())
    } else {
        Err(CufftError(code))
    }
}

/// FFT direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Forward,
    Inverse,
}

impl Direction {
    fn to_cufft(self) -> i32 {
        match self {
            Direction::Forward => CUFFT_FORWARD,
            Direction::Inverse => CUFFT_INVERSE,
        }
    }
}

/// Complex number (single precision)
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Complex32 {
    pub x: f32,
    pub y: f32,
}

impl Complex32 {
    pub fn new(re: f32, im: f32) -> Self {
        Self { x: re, y: im }
    }
}

/// Complex number (double precision)
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Complex64 {
    pub x: f64,
    pub y: f64,
}

impl Complex64 {
    pub fn new(re: f64, im: f64) -> Self {
        Self { x: re, y: im }
    }
}

/// 1D FFT Plan (Complex-to-Complex, single precision)
pub struct Plan1dC2C {
    handle: cufftHandle,
    size: i32,
}

impl Plan1dC2C {
    /// Create a new 1D C2C FFT plan.
    pub fn new(nx: i32) -> Result<Self> {
        let mut handle = 0;
        unsafe { check(cufftPlan1d(&mut handle, nx, CUFFT_C2C, 1))? };
        Ok(Self { handle, size: nx })
    }

    /// Create a new 1D C2C FFT plan for batched transforms.
    pub fn new_batched(nx: i32, batch: i32) -> Result<Self> {
        let mut handle = 0;
        unsafe { check(cufftPlan1d(&mut handle, nx, CUFFT_C2C, batch))? };
        Ok(Self { handle, size: nx })
    }

    /// Set the stream for this plan.
    pub fn set_stream(&self, stream: &Stream) -> Result<()> {
        unsafe { check(cufftSetStream(self.handle, stream.as_raw())) }
    }

    /// Execute the FFT in-place.
    pub fn exec_inplace(&self, data: *mut Complex32, direction: Direction) -> Result<()> {
        unsafe {
            check(cufftExecC2C(
                self.handle,
                data as *mut cufftComplex,
                data as *mut cufftComplex,
                direction.to_cufft(),
            ))
        }
    }

    /// Execute the FFT out-of-place.
    pub fn exec(
        &self,
        input: *const Complex32,
        output: *mut Complex32,
        direction: Direction,
    ) -> Result<()> {
        unsafe {
            check(cufftExecC2C(
                self.handle,
                input as *mut cufftComplex,
                output as *mut cufftComplex,
                direction.to_cufft(),
            ))
        }
    }

    /// Get the size of the plan.
    pub fn size(&self) -> i32 {
        self.size
    }

    /// Get the raw handle.
    pub fn as_raw(&self) -> cufftHandle {
        self.handle
    }
}

impl Drop for Plan1dC2C {
    fn drop(&mut self) {
        unsafe { cufftDestroy(self.handle) };
    }
}

/// 1D FFT Plan (Real-to-Complex, single precision)
pub struct Plan1dR2C {
    handle: cufftHandle,
    size: i32,
}

impl Plan1dR2C {
    /// Create a new 1D R2C FFT plan.
    pub fn new(nx: i32) -> Result<Self> {
        let mut handle = 0;
        unsafe { check(cufftPlan1d(&mut handle, nx, CUFFT_R2C, 1))? };
        Ok(Self { handle, size: nx })
    }

    /// Set the stream for this plan.
    pub fn set_stream(&self, stream: &Stream) -> Result<()> {
        unsafe { check(cufftSetStream(self.handle, stream.as_raw())) }
    }

    /// Execute the FFT.
    pub fn exec(&self, input: *const f32, output: *mut Complex32) -> Result<()> {
        unsafe {
            check(cufftExecR2C(
                self.handle,
                input as *mut cufftReal,
                output as *mut cufftComplex,
            ))
        }
    }

    /// Get the size of the plan.
    pub fn size(&self) -> i32 {
        self.size
    }

    /// Get the raw handle.
    pub fn as_raw(&self) -> cufftHandle {
        self.handle
    }
}

impl Drop for Plan1dR2C {
    fn drop(&mut self) {
        unsafe { cufftDestroy(self.handle) };
    }
}

/// 1D FFT Plan (Complex-to-Real, single precision)
pub struct Plan1dC2R {
    handle: cufftHandle,
    size: i32,
}

impl Plan1dC2R {
    /// Create a new 1D C2R FFT plan.
    pub fn new(nx: i32) -> Result<Self> {
        let mut handle = 0;
        unsafe { check(cufftPlan1d(&mut handle, nx, CUFFT_C2R, 1))? };
        Ok(Self { handle, size: nx })
    }

    /// Set the stream for this plan.
    pub fn set_stream(&self, stream: &Stream) -> Result<()> {
        unsafe { check(cufftSetStream(self.handle, stream.as_raw())) }
    }

    /// Execute the FFT.
    pub fn exec(&self, input: *const Complex32, output: *mut f32) -> Result<()> {
        unsafe {
            check(cufftExecC2R(
                self.handle,
                input as *mut cufftComplex,
                output as *mut cufftReal,
            ))
        }
    }

    /// Get the size of the plan.
    pub fn size(&self) -> i32 {
        self.size
    }

    /// Get the raw handle.
    pub fn as_raw(&self) -> cufftHandle {
        self.handle
    }
}

impl Drop for Plan1dC2R {
    fn drop(&mut self) {
        unsafe { cufftDestroy(self.handle) };
    }
}

/// 2D FFT Plan (Complex-to-Complex, single precision)
pub struct Plan2dC2C {
    handle: cufftHandle,
    nx: i32,
    ny: i32,
}

impl Plan2dC2C {
    /// Create a new 2D C2C FFT plan.
    pub fn new(nx: i32, ny: i32) -> Result<Self> {
        let mut handle = 0;
        unsafe { check(cufftPlan2d(&mut handle, nx, ny, CUFFT_C2C))? };
        Ok(Self { handle, nx, ny })
    }

    /// Set the stream for this plan.
    pub fn set_stream(&self, stream: &Stream) -> Result<()> {
        unsafe { check(cufftSetStream(self.handle, stream.as_raw())) }
    }

    /// Execute the FFT in-place.
    pub fn exec_inplace(&self, data: *mut Complex32, direction: Direction) -> Result<()> {
        unsafe {
            check(cufftExecC2C(
                self.handle,
                data as *mut cufftComplex,
                data as *mut cufftComplex,
                direction.to_cufft(),
            ))
        }
    }

    /// Execute the FFT out-of-place.
    pub fn exec(
        &self,
        input: *const Complex32,
        output: *mut Complex32,
        direction: Direction,
    ) -> Result<()> {
        unsafe {
            check(cufftExecC2C(
                self.handle,
                input as *mut cufftComplex,
                output as *mut cufftComplex,
                direction.to_cufft(),
            ))
        }
    }

    /// Get the dimensions.
    pub fn dimensions(&self) -> (i32, i32) {
        (self.nx, self.ny)
    }

    /// Get the raw handle.
    pub fn as_raw(&self) -> cufftHandle {
        self.handle
    }
}

impl Drop for Plan2dC2C {
    fn drop(&mut self) {
        unsafe { cufftDestroy(self.handle) };
    }
}

/// 3D FFT Plan (Complex-to-Complex, single precision)
pub struct Plan3dC2C {
    handle: cufftHandle,
    nx: i32,
    ny: i32,
    nz: i32,
}

impl Plan3dC2C {
    /// Create a new 3D C2C FFT plan.
    pub fn new(nx: i32, ny: i32, nz: i32) -> Result<Self> {
        let mut handle = 0;
        unsafe { check(cufftPlan3d(&mut handle, nx, ny, nz, CUFFT_C2C))? };
        Ok(Self { handle, nx, ny, nz })
    }

    /// Set the stream for this plan.
    pub fn set_stream(&self, stream: &Stream) -> Result<()> {
        unsafe { check(cufftSetStream(self.handle, stream.as_raw())) }
    }

    /// Execute the FFT in-place.
    pub fn exec_inplace(&self, data: *mut Complex32, direction: Direction) -> Result<()> {
        unsafe {
            check(cufftExecC2C(
                self.handle,
                data as *mut cufftComplex,
                data as *mut cufftComplex,
                direction.to_cufft(),
            ))
        }
    }

    /// Get the dimensions.
    pub fn dimensions(&self) -> (i32, i32, i32) {
        (self.nx, self.ny, self.nz)
    }

    /// Get the raw handle.
    pub fn as_raw(&self) -> cufftHandle {
        self.handle
    }
}

impl Drop for Plan3dC2C {
    fn drop(&mut self) {
        unsafe { cufftDestroy(self.handle) };
    }
}

unsafe impl Send for Plan1dC2C {}
unsafe impl Send for Plan1dR2C {}
unsafe impl Send for Plan1dC2R {}
unsafe impl Send for Plan2dC2C {}
unsafe impl Send for Plan3dC2C {}
