//! Safe Rust wrapper for cuDNN.

use cuda_runtime::Stream;
use cudnn_sys::*;
use std::ptr;
use thiserror::Error;

#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
#[error("cuDNN Error: {0}")]
pub struct CudnnError(pub i32);

pub type Result<T> = std::result::Result<T, CudnnError>;

#[inline]
fn check(code: cudnnStatus_t) -> Result<()> {
    if code == CUDNN_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(CudnnError(code))
    }
}

/// Data type for tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Float,
    Double,
    Half,
    Int8,
    Int32,
    Int8x4,
    Uint8,
    Uint8x4,
    Int8x32,
    BFloat16,
}

impl DataType {
    fn to_cudnn(self) -> cudnnDataType_t {
        match self {
            DataType::Float => CUDNN_DATA_FLOAT,
            DataType::Double => CUDNN_DATA_DOUBLE,
            DataType::Half => CUDNN_DATA_HALF,
            DataType::Int8 => CUDNN_DATA_INT8,
            DataType::Int32 => CUDNN_DATA_INT32,
            DataType::Int8x4 => CUDNN_DATA_INT8x4,
            DataType::Uint8 => CUDNN_DATA_UINT8,
            DataType::Uint8x4 => CUDNN_DATA_UINT8x4,
            DataType::Int8x32 => CUDNN_DATA_INT8x32,
            DataType::BFloat16 => CUDNN_DATA_BFLOAT16,
        }
    }
}

/// Tensor format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorFormat {
    Nchw,
    Nhwc,
    NchwVectC,
}

impl TensorFormat {
    fn to_cudnn(self) -> cudnnTensorFormat_t {
        match self {
            TensorFormat::Nchw => CUDNN_TENSOR_NCHW,
            TensorFormat::Nhwc => CUDNN_TENSOR_NHWC,
            TensorFormat::NchwVectC => CUDNN_TENSOR_NCHW_VECT_C,
        }
    }
}

/// Activation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationMode {
    Sigmoid,
    Relu,
    Tanh,
    ClippedRelu,
    Elu,
    Identity,
}

impl ActivationMode {
    fn to_cudnn(self) -> cudnnActivationMode_t {
        match self {
            ActivationMode::Sigmoid => CUDNN_ACTIVATION_SIGMOID,
            ActivationMode::Relu => CUDNN_ACTIVATION_RELU,
            ActivationMode::Tanh => CUDNN_ACTIVATION_TANH,
            ActivationMode::ClippedRelu => CUDNN_ACTIVATION_CLIPPED_RELU,
            ActivationMode::Elu => CUDNN_ACTIVATION_ELU,
            ActivationMode::Identity => CUDNN_ACTIVATION_IDENTITY,
        }
    }
}

/// cuDNN Handle wrapper with automatic resource management.
pub struct Handle {
    handle: cudnnHandle_t,
}

impl Handle {
    /// Create a new cuDNN handle.
    pub fn new() -> Result<Self> {
        let mut handle = ptr::null_mut();
        unsafe { check(cudnnCreate(&mut handle))? };
        Ok(Self { handle })
    }

    /// Set the stream for this handle.
    pub fn set_stream(&self, stream: &Stream) -> Result<()> {
        unsafe { check(cudnnSetStream(self.handle, stream.as_raw())) }
    }

    /// Get cuDNN version.
    pub fn version() -> usize {
        unsafe { cudnnGetVersion() }
    }

    /// Get the raw handle.
    pub fn as_raw(&self) -> cudnnHandle_t {
        self.handle
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { cudnnDestroy(self.handle) };
        }
    }
}

unsafe impl Send for Handle {}
unsafe impl Sync for Handle {}

/// Tensor descriptor.
pub struct TensorDescriptor {
    desc: cudnnTensorDescriptor_t,
}

impl TensorDescriptor {
    /// Create a new tensor descriptor.
    pub fn new() -> Result<Self> {
        let mut desc = ptr::null_mut();
        unsafe { check(cudnnCreateTensorDescriptor(&mut desc))? };
        Ok(Self { desc })
    }

    /// Set 4D tensor descriptor.
    pub fn set_4d(
        &self,
        format: TensorFormat,
        data_type: DataType,
        n: i32,
        c: i32,
        h: i32,
        w: i32,
    ) -> Result<()> {
        unsafe {
            check(cudnnSetTensor4dDescriptor(
                self.desc,
                format.to_cudnn(),
                data_type.to_cudnn(),
                n,
                c,
                h,
                w,
            ))
        }
    }

    /// Get the raw descriptor.
    pub fn as_raw(&self) -> cudnnTensorDescriptor_t {
        self.desc
    }
}

impl Drop for TensorDescriptor {
    fn drop(&mut self) {
        if !self.desc.is_null() {
            unsafe { cudnnDestroyTensorDescriptor(self.desc) };
        }
    }
}

/// Filter descriptor.
pub struct FilterDescriptor {
    desc: cudnnFilterDescriptor_t,
}

impl FilterDescriptor {
    /// Create a new filter descriptor.
    pub fn new() -> Result<Self> {
        let mut desc = ptr::null_mut();
        unsafe { check(cudnnCreateFilterDescriptor(&mut desc))? };
        Ok(Self { desc })
    }

    /// Set 4D filter descriptor.
    pub fn set_4d(
        &self,
        data_type: DataType,
        format: TensorFormat,
        k: i32,
        c: i32,
        h: i32,
        w: i32,
    ) -> Result<()> {
        unsafe {
            check(cudnnSetFilter4dDescriptor(
                self.desc,
                data_type.to_cudnn(),
                format.to_cudnn(),
                k,
                c,
                h,
                w,
            ))
        }
    }

    /// Get the raw descriptor.
    pub fn as_raw(&self) -> cudnnFilterDescriptor_t {
        self.desc
    }
}

impl Drop for FilterDescriptor {
    fn drop(&mut self) {
        if !self.desc.is_null() {
            unsafe { cudnnDestroyFilterDescriptor(self.desc) };
        }
    }
}

/// Convolution descriptor.
pub struct ConvolutionDescriptor {
    desc: cudnnConvolutionDescriptor_t,
}

impl ConvolutionDescriptor {
    /// Create a new convolution descriptor.
    pub fn new() -> Result<Self> {
        let mut desc = ptr::null_mut();
        unsafe { check(cudnnCreateConvolutionDescriptor(&mut desc))? };
        Ok(Self { desc })
    }

    /// Set 2D convolution descriptor.
    pub fn set_2d(
        &self,
        pad_h: i32,
        pad_w: i32,
        stride_h: i32,
        stride_w: i32,
        dilation_h: i32,
        dilation_w: i32,
        mode: i32,
        data_type: DataType,
    ) -> Result<()> {
        unsafe {
            check(cudnnSetConvolution2dDescriptor(
                self.desc,
                pad_h,
                pad_w,
                stride_h,
                stride_w,
                dilation_h,
                dilation_w,
                mode,
                data_type.to_cudnn(),
            ))
        }
    }

    /// Get the raw descriptor.
    pub fn as_raw(&self) -> cudnnConvolutionDescriptor_t {
        self.desc
    }
}

impl Drop for ConvolutionDescriptor {
    fn drop(&mut self) {
        if !self.desc.is_null() {
            unsafe { cudnnDestroyConvolutionDescriptor(self.desc) };
        }
    }
}

/// Activation descriptor.
pub struct ActivationDescriptor {
    desc: cudnnActivationDescriptor_t,
}

impl ActivationDescriptor {
    /// Create a new activation descriptor.
    pub fn new() -> Result<Self> {
        let mut desc = ptr::null_mut();
        unsafe { check(cudnnCreateActivationDescriptor(&mut desc))? };
        Ok(Self { desc })
    }

    /// Set the activation descriptor.
    pub fn set(&self, mode: ActivationMode, nan_propagation: i32, coef: f64) -> Result<()> {
        unsafe { check(cudnnSetActivationDescriptor(self.desc, mode.to_cudnn(), nan_propagation, coef)) }
    }

    /// Get the raw descriptor.
    pub fn as_raw(&self) -> cudnnActivationDescriptor_t {
        self.desc
    }
}

impl Drop for ActivationDescriptor {
    fn drop(&mut self) {
        if !self.desc.is_null() {
            unsafe { cudnnDestroyActivationDescriptor(self.desc) };
        }
    }
}

/// Pooling descriptor.
pub struct PoolingDescriptor {
    desc: cudnnPoolingDescriptor_t,
}

impl PoolingDescriptor {
    /// Create a new pooling descriptor.
    pub fn new() -> Result<Self> {
        let mut desc = ptr::null_mut();
        unsafe { check(cudnnCreatePoolingDescriptor(&mut desc))? };
        Ok(Self { desc })
    }

    /// Set 2D pooling descriptor.
    pub fn set_2d(
        &self,
        mode: i32,
        nan_propagation: i32,
        window_h: i32,
        window_w: i32,
        pad_h: i32,
        pad_w: i32,
        stride_h: i32,
        stride_w: i32,
    ) -> Result<()> {
        unsafe {
            check(cudnnSetPooling2dDescriptor(
                self.desc,
                mode,
                nan_propagation,
                window_h,
                window_w,
                pad_h,
                pad_w,
                stride_h,
                stride_w,
            ))
        }
    }

    /// Get the raw descriptor.
    pub fn as_raw(&self) -> cudnnPoolingDescriptor_t {
        self.desc
    }
}

impl Drop for PoolingDescriptor {
    fn drop(&mut self) {
        if !self.desc.is_null() {
            unsafe { cudnnDestroyPoolingDescriptor(self.desc) };
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
