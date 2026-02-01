//! Safe Rust wrapper for CUPTI (CUDA Profiling Tools Interface).

use cupti_sys::*;
use std::ffi::CStr;
use std::ptr;
use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq, Eq)]
#[error("CUPTI Error: {0} ({1})")]
pub struct CuptiError(pub i32, pub String);

impl CuptiError {
    pub fn from_code(code: CUptiResult) -> Self {
        let msg = unsafe {
            let mut ptr = ptr::null();
            if cuptiGetResultString(code, &mut ptr) == CUPTI_SUCCESS && !ptr.is_null() {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            } else {
                "Unknown error".to_string()
            }
        };
        CuptiError(code, msg)
    }
}

pub type Result<T> = std::result::Result<T, CuptiError>;

#[inline]
fn check(code: CUptiResult) -> Result<()> {
    if code == CUPTI_SUCCESS {
        Ok(())
    } else {
        Err(CuptiError::from_code(code))
    }
}

/// Get CUPTI version.
pub fn version() -> Result<u32> {
    let mut version = 0;
    unsafe { check(cuptiGetVersion(&mut version))? };
    Ok(version)
}

/// Activity kind for profiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivityKind {
    Memcpy,
    Memset,
    Kernel,
    Driver,
    Runtime,
    Event,
    Metric,
    Device,
    Context,
    ConcurrentKernel,
    Name,
    Marker,
    Overhead,
}

impl ActivityKind {
    fn to_cupti(self) -> CUpti_ActivityKind {
        match self {
            ActivityKind::Memcpy => CUPTI_ACTIVITY_KIND_MEMCPY,
            ActivityKind::Memset => CUPTI_ACTIVITY_KIND_MEMSET,
            ActivityKind::Kernel => CUPTI_ACTIVITY_KIND_KERNEL,
            ActivityKind::Driver => CUPTI_ACTIVITY_KIND_DRIVER,
            ActivityKind::Runtime => CUPTI_ACTIVITY_KIND_RUNTIME,
            ActivityKind::Event => CUPTI_ACTIVITY_KIND_EVENT,
            ActivityKind::Metric => CUPTI_ACTIVITY_KIND_METRIC,
            ActivityKind::Device => CUPTI_ACTIVITY_KIND_DEVICE,
            ActivityKind::Context => CUPTI_ACTIVITY_KIND_CONTEXT,
            ActivityKind::ConcurrentKernel => CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
            ActivityKind::Name => CUPTI_ACTIVITY_KIND_NAME,
            ActivityKind::Marker => CUPTI_ACTIVITY_KIND_MARKER,
            ActivityKind::Overhead => CUPTI_ACTIVITY_KIND_OVERHEAD,
        }
    }
}

/// Callback domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallbackDomain {
    DriverApi,
    RuntimeApi,
    Resource,
    Synchronize,
    Nvtx,
}

impl CallbackDomain {
    fn to_cupti(self) -> CUpti_CallbackDomain {
        match self {
            CallbackDomain::DriverApi => CUPTI_CB_DOMAIN_DRIVER_API,
            CallbackDomain::RuntimeApi => CUPTI_CB_DOMAIN_RUNTIME_API,
            CallbackDomain::Resource => CUPTI_CB_DOMAIN_RESOURCE,
            CallbackDomain::Synchronize => CUPTI_CB_DOMAIN_SYNCHRONIZE,
            CallbackDomain::Nvtx => CUPTI_CB_DOMAIN_NVTX,
        }
    }
}

/// Enable activity profiling for a specific kind.
pub fn activity_enable(kind: ActivityKind) -> Result<()> {
    unsafe { check(cuptiActivityEnable(kind.to_cupti())) }
}

/// Disable activity profiling for a specific kind.
pub fn activity_disable(kind: ActivityKind) -> Result<()> {
    unsafe { check(cuptiActivityDisable(kind.to_cupti())) }
}

/// Flush all activity records.
pub fn activity_flush_all(flag: u32) -> Result<()> {
    unsafe { check(cuptiActivityFlushAll(flag)) }
}

/// Subscriber handle wrapper.
pub struct Subscriber {
    handle: CUpti_SubscriberHandle,
}

impl Subscriber {
    /// Create a new subscriber with a callback function.
    /// 
    /// # Safety
    /// The callback function must be valid for the lifetime of the subscriber.
    pub unsafe fn new(
        callback: CUpti_CallbackFunc,
        userdata: *mut std::ffi::c_void,
    ) -> Result<Self> {
        let mut handle = ptr::null_mut();
        check(cuptiSubscribe(&mut handle, callback, userdata))?;
        Ok(Self { handle })
    }

    /// Enable callbacks for an entire domain.
    pub fn enable_domain(&self, enable: bool, domain: CallbackDomain) -> Result<()> {
        unsafe {
            check(cuptiEnableDomain(
                if enable { 1 } else { 0 },
                self.handle,
                domain.to_cupti(),
            ))
        }
    }

    /// Enable a specific callback.
    pub fn enable_callback(
        &self,
        enable: bool,
        domain: CallbackDomain,
        callback_id: u32,
    ) -> Result<()> {
        unsafe {
            check(cuptiEnableCallback(
                if enable { 1 } else { 0 },
                self.handle,
                domain.to_cupti(),
                callback_id,
            ))
        }
    }
}

impl Drop for Subscriber {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { cuptiUnsubscribe(self.handle) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        // This may fail if CUPTI is not available
        let _ = version();
    }
}
