//! Raw FFI bindings to CUPTI (CUDA Profiling Tools Interface).

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use cuda_driver_sys::{CUcontext, CUdevice};
use libc::{c_char, c_int, c_uint, c_void, size_t};

pub type CUptiResult = c_int;
pub const CUPTI_SUCCESS: CUptiResult = 0;
pub const CUPTI_ERROR_INVALID_PARAMETER: CUptiResult = 1;
pub const CUPTI_ERROR_INVALID_DEVICE: CUptiResult = 2;
pub const CUPTI_ERROR_INVALID_CONTEXT: CUptiResult = 3;
pub const CUPTI_ERROR_INVALID_EVENT_DOMAIN_ID: CUptiResult = 4;
pub const CUPTI_ERROR_INVALID_EVENT_ID: CUptiResult = 5;
pub const CUPTI_ERROR_INVALID_EVENT_NAME: CUptiResult = 6;
pub const CUPTI_ERROR_INVALID_OPERATION: CUptiResult = 7;
pub const CUPTI_ERROR_OUT_OF_MEMORY: CUptiResult = 8;
pub const CUPTI_ERROR_HARDWARE: CUptiResult = 9;
pub const CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT: CUptiResult = 10;
pub const CUPTI_ERROR_API_NOT_IMPLEMENTED: CUptiResult = 11;
pub const CUPTI_ERROR_MAX_LIMIT_REACHED: CUptiResult = 12;
pub const CUPTI_ERROR_NOT_READY: CUptiResult = 13;
pub const CUPTI_ERROR_NOT_COMPATIBLE: CUptiResult = 14;
pub const CUPTI_ERROR_NOT_INITIALIZED: CUptiResult = 15;
pub const CUPTI_ERROR_INVALID_METRIC_ID: CUptiResult = 16;
pub const CUPTI_ERROR_INVALID_METRIC_NAME: CUptiResult = 17;
pub const CUPTI_ERROR_QUEUE_EMPTY: CUptiResult = 18;
pub const CUPTI_ERROR_INVALID_HANDLE: CUptiResult = 19;
pub const CUPTI_ERROR_INVALID_STREAM: CUptiResult = 20;
pub const CUPTI_ERROR_INVALID_KIND: CUptiResult = 21;
pub const CUPTI_ERROR_INVALID_EVENT_VALUE: CUptiResult = 22;
pub const CUPTI_ERROR_DISABLED: CUptiResult = 23;
pub const CUPTI_ERROR_INVALID_MODULE: CUptiResult = 24;
pub const CUPTI_ERROR_INVALID_METRIC_VALUE: CUptiResult = 25;
pub const CUPTI_ERROR_HARDWARE_BUSY: CUptiResult = 26;
pub const CUPTI_ERROR_UNKNOWN: CUptiResult = 999;

pub type CUpti_ActivityKind = c_int;
pub const CUPTI_ACTIVITY_KIND_INVALID: CUpti_ActivityKind = 0;
pub const CUPTI_ACTIVITY_KIND_MEMCPY: CUpti_ActivityKind = 1;
pub const CUPTI_ACTIVITY_KIND_MEMSET: CUpti_ActivityKind = 2;
pub const CUPTI_ACTIVITY_KIND_KERNEL: CUpti_ActivityKind = 3;
pub const CUPTI_ACTIVITY_KIND_DRIVER: CUpti_ActivityKind = 4;
pub const CUPTI_ACTIVITY_KIND_RUNTIME: CUpti_ActivityKind = 5;
pub const CUPTI_ACTIVITY_KIND_EVENT: CUpti_ActivityKind = 6;
pub const CUPTI_ACTIVITY_KIND_METRIC: CUpti_ActivityKind = 7;
pub const CUPTI_ACTIVITY_KIND_DEVICE: CUpti_ActivityKind = 8;
pub const CUPTI_ACTIVITY_KIND_CONTEXT: CUpti_ActivityKind = 9;
pub const CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: CUpti_ActivityKind = 10;
pub const CUPTI_ACTIVITY_KIND_NAME: CUpti_ActivityKind = 11;
pub const CUPTI_ACTIVITY_KIND_MARKER: CUpti_ActivityKind = 12;
pub const CUPTI_ACTIVITY_KIND_OVERHEAD: CUpti_ActivityKind = 17;

pub type CUpti_CallbackDomain = c_int;
pub const CUPTI_CB_DOMAIN_INVALID: CUpti_CallbackDomain = 0;
pub const CUPTI_CB_DOMAIN_DRIVER_API: CUpti_CallbackDomain = 1;
pub const CUPTI_CB_DOMAIN_RUNTIME_API: CUpti_CallbackDomain = 2;
pub const CUPTI_CB_DOMAIN_RESOURCE: CUpti_CallbackDomain = 3;
pub const CUPTI_CB_DOMAIN_SYNCHRONIZE: CUpti_CallbackDomain = 4;
pub const CUPTI_CB_DOMAIN_NVTX: CUpti_CallbackDomain = 5;

pub type CUpti_CallbackId = c_uint;

#[repr(C)]
pub struct CUpti_Subscriber_st { _unused: [u8; 0] }
pub type CUpti_SubscriberHandle = *mut CUpti_Subscriber_st;

pub type CUpti_CallbackFunc = Option<
    unsafe extern "C" fn(
        userdata: *mut c_void,
        domain: CUpti_CallbackDomain,
        cbid: CUpti_CallbackId,
        cbdata: *const c_void,
    ),
>;

pub type CUpti_BuffersCallbackRequestFunc = Option<
    unsafe extern "C" fn(
        buffer: *mut *mut u8,
        size: *mut size_t,
        maxNumRecords: *mut size_t,
    ),
>;

pub type CUpti_BuffersCallbackCompleteFunc = Option<
    unsafe extern "C" fn(
        context: CUcontext,
        streamId: c_uint,
        buffer: *mut u8,
        size: size_t,
        validSize: size_t,
    ),
>;

extern "C" {
    pub fn cuptiGetVersion(version: *mut c_uint) -> CUptiResult;
    pub fn cuptiGetResultString(result: CUptiResult, str: *mut *const c_char) -> CUptiResult;

    // Subscriber
    pub fn cuptiSubscribe(
        subscriber: *mut CUpti_SubscriberHandle,
        callback: CUpti_CallbackFunc,
        userdata: *mut c_void,
    ) -> CUptiResult;
    pub fn cuptiUnsubscribe(subscriber: CUpti_SubscriberHandle) -> CUptiResult;
    pub fn cuptiEnableDomain(
        enable: c_uint,
        subscriber: CUpti_SubscriberHandle,
        domain: CUpti_CallbackDomain,
    ) -> CUptiResult;
    pub fn cuptiEnableCallback(
        enable: c_uint,
        subscriber: CUpti_SubscriberHandle,
        domain: CUpti_CallbackDomain,
        cbid: CUpti_CallbackId,
    ) -> CUptiResult;

    // Activity
    pub fn cuptiActivityEnable(kind: CUpti_ActivityKind) -> CUptiResult;
    pub fn cuptiActivityDisable(kind: CUpti_ActivityKind) -> CUptiResult;
    pub fn cuptiActivityRegisterCallbacks(
        funcBufferRequested: CUpti_BuffersCallbackRequestFunc,
        funcBufferCompleted: CUpti_BuffersCallbackCompleteFunc,
    ) -> CUptiResult;
    pub fn cuptiActivityFlushAll(flag: c_uint) -> CUptiResult;
    pub fn cuptiActivityGetNextRecord(
        buffer: *mut u8,
        validBufferSizeBytes: size_t,
        record: *mut *mut c_void,
    ) -> CUptiResult;

    // Device
    pub fn cuptiDeviceGetNumEventDomains(device: CUdevice, numDomains: *mut c_uint) -> CUptiResult;
    pub fn cuptiDeviceGetTimestamp(context: CUcontext, timestamp: *mut u64) -> CUptiResult;
}
