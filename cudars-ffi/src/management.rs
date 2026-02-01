//! NVML FFI exports.

use super::CudaRsResult;
use nvml::{self, Device};
use libc::c_uint;

/// Initialize NVML.
#[no_mangle]
pub extern "C" fn cudars_nvml_init() -> CudaRsResult {
    match nvml::init() {
        Ok(()) => CudaRsResult::Success,
        Err(_) => CudaRsResult::ErrorNotInitialized,
    }
}

/// Shutdown NVML.
#[no_mangle]
pub extern "C" fn cudars_nvml_shutdown() -> CudaRsResult {
    match nvml::shutdown() {
        Ok(()) => CudaRsResult::Success,
        Err(_) => CudaRsResult::ErrorUnknown,
    }
}

/// Get GPU device count.
#[no_mangle]
pub extern "C" fn cudars_nvml_device_get_count(count: *mut c_uint) -> CudaRsResult {
    if count.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }
    match nvml::device_count() {
        Ok(c) => {
            unsafe { *count = c };
            CudaRsResult::Success
        }
        Err(_) => CudaRsResult::ErrorUnknown,
    }
}

/// Memory info structure for C.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CudaRsMemoryInfo {
    pub total: u64,
    pub free: u64,
    pub used: u64,
}

/// Utilization rates structure for C.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CudaRsUtilizationRates {
    pub gpu: c_uint,
    pub memory: c_uint,
}

/// Get memory info for a device.
#[no_mangle]
pub extern "C" fn cudars_nvml_device_get_memory_info(
    index: c_uint,
    info: *mut CudaRsMemoryInfo,
) -> CudaRsResult {
    if info.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }
    
    let device = match Device::by_index(index) {
        Ok(d) => d,
        Err(_) => return CudaRsResult::ErrorInvalidValue,
    };
    
    match device.memory_info() {
        Ok(mem) => {
            unsafe {
                (*info).total = mem.total;
                (*info).free = mem.free;
                (*info).used = mem.used;
            }
            CudaRsResult::Success
        }
        Err(_) => CudaRsResult::ErrorUnknown,
    }
}

/// Get utilization rates for a device.
#[no_mangle]
pub extern "C" fn cudars_nvml_device_get_utilization_rates(
    index: c_uint,
    util: *mut CudaRsUtilizationRates,
) -> CudaRsResult {
    if util.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }
    
    let device = match Device::by_index(index) {
        Ok(d) => d,
        Err(_) => return CudaRsResult::ErrorInvalidValue,
    };
    
    match device.utilization_rates() {
        Ok(rates) => {
            unsafe {
                (*util).gpu = rates.gpu;
                (*util).memory = rates.memory;
            }
            CudaRsResult::Success
        }
        Err(_) => CudaRsResult::ErrorUnknown,
    }
}

/// Get temperature for a device.
#[no_mangle]
pub extern "C" fn cudars_nvml_device_get_temperature(
    index: c_uint,
    temp: *mut c_uint,
) -> CudaRsResult {
    if temp.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }
    
    let device = match Device::by_index(index) {
        Ok(d) => d,
        Err(_) => return CudaRsResult::ErrorInvalidValue,
    };
    
    match device.temperature() {
        Ok(t) => {
            unsafe { *temp = t };
            CudaRsResult::Success
        }
        Err(_) => CudaRsResult::ErrorUnknown,
    }
}

/// Get power usage for a device (in milliwatts).
#[no_mangle]
pub extern "C" fn cudars_nvml_device_get_power_usage(
    index: c_uint,
    power: *mut c_uint,
) -> CudaRsResult {
    if power.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }
    
    let device = match Device::by_index(index) {
        Ok(d) => d,
        Err(_) => return CudaRsResult::ErrorInvalidValue,
    };
    
    match device.power_usage() {
        Ok(p) => {
            unsafe { *power = p };
            CudaRsResult::Success
        }
        Err(_) => CudaRsResult::ErrorUnknown,
    }
}

/// Get fan speed for a device (percentage).
#[no_mangle]
pub extern "C" fn cudars_nvml_device_get_fan_speed(
    index: c_uint,
    speed: *mut c_uint,
) -> CudaRsResult {
    if speed.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }
    
    let device = match Device::by_index(index) {
        Ok(d) => d,
        Err(_) => return CudaRsResult::ErrorInvalidValue,
    };
    
    match device.fan_speed() {
        Ok(s) => {
            unsafe { *speed = s };
            CudaRsResult::Success
        }
        Err(_) => CudaRsResult::ErrorUnknown,
    }
}
