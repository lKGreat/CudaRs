//! Safe Rust wrapper for NVML (NVIDIA Management Library).

use nvml_sys::*;
use std::ffi::CStr;
use std::ptr;
use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq, Eq)]
#[error("NVML Error: {0} ({1})")]
pub struct NvmlError(pub i32, pub String);

impl NvmlError {
    pub fn from_code(code: nvmlReturn_t) -> Self {
        let msg = unsafe {
            let ptr = nvmlErrorString(code);
            if ptr.is_null() {
                "Unknown error".to_string()
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        NvmlError(code, msg)
    }
}

pub type Result<T> = std::result::Result<T, NvmlError>;

#[inline]
fn check(code: nvmlReturn_t) -> Result<()> {
    if code == NVML_SUCCESS {
        Ok(())
    } else {
        Err(NvmlError::from_code(code))
    }
}

/// Initialize NVML. Must be called before any other NVML functions.
pub fn init() -> Result<()> {
    unsafe { check(nvmlInit_v2()) }
}

/// Shutdown NVML. Should be called after all NVML operations.
pub fn shutdown() -> Result<()> {
    unsafe { check(nvmlShutdown()) }
}

/// Get the NVIDIA driver version.
pub fn driver_version() -> Result<String> {
    let mut version = [0i8; 80];
    unsafe { check(nvmlSystemGetDriverVersion(version.as_mut_ptr(), 80))? };
    Ok(unsafe { CStr::from_ptr(version.as_ptr()) }
        .to_string_lossy()
        .into_owned())
}

/// Get the NVML version.
pub fn nvml_version() -> Result<String> {
    let mut version = [0i8; 80];
    unsafe { check(nvmlSystemGetNVMLVersion(version.as_mut_ptr(), 80))? };
    Ok(unsafe { CStr::from_ptr(version.as_ptr()) }
        .to_string_lossy()
        .into_owned())
}

/// Get the CUDA driver version.
pub fn cuda_driver_version() -> Result<i32> {
    let mut version = 0;
    unsafe { check(nvmlSystemGetCudaDriverVersion_v2(&mut version))? };
    Ok(version)
}

/// Get the number of GPU devices.
pub fn device_count() -> Result<u32> {
    let mut count = 0;
    unsafe { check(nvmlDeviceGetCount_v2(&mut count))? };
    Ok(count)
}

/// GPU Device wrapper.
pub struct Device {
    handle: nvmlDevice_t,
}

impl Device {
    /// Get a device by index.
    pub fn by_index(index: u32) -> Result<Self> {
        let mut handle = ptr::null_mut();
        unsafe { check(nvmlDeviceGetHandleByIndex_v2(index, &mut handle))? };
        Ok(Self { handle })
    }

    /// Get a device by UUID.
    pub fn by_uuid(uuid: &str) -> Result<Self> {
        let uuid = std::ffi::CString::new(uuid).expect("Invalid UUID");
        let mut handle = ptr::null_mut();
        unsafe { check(nvmlDeviceGetHandleByUUID(uuid.as_ptr(), &mut handle))? };
        Ok(Self { handle })
    }

    /// Get the device name.
    pub fn name(&self) -> Result<String> {
        let mut name = [0i8; NVML_DEVICE_NAME_BUFFER_SIZE];
        unsafe {
            check(nvmlDeviceGetName(
                self.handle,
                name.as_mut_ptr(),
                NVML_DEVICE_NAME_BUFFER_SIZE as u32,
            ))?
        };
        Ok(unsafe { CStr::from_ptr(name.as_ptr()) }
            .to_string_lossy()
            .into_owned())
    }

    /// Get the device UUID.
    pub fn uuid(&self) -> Result<String> {
        let mut uuid = [0i8; NVML_DEVICE_UUID_BUFFER_SIZE];
        unsafe {
            check(nvmlDeviceGetUUID(
                self.handle,
                uuid.as_mut_ptr(),
                NVML_DEVICE_UUID_BUFFER_SIZE as u32,
            ))?
        };
        Ok(unsafe { CStr::from_ptr(uuid.as_ptr()) }
            .to_string_lossy()
            .into_owned())
    }

    /// Get the device index.
    pub fn index(&self) -> Result<u32> {
        let mut index = 0;
        unsafe { check(nvmlDeviceGetIndex(self.handle, &mut index))? };
        Ok(index)
    }

    /// Get memory information.
    pub fn memory_info(&self) -> Result<MemoryInfo> {
        let mut memory = nvmlMemory_t {
            total: 0,
            free: 0,
            used: 0,
        };
        unsafe { check(nvmlDeviceGetMemoryInfo(self.handle, &mut memory))? };
        Ok(MemoryInfo {
            total: memory.total as u64,
            free: memory.free as u64,
            used: memory.used as u64,
        })
    }

    /// Get GPU utilization rates.
    pub fn utilization_rates(&self) -> Result<UtilizationRates> {
        let mut util = nvmlUtilization_t { gpu: 0, memory: 0 };
        unsafe { check(nvmlDeviceGetUtilizationRates(self.handle, &mut util))? };
        Ok(UtilizationRates {
            gpu: util.gpu,
            memory: util.memory,
        })
    }

    /// Get GPU temperature in Celsius.
    pub fn temperature(&self) -> Result<u32> {
        let mut temp = 0;
        unsafe { check(nvmlDeviceGetTemperature(self.handle, NVML_TEMPERATURE_GPU, &mut temp))? };
        Ok(temp)
    }

    /// Get fan speed as a percentage.
    pub fn fan_speed(&self) -> Result<u32> {
        let mut speed = 0;
        unsafe { check(nvmlDeviceGetFanSpeed(self.handle, &mut speed))? };
        Ok(speed)
    }

    /// Get power usage in milliwatts.
    pub fn power_usage(&self) -> Result<u32> {
        let mut power = 0;
        unsafe { check(nvmlDeviceGetPowerUsage(self.handle, &mut power))? };
        Ok(power)
    }

    /// Get power management limit in milliwatts.
    pub fn power_limit(&self) -> Result<u32> {
        let mut limit = 0;
        unsafe { check(nvmlDeviceGetPowerManagementLimit(self.handle, &mut limit))? };
        Ok(limit)
    }

    /// Get clock speed in MHz for the specified clock type.
    pub fn clock_info(&self, clock_type: ClockType) -> Result<u32> {
        let mut clock = 0;
        unsafe { check(nvmlDeviceGetClockInfo(self.handle, clock_type.to_nvml(), &mut clock))? };
        Ok(clock)
    }

    /// Get max clock speed in MHz for the specified clock type.
    pub fn max_clock_info(&self, clock_type: ClockType) -> Result<u32> {
        let mut clock = 0;
        unsafe { check(nvmlDeviceGetMaxClockInfo(self.handle, clock_type.to_nvml(), &mut clock))? };
        Ok(clock)
    }

    /// Get compute capability.
    pub fn compute_capability(&self) -> Result<(i32, i32)> {
        let mut major = 0;
        let mut minor = 0;
        unsafe { check(nvmlDeviceGetCudaComputeCapability(self.handle, &mut major, &mut minor))? };
        Ok((major, minor))
    }

    /// Get the current performance state.
    pub fn performance_state(&self) -> Result<i32> {
        let mut state = 0;
        unsafe { check(nvmlDeviceGetPerformanceState(self.handle, &mut state))? };
        Ok(state)
    }
}

/// Memory information.
#[derive(Debug, Clone, Copy)]
pub struct MemoryInfo {
    pub total: u64,
    pub free: u64,
    pub used: u64,
}

/// Utilization rates.
#[derive(Debug, Clone, Copy)]
pub struct UtilizationRates {
    pub gpu: u32,
    pub memory: u32,
}

/// Clock type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClockType {
    Graphics,
    Sm,
    Memory,
    Video,
}

impl ClockType {
    fn to_nvml(self) -> nvmlClockType_t {
        match self {
            ClockType::Graphics => NVML_CLOCK_GRAPHICS,
            ClockType::Sm => NVML_CLOCK_SM,
            ClockType::Memory => NVML_CLOCK_MEM,
            ClockType::Video => NVML_CLOCK_VIDEO,
        }
    }
}

/// RAII guard for NVML initialization.
pub struct NvmlGuard;

impl NvmlGuard {
    /// Initialize NVML and return a guard that will shutdown on drop.
    pub fn new() -> Result<Self> {
        init()?;
        Ok(Self)
    }
}

impl Drop for NvmlGuard {
    fn drop(&mut self) {
        let _ = shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_shutdown() {
        // This may fail if no NVIDIA driver is installed
        let _ = NvmlGuard::new();
    }
}
