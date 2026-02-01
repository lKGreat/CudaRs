//! Raw FFI bindings to NVML (NVIDIA Management Library).

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use libc::{c_char, c_int, c_uint, c_ulong};

pub type nvmlReturn_t = c_int;
pub const NVML_SUCCESS: nvmlReturn_t = 0;
pub const NVML_ERROR_UNINITIALIZED: nvmlReturn_t = 1;
pub const NVML_ERROR_INVALID_ARGUMENT: nvmlReturn_t = 2;
pub const NVML_ERROR_NOT_SUPPORTED: nvmlReturn_t = 3;
pub const NVML_ERROR_NO_PERMISSION: nvmlReturn_t = 4;
pub const NVML_ERROR_ALREADY_INITIALIZED: nvmlReturn_t = 5;
pub const NVML_ERROR_NOT_FOUND: nvmlReturn_t = 6;
pub const NVML_ERROR_INSUFFICIENT_SIZE: nvmlReturn_t = 7;
pub const NVML_ERROR_INSUFFICIENT_POWER: nvmlReturn_t = 8;
pub const NVML_ERROR_DRIVER_NOT_LOADED: nvmlReturn_t = 9;
pub const NVML_ERROR_TIMEOUT: nvmlReturn_t = 10;
pub const NVML_ERROR_IRQ_ISSUE: nvmlReturn_t = 11;
pub const NVML_ERROR_LIBRARY_NOT_FOUND: nvmlReturn_t = 12;
pub const NVML_ERROR_FUNCTION_NOT_FOUND: nvmlReturn_t = 13;
pub const NVML_ERROR_CORRUPTED_INFOROM: nvmlReturn_t = 14;
pub const NVML_ERROR_GPU_IS_LOST: nvmlReturn_t = 15;
pub const NVML_ERROR_RESET_REQUIRED: nvmlReturn_t = 16;
pub const NVML_ERROR_OPERATING_SYSTEM: nvmlReturn_t = 17;
pub const NVML_ERROR_LIB_RM_VERSION_MISMATCH: nvmlReturn_t = 18;
pub const NVML_ERROR_IN_USE: nvmlReturn_t = 19;
pub const NVML_ERROR_MEMORY: nvmlReturn_t = 20;
pub const NVML_ERROR_NO_DATA: nvmlReturn_t = 21;
pub const NVML_ERROR_VGPU_ECC_NOT_SUPPORTED: nvmlReturn_t = 22;
pub const NVML_ERROR_UNKNOWN: nvmlReturn_t = 999;

pub type nvmlTemperatureSensors_t = c_int;
pub const NVML_TEMPERATURE_GPU: nvmlTemperatureSensors_t = 0;
pub const NVML_TEMPERATURE_COUNT: nvmlTemperatureSensors_t = 1;

pub type nvmlClockType_t = c_int;
pub const NVML_CLOCK_GRAPHICS: nvmlClockType_t = 0;
pub const NVML_CLOCK_SM: nvmlClockType_t = 1;
pub const NVML_CLOCK_MEM: nvmlClockType_t = 2;
pub const NVML_CLOCK_VIDEO: nvmlClockType_t = 3;
pub const NVML_CLOCK_COUNT: nvmlClockType_t = 4;

pub type nvmlMemoryLocation_t = c_int;
pub const NVML_MEMORY_LOCATION_L1_CACHE: nvmlMemoryLocation_t = 0;
pub const NVML_MEMORY_LOCATION_L2_CACHE: nvmlMemoryLocation_t = 1;
pub const NVML_MEMORY_LOCATION_DRAM: nvmlMemoryLocation_t = 2;
pub const NVML_MEMORY_LOCATION_DEVICE_MEMORY: nvmlMemoryLocation_t = 2;
pub const NVML_MEMORY_LOCATION_REGISTER_FILE: nvmlMemoryLocation_t = 3;
pub const NVML_MEMORY_LOCATION_TEXTURE_MEMORY: nvmlMemoryLocation_t = 4;

pub type nvmlPstates_t = c_int;
pub const NVML_PSTATE_0: nvmlPstates_t = 0;
pub const NVML_PSTATE_1: nvmlPstates_t = 1;
pub const NVML_PSTATE_2: nvmlPstates_t = 2;
pub const NVML_PSTATE_3: nvmlPstates_t = 3;
pub const NVML_PSTATE_4: nvmlPstates_t = 4;
pub const NVML_PSTATE_5: nvmlPstates_t = 5;
pub const NVML_PSTATE_6: nvmlPstates_t = 6;
pub const NVML_PSTATE_7: nvmlPstates_t = 7;
pub const NVML_PSTATE_8: nvmlPstates_t = 8;
pub const NVML_PSTATE_9: nvmlPstates_t = 9;
pub const NVML_PSTATE_10: nvmlPstates_t = 10;
pub const NVML_PSTATE_11: nvmlPstates_t = 11;
pub const NVML_PSTATE_12: nvmlPstates_t = 12;
pub const NVML_PSTATE_13: nvmlPstates_t = 13;
pub const NVML_PSTATE_14: nvmlPstates_t = 14;
pub const NVML_PSTATE_15: nvmlPstates_t = 15;
pub const NVML_PSTATE_UNKNOWN: nvmlPstates_t = 32;

#[repr(C)]
pub struct nvmlDevice_st { _unused: [u8; 0] }
pub type nvmlDevice_t = *mut nvmlDevice_st;

#[repr(C)]
pub struct nvmlUnit_st { _unused: [u8; 0] }
pub type nvmlUnit_t = *mut nvmlUnit_st;

pub const NVML_DEVICE_NAME_BUFFER_SIZE: usize = 64;
pub const NVML_DEVICE_UUID_BUFFER_SIZE: usize = 80;
pub const NVML_DEVICE_SERIAL_BUFFER_SIZE: usize = 30;
pub const NVML_DEVICE_VBIOS_VERSION_BUFFER_SIZE: usize = 32;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct nvmlMemory_t {
    pub total: c_ulong,
    pub free: c_ulong,
    pub used: c_ulong,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct nvmlUtilization_t {
    pub gpu: c_uint,
    pub memory: c_uint,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct nvmlProcessInfo_t {
    pub pid: c_uint,
    pub usedGpuMemory: c_ulong,
    pub gpuInstanceId: c_uint,
    pub computeInstanceId: c_uint,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct nvmlPciInfo_t {
    pub busIdLegacy: [c_char; 16],
    pub domain: c_uint,
    pub bus: c_uint,
    pub device: c_uint,
    pub pciDeviceId: c_uint,
    pub pciSubSystemId: c_uint,
    pub busId: [c_char; 32],
}

extern "C" {
    // Initialization
    pub fn nvmlInit() -> nvmlReturn_t;
    pub fn nvmlInit_v2() -> nvmlReturn_t;
    pub fn nvmlShutdown() -> nvmlReturn_t;
    pub fn nvmlErrorString(result: nvmlReturn_t) -> *const c_char;
    pub fn nvmlSystemGetDriverVersion(version: *mut c_char, length: c_uint) -> nvmlReturn_t;
    pub fn nvmlSystemGetNVMLVersion(version: *mut c_char, length: c_uint) -> nvmlReturn_t;
    pub fn nvmlSystemGetCudaDriverVersion(cudaDriverVersion: *mut c_int) -> nvmlReturn_t;
    pub fn nvmlSystemGetCudaDriverVersion_v2(cudaDriverVersion: *mut c_int) -> nvmlReturn_t;

    // Device queries
    pub fn nvmlDeviceGetCount(deviceCount: *mut c_uint) -> nvmlReturn_t;
    pub fn nvmlDeviceGetCount_v2(deviceCount: *mut c_uint) -> nvmlReturn_t;
    pub fn nvmlDeviceGetHandleByIndex(index: c_uint, device: *mut nvmlDevice_t) -> nvmlReturn_t;
    pub fn nvmlDeviceGetHandleByIndex_v2(index: c_uint, device: *mut nvmlDevice_t) -> nvmlReturn_t;
    pub fn nvmlDeviceGetHandleBySerial(serial: *const c_char, device: *mut nvmlDevice_t) -> nvmlReturn_t;
    pub fn nvmlDeviceGetHandleByUUID(uuid: *const c_char, device: *mut nvmlDevice_t) -> nvmlReturn_t;
    pub fn nvmlDeviceGetHandleByPciBusId(pciBusId: *const c_char, device: *mut nvmlDevice_t) -> nvmlReturn_t;
    pub fn nvmlDeviceGetHandleByPciBusId_v2(pciBusId: *const c_char, device: *mut nvmlDevice_t) -> nvmlReturn_t;

    // Device properties
    pub fn nvmlDeviceGetName(device: nvmlDevice_t, name: *mut c_char, length: c_uint) -> nvmlReturn_t;
    pub fn nvmlDeviceGetUUID(device: nvmlDevice_t, uuid: *mut c_char, length: c_uint) -> nvmlReturn_t;
    pub fn nvmlDeviceGetSerial(device: nvmlDevice_t, serial: *mut c_char, length: c_uint) -> nvmlReturn_t;
    pub fn nvmlDeviceGetIndex(device: nvmlDevice_t, index: *mut c_uint) -> nvmlReturn_t;
    pub fn nvmlDeviceGetPciInfo(device: nvmlDevice_t, pci: *mut nvmlPciInfo_t) -> nvmlReturn_t;
    pub fn nvmlDeviceGetPciInfo_v2(device: nvmlDevice_t, pci: *mut nvmlPciInfo_t) -> nvmlReturn_t;
    pub fn nvmlDeviceGetPciInfo_v3(device: nvmlDevice_t, pci: *mut nvmlPciInfo_t) -> nvmlReturn_t;
    pub fn nvmlDeviceGetCudaComputeCapability(device: nvmlDevice_t, major: *mut c_int, minor: *mut c_int) -> nvmlReturn_t;

    // Device status
    pub fn nvmlDeviceGetMemoryInfo(device: nvmlDevice_t, memory: *mut nvmlMemory_t) -> nvmlReturn_t;
    pub fn nvmlDeviceGetUtilizationRates(device: nvmlDevice_t, utilization: *mut nvmlUtilization_t) -> nvmlReturn_t;
    pub fn nvmlDeviceGetTemperature(device: nvmlDevice_t, sensorType: nvmlTemperatureSensors_t, temp: *mut c_uint) -> nvmlReturn_t;
    pub fn nvmlDeviceGetFanSpeed(device: nvmlDevice_t, speed: *mut c_uint) -> nvmlReturn_t;
    pub fn nvmlDeviceGetFanSpeed_v2(device: nvmlDevice_t, fan: c_uint, speed: *mut c_uint) -> nvmlReturn_t;
    pub fn nvmlDeviceGetPowerUsage(device: nvmlDevice_t, power: *mut c_uint) -> nvmlReturn_t;
    pub fn nvmlDeviceGetPowerManagementLimit(device: nvmlDevice_t, limit: *mut c_uint) -> nvmlReturn_t;
    pub fn nvmlDeviceGetClockInfo(device: nvmlDevice_t, type_: nvmlClockType_t, clock: *mut c_uint) -> nvmlReturn_t;
    pub fn nvmlDeviceGetMaxClockInfo(device: nvmlDevice_t, type_: nvmlClockType_t, clock: *mut c_uint) -> nvmlReturn_t;
    pub fn nvmlDeviceGetPerformanceState(device: nvmlDevice_t, pState: *mut nvmlPstates_t) -> nvmlReturn_t;

    // Process queries
    pub fn nvmlDeviceGetComputeRunningProcesses(
        device: nvmlDevice_t,
        infoCount: *mut c_uint,
        infos: *mut nvmlProcessInfo_t,
    ) -> nvmlReturn_t;
    pub fn nvmlDeviceGetGraphicsRunningProcesses(
        device: nvmlDevice_t,
        infoCount: *mut c_uint,
        infos: *mut nvmlProcessInfo_t,
    ) -> nvmlReturn_t;
}
