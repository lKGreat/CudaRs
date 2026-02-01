//! Raw FFI bindings to CUDA Driver API.
//!
//! This crate provides unsafe, low-level bindings to the CUDA Driver API.
//! For a safer interface, use the `cuda-driver` crate.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(clippy::all)]

use libc::{c_char, c_int, c_uint, c_ulong, c_void, size_t};

// ============================================================================
// Basic Types
// ============================================================================

pub type CUdevice = c_int;
pub type CUdeviceptr = c_ulong;

pub type CUresult = c_int;

pub const CUDA_SUCCESS: CUresult = 0;
pub const CUDA_ERROR_INVALID_VALUE: CUresult = 1;
pub const CUDA_ERROR_OUT_OF_MEMORY: CUresult = 2;
pub const CUDA_ERROR_NOT_INITIALIZED: CUresult = 3;
pub const CUDA_ERROR_DEINITIALIZED: CUresult = 4;
pub const CUDA_ERROR_PROFILER_DISABLED: CUresult = 5;
pub const CUDA_ERROR_NO_DEVICE: CUresult = 100;
pub const CUDA_ERROR_INVALID_DEVICE: CUresult = 101;
pub const CUDA_ERROR_DEVICE_NOT_LICENSED: CUresult = 102;
pub const CUDA_ERROR_INVALID_IMAGE: CUresult = 200;
pub const CUDA_ERROR_INVALID_CONTEXT: CUresult = 201;
pub const CUDA_ERROR_CONTEXT_ALREADY_CURRENT: CUresult = 202;
pub const CUDA_ERROR_MAP_FAILED: CUresult = 205;
pub const CUDA_ERROR_UNMAP_FAILED: CUresult = 206;
pub const CUDA_ERROR_ARRAY_IS_MAPPED: CUresult = 207;
pub const CUDA_ERROR_ALREADY_MAPPED: CUresult = 208;
pub const CUDA_ERROR_NO_BINARY_FOR_GPU: CUresult = 209;
pub const CUDA_ERROR_ALREADY_ACQUIRED: CUresult = 210;
pub const CUDA_ERROR_NOT_MAPPED: CUresult = 211;
pub const CUDA_ERROR_NOT_MAPPED_AS_ARRAY: CUresult = 212;
pub const CUDA_ERROR_NOT_MAPPED_AS_POINTER: CUresult = 213;
pub const CUDA_ERROR_ECC_UNCORRECTABLE: CUresult = 214;
pub const CUDA_ERROR_UNSUPPORTED_LIMIT: CUresult = 215;
pub const CUDA_ERROR_CONTEXT_ALREADY_IN_USE: CUresult = 216;
pub const CUDA_ERROR_PEER_ACCESS_UNSUPPORTED: CUresult = 217;
pub const CUDA_ERROR_INVALID_PTX: CUresult = 218;
pub const CUDA_ERROR_INVALID_GRAPHICS_CONTEXT: CUresult = 219;
pub const CUDA_ERROR_NVLINK_UNCORRECTABLE: CUresult = 220;
pub const CUDA_ERROR_JIT_COMPILER_NOT_FOUND: CUresult = 221;
pub const CUDA_ERROR_INVALID_SOURCE: CUresult = 300;
pub const CUDA_ERROR_FILE_NOT_FOUND: CUresult = 301;
pub const CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND: CUresult = 302;
pub const CUDA_ERROR_SHARED_OBJECT_INIT_FAILED: CUresult = 303;
pub const CUDA_ERROR_OPERATING_SYSTEM: CUresult = 304;
pub const CUDA_ERROR_INVALID_HANDLE: CUresult = 400;
pub const CUDA_ERROR_ILLEGAL_STATE: CUresult = 401;
pub const CUDA_ERROR_NOT_FOUND: CUresult = 500;
pub const CUDA_ERROR_NOT_READY: CUresult = 600;
pub const CUDA_ERROR_ILLEGAL_ADDRESS: CUresult = 700;
pub const CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: CUresult = 701;
pub const CUDA_ERROR_LAUNCH_TIMEOUT: CUresult = 702;
pub const CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: CUresult = 703;
pub const CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED: CUresult = 704;
pub const CUDA_ERROR_PEER_ACCESS_NOT_ENABLED: CUresult = 705;
pub const CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE: CUresult = 708;
pub const CUDA_ERROR_CONTEXT_IS_DESTROYED: CUresult = 709;
pub const CUDA_ERROR_ASSERT: CUresult = 710;
pub const CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED: CUresult = 712;
pub const CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED: CUresult = 713;
pub const CUDA_ERROR_HARDWARE_STACK_ERROR: CUresult = 714;
pub const CUDA_ERROR_ILLEGAL_INSTRUCTION: CUresult = 715;
pub const CUDA_ERROR_MISALIGNED_ADDRESS: CUresult = 716;
pub const CUDA_ERROR_INVALID_ADDRESS_SPACE: CUresult = 717;
pub const CUDA_ERROR_INVALID_PC: CUresult = 718;
pub const CUDA_ERROR_LAUNCH_FAILED: CUresult = 719;
pub const CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE: CUresult = 720;
pub const CUDA_ERROR_NOT_PERMITTED: CUresult = 800;
pub const CUDA_ERROR_NOT_SUPPORTED: CUresult = 801;
pub const CUDA_ERROR_SYSTEM_NOT_READY: CUresult = 802;
pub const CUDA_ERROR_SYSTEM_DRIVER_MISMATCH: CUresult = 803;
pub const CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE: CUresult = 804;
pub const CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED: CUresult = 900;
pub const CUDA_ERROR_STREAM_CAPTURE_INVALIDATED: CUresult = 901;
pub const CUDA_ERROR_STREAM_CAPTURE_MERGE: CUresult = 902;
pub const CUDA_ERROR_STREAM_CAPTURE_UNMATCHED: CUresult = 903;
pub const CUDA_ERROR_STREAM_CAPTURE_UNJOINED: CUresult = 904;
pub const CUDA_ERROR_STREAM_CAPTURE_ISOLATION: CUresult = 905;
pub const CUDA_ERROR_STREAM_CAPTURE_IMPLICIT: CUresult = 906;
pub const CUDA_ERROR_CAPTURED_EVENT: CUresult = 907;
pub const CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD: CUresult = 908;
pub const CUDA_ERROR_TIMEOUT: CUresult = 909;
pub const CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE: CUresult = 910;
pub const CUDA_ERROR_UNKNOWN: CUresult = 999;

// ============================================================================
// Opaque Types
// ============================================================================

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUctx_st {
    _unused: [u8; 0],
}
pub type CUcontext = *mut CUctx_st;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUmod_st {
    _unused: [u8; 0],
}
pub type CUmodule = *mut CUmod_st;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUfunc_st {
    _unused: [u8; 0],
}
pub type CUfunction = *mut CUfunc_st;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUstream_st {
    _unused: [u8; 0],
}
pub type CUstream = *mut CUstream_st;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUevent_st {
    _unused: [u8; 0],
}
pub type CUevent = *mut CUevent_st;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUarray_st {
    _unused: [u8; 0],
}
pub type CUarray = *mut CUarray_st;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUtexref_st {
    _unused: [u8; 0],
}
pub type CUtexref = *mut CUtexref_st;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUsurfref_st {
    _unused: [u8; 0],
}
pub type CUsurfref = *mut CUsurfref_st;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUlinkState_st {
    _unused: [u8; 0],
}
pub type CUlinkState = *mut CUlinkState_st;

// ============================================================================
// Device Attributes
// ============================================================================

pub type CUdevice_attribute = c_int;

pub const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: CUdevice_attribute = 1;
pub const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X: CUdevice_attribute = 2;
pub const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y: CUdevice_attribute = 3;
pub const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z: CUdevice_attribute = 4;
pub const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X: CUdevice_attribute = 5;
pub const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y: CUdevice_attribute = 6;
pub const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z: CUdevice_attribute = 7;
pub const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: CUdevice_attribute = 8;
pub const CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY: CUdevice_attribute = 9;
pub const CU_DEVICE_ATTRIBUTE_WARP_SIZE: CUdevice_attribute = 10;
pub const CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK: CUdevice_attribute = 12;
pub const CU_DEVICE_ATTRIBUTE_CLOCK_RATE: CUdevice_attribute = 13;
pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: CUdevice_attribute = 16;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: CUdevice_attribute = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: CUdevice_attribute = 76;
pub const CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR: CUdevice_attribute = 106;

// ============================================================================
// Context Flags
// ============================================================================

pub type CUctx_flags = c_uint;

pub const CU_CTX_SCHED_AUTO: CUctx_flags = 0x00;
pub const CU_CTX_SCHED_SPIN: CUctx_flags = 0x01;
pub const CU_CTX_SCHED_YIELD: CUctx_flags = 0x02;
pub const CU_CTX_SCHED_BLOCKING_SYNC: CUctx_flags = 0x04;
pub const CU_CTX_MAP_HOST: CUctx_flags = 0x08;
pub const CU_CTX_LMEM_RESIZE_TO_MAX: CUctx_flags = 0x10;

// ============================================================================
// Stream Flags
// ============================================================================

pub const CU_STREAM_DEFAULT: c_uint = 0x0;
pub const CU_STREAM_NON_BLOCKING: c_uint = 0x1;

// ============================================================================
// Event Flags
// ============================================================================

pub const CU_EVENT_DEFAULT: c_uint = 0x0;
pub const CU_EVENT_BLOCKING_SYNC: c_uint = 0x1;
pub const CU_EVENT_DISABLE_TIMING: c_uint = 0x2;
pub const CU_EVENT_INTERPROCESS: c_uint = 0x4;

// ============================================================================
// JIT Options
// ============================================================================

pub type CUjit_option = c_int;

pub const CU_JIT_MAX_REGISTERS: CUjit_option = 0;
pub const CU_JIT_THREADS_PER_BLOCK: CUjit_option = 1;
pub const CU_JIT_WALL_TIME: CUjit_option = 2;
pub const CU_JIT_INFO_LOG_BUFFER: CUjit_option = 3;
pub const CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: CUjit_option = 4;
pub const CU_JIT_ERROR_LOG_BUFFER: CUjit_option = 5;
pub const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: CUjit_option = 6;
pub const CU_JIT_OPTIMIZATION_LEVEL: CUjit_option = 7;
pub const CU_JIT_TARGET_FROM_CUCONTEXT: CUjit_option = 8;
pub const CU_JIT_TARGET: CUjit_option = 9;
pub const CU_JIT_FALLBACK_STRATEGY: CUjit_option = 10;
pub const CU_JIT_GENERATE_DEBUG_INFO: CUjit_option = 11;
pub const CU_JIT_LOG_VERBOSE: CUjit_option = 12;
pub const CU_JIT_GENERATE_LINE_INFO: CUjit_option = 13;
pub const CU_JIT_CACHE_MODE: CUjit_option = 14;

// ============================================================================
// External Functions - Initialization
// ============================================================================

extern "C" {
    pub fn cuInit(Flags: c_uint) -> CUresult;
    pub fn cuDriverGetVersion(driverVersion: *mut c_int) -> CUresult;
}

// ============================================================================
// External Functions - Device Management
// ============================================================================

extern "C" {
    pub fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
    pub fn cuDeviceGetCount(count: *mut c_int) -> CUresult;
    pub fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult;
    pub fn cuDeviceTotalMem(bytes: *mut size_t, dev: CUdevice) -> CUresult;
    pub fn cuDeviceGetAttribute(
        pi: *mut c_int,
        attrib: CUdevice_attribute,
        dev: CUdevice,
    ) -> CUresult;
    pub fn cuDeviceGetUuid(uuid: *mut [c_char; 16], dev: CUdevice) -> CUresult;
    pub fn cuDeviceComputeCapability(
        major: *mut c_int,
        minor: *mut c_int,
        dev: CUdevice,
    ) -> CUresult;
}

// ============================================================================
// External Functions - Context Management
// ============================================================================

extern "C" {
    pub fn cuCtxCreate(pctx: *mut CUcontext, flags: c_uint, dev: CUdevice) -> CUresult;
    pub fn cuCtxDestroy(ctx: CUcontext) -> CUresult;
    pub fn cuCtxPushCurrent(ctx: CUcontext) -> CUresult;
    pub fn cuCtxPopCurrent(pctx: *mut CUcontext) -> CUresult;
    pub fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
    pub fn cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult;
    pub fn cuCtxGetDevice(device: *mut CUdevice) -> CUresult;
    pub fn cuCtxSynchronize() -> CUresult;
    pub fn cuCtxGetApiVersion(ctx: CUcontext, version: *mut c_uint) -> CUresult;
}

// ============================================================================
// External Functions - Primary Context Management
// ============================================================================

extern "C" {
    pub fn cuDevicePrimaryCtxRetain(pctx: *mut CUcontext, dev: CUdevice) -> CUresult;
    pub fn cuDevicePrimaryCtxRelease(dev: CUdevice) -> CUresult;
    pub fn cuDevicePrimaryCtxSetFlags(dev: CUdevice, flags: c_uint) -> CUresult;
    pub fn cuDevicePrimaryCtxGetState(
        dev: CUdevice,
        flags: *mut c_uint,
        active: *mut c_int,
    ) -> CUresult;
    pub fn cuDevicePrimaryCtxReset(dev: CUdevice) -> CUresult;
}

// ============================================================================
// External Functions - Module Management
// ============================================================================

extern "C" {
    pub fn cuModuleLoad(module: *mut CUmodule, fname: *const c_char) -> CUresult;
    pub fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;
    pub fn cuModuleLoadDataEx(
        module: *mut CUmodule,
        image: *const c_void,
        numOptions: c_uint,
        options: *mut CUjit_option,
        optionValues: *mut *mut c_void,
    ) -> CUresult;
    pub fn cuModuleLoadFatBinary(module: *mut CUmodule, fatCubin: *const c_void) -> CUresult;
    pub fn cuModuleUnload(hmod: CUmodule) -> CUresult;
    pub fn cuModuleGetFunction(
        hfunc: *mut CUfunction,
        hmod: CUmodule,
        name: *const c_char,
    ) -> CUresult;
    pub fn cuModuleGetGlobal(
        dptr: *mut CUdeviceptr,
        bytes: *mut size_t,
        hmod: CUmodule,
        name: *const c_char,
    ) -> CUresult;
}

// ============================================================================
// External Functions - Memory Management
// ============================================================================

extern "C" {
    pub fn cuMemAlloc(dptr: *mut CUdeviceptr, bytesize: size_t) -> CUresult;
    pub fn cuMemAllocPitch(
        dptr: *mut CUdeviceptr,
        pPitch: *mut size_t,
        WidthInBytes: size_t,
        Height: size_t,
        ElementSizeBytes: c_uint,
    ) -> CUresult;
    pub fn cuMemFree(dptr: CUdeviceptr) -> CUresult;
    pub fn cuMemGetInfo(free: *mut size_t, total: *mut size_t) -> CUresult;
    pub fn cuMemAllocHost(pp: *mut *mut c_void, bytesize: size_t) -> CUresult;
    pub fn cuMemFreeHost(p: *mut c_void) -> CUresult;
    pub fn cuMemHostAlloc(pp: *mut *mut c_void, bytesize: size_t, Flags: c_uint) -> CUresult;
    pub fn cuMemHostGetDevicePointer(
        pdptr: *mut CUdeviceptr,
        p: *mut c_void,
        Flags: c_uint,
    ) -> CUresult;
    pub fn cuMemHostRegister(p: *mut c_void, bytesize: size_t, Flags: c_uint) -> CUresult;
    pub fn cuMemHostUnregister(p: *mut c_void) -> CUresult;
    pub fn cuMemcpy(dst: CUdeviceptr, src: CUdeviceptr, ByteCount: size_t) -> CUresult;
    pub fn cuMemcpyHtoD(dstDevice: CUdeviceptr, srcHost: *const c_void, ByteCount: size_t) -> CUresult;
    pub fn cuMemcpyDtoH(dstHost: *mut c_void, srcDevice: CUdeviceptr, ByteCount: size_t) -> CUresult;
    pub fn cuMemcpyDtoD(dstDevice: CUdeviceptr, srcDevice: CUdeviceptr, ByteCount: size_t) -> CUresult;
    pub fn cuMemcpyHtoDAsync(
        dstDevice: CUdeviceptr,
        srcHost: *const c_void,
        ByteCount: size_t,
        hStream: CUstream,
    ) -> CUresult;
    pub fn cuMemcpyDtoHAsync(
        dstHost: *mut c_void,
        srcDevice: CUdeviceptr,
        ByteCount: size_t,
        hStream: CUstream,
    ) -> CUresult;
    pub fn cuMemcpyDtoDAsync(
        dstDevice: CUdeviceptr,
        srcDevice: CUdeviceptr,
        ByteCount: size_t,
        hStream: CUstream,
    ) -> CUresult;
    pub fn cuMemsetD8(dstDevice: CUdeviceptr, uc: c_char, N: size_t) -> CUresult;
    pub fn cuMemsetD16(dstDevice: CUdeviceptr, us: u16, N: size_t) -> CUresult;
    pub fn cuMemsetD32(dstDevice: CUdeviceptr, ui: c_uint, N: size_t) -> CUresult;
    pub fn cuMemsetD8Async(dstDevice: CUdeviceptr, uc: c_char, N: size_t, hStream: CUstream) -> CUresult;
    pub fn cuMemsetD16Async(dstDevice: CUdeviceptr, us: u16, N: size_t, hStream: CUstream) -> CUresult;
    pub fn cuMemsetD32Async(dstDevice: CUdeviceptr, ui: c_uint, N: size_t, hStream: CUstream) -> CUresult;
}

// ============================================================================
// External Functions - Unified Addressing
// ============================================================================

extern "C" {
    pub fn cuMemAllocManaged(dptr: *mut CUdeviceptr, bytesize: size_t, flags: c_uint) -> CUresult;
    pub fn cuMemPrefetchAsync(
        devPtr: CUdeviceptr,
        count: size_t,
        dstDevice: CUdevice,
        hStream: CUstream,
    ) -> CUresult;
    pub fn cuMemAdvise(
        devPtr: CUdeviceptr,
        count: size_t,
        advice: c_int,
        device: CUdevice,
    ) -> CUresult;
    pub fn cuPointerGetAttribute(
        data: *mut c_void,
        attribute: c_int,
        ptr: CUdeviceptr,
    ) -> CUresult;
    pub fn cuPointerGetAttributes(
        numAttributes: c_uint,
        attributes: *mut c_int,
        data: *mut *mut c_void,
        ptr: CUdeviceptr,
    ) -> CUresult;
}

pub const CU_MEM_ATTACH_GLOBAL: c_uint = 0x1;
pub const CU_MEM_ATTACH_HOST: c_uint = 0x2;
pub const CU_MEM_ATTACH_SINGLE: c_uint = 0x4;

// ============================================================================
// External Functions - Stream Management
// ============================================================================

extern "C" {
    pub fn cuStreamCreate(phStream: *mut CUstream, Flags: c_uint) -> CUresult;
    pub fn cuStreamCreateWithPriority(
        phStream: *mut CUstream,
        flags: c_uint,
        priority: c_int,
    ) -> CUresult;
    pub fn cuStreamDestroy(hStream: CUstream) -> CUresult;
    pub fn cuStreamSynchronize(hStream: CUstream) -> CUresult;
    pub fn cuStreamQuery(hStream: CUstream) -> CUresult;
    pub fn cuStreamWaitEvent(hStream: CUstream, hEvent: CUevent, Flags: c_uint) -> CUresult;
    pub fn cuStreamGetPriority(hStream: CUstream, priority: *mut c_int) -> CUresult;
    pub fn cuStreamGetFlags(hStream: CUstream, flags: *mut c_uint) -> CUresult;
    pub fn cuStreamGetCtx(hStream: CUstream, pctx: *mut CUcontext) -> CUresult;
}

// ============================================================================
// External Functions - Event Management
// ============================================================================

extern "C" {
    pub fn cuEventCreate(phEvent: *mut CUevent, Flags: c_uint) -> CUresult;
    pub fn cuEventDestroy(hEvent: CUevent) -> CUresult;
    pub fn cuEventRecord(hEvent: CUevent, hStream: CUstream) -> CUresult;
    pub fn cuEventSynchronize(hEvent: CUevent) -> CUresult;
    pub fn cuEventQuery(hEvent: CUevent) -> CUresult;
    pub fn cuEventElapsedTime(pMilliseconds: *mut f32, hStart: CUevent, hEnd: CUevent) -> CUresult;
}

// ============================================================================
// External Functions - Execution Control
// ============================================================================

extern "C" {
    pub fn cuFuncGetAttribute(pi: *mut c_int, attrib: c_int, hfunc: CUfunction) -> CUresult;
    pub fn cuFuncSetAttribute(hfunc: CUfunction, attrib: c_int, value: c_int) -> CUresult;
    pub fn cuFuncSetCacheConfig(hfunc: CUfunction, config: c_int) -> CUresult;
    pub fn cuFuncSetSharedMemConfig(hfunc: CUfunction, config: c_int) -> CUresult;
    pub fn cuLaunchKernel(
        f: CUfunction,
        gridDimX: c_uint,
        gridDimY: c_uint,
        gridDimZ: c_uint,
        blockDimX: c_uint,
        blockDimY: c_uint,
        blockDimZ: c_uint,
        sharedMemBytes: c_uint,
        hStream: CUstream,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;
    pub fn cuLaunchCooperativeKernel(
        f: CUfunction,
        gridDimX: c_uint,
        gridDimY: c_uint,
        gridDimZ: c_uint,
        blockDimX: c_uint,
        blockDimY: c_uint,
        blockDimZ: c_uint,
        sharedMemBytes: c_uint,
        hStream: CUstream,
        kernelParams: *mut *mut c_void,
    ) -> CUresult;
}

// ============================================================================
// External Functions - Linker
// ============================================================================

pub type CUjitInputType = c_int;

pub const CU_JIT_INPUT_CUBIN: CUjitInputType = 0;
pub const CU_JIT_INPUT_PTX: CUjitInputType = 1;
pub const CU_JIT_INPUT_FATBINARY: CUjitInputType = 2;
pub const CU_JIT_INPUT_OBJECT: CUjitInputType = 3;
pub const CU_JIT_INPUT_LIBRARY: CUjitInputType = 4;
pub const CU_JIT_INPUT_NVVM: CUjitInputType = 5;

extern "C" {
    pub fn cuLinkCreate(
        numOptions: c_uint,
        options: *mut CUjit_option,
        optionValues: *mut *mut c_void,
        stateOut: *mut CUlinkState,
    ) -> CUresult;
    pub fn cuLinkAddData(
        state: CUlinkState,
        type_: CUjitInputType,
        data: *mut c_void,
        size: size_t,
        name: *const c_char,
        numOptions: c_uint,
        options: *mut CUjit_option,
        optionValues: *mut *mut c_void,
    ) -> CUresult;
    pub fn cuLinkAddFile(
        state: CUlinkState,
        type_: CUjitInputType,
        path: *const c_char,
        numOptions: c_uint,
        options: *mut CUjit_option,
        optionValues: *mut *mut c_void,
    ) -> CUresult;
    pub fn cuLinkComplete(
        state: CUlinkState,
        cubinOut: *mut *mut c_void,
        sizeOut: *mut size_t,
    ) -> CUresult;
    pub fn cuLinkDestroy(state: CUlinkState) -> CUresult;
}

// ============================================================================
// External Functions - Occupancy
// ============================================================================

extern "C" {
    pub fn cuOccupancyMaxActiveBlocksPerMultiprocessor(
        numBlocks: *mut c_int,
        func: CUfunction,
        blockSize: c_int,
        dynamicSMemSize: size_t,
    ) -> CUresult;
    pub fn cuOccupancyMaxPotentialBlockSize(
        minGridSize: *mut c_int,
        blockSize: *mut c_int,
        func: CUfunction,
        blockSizeToDynamicSMemSize: Option<unsafe extern "C" fn(blockSize: c_int) -> size_t>,
        dynamicSMemSize: size_t,
        blockSizeLimit: c_int,
    ) -> CUresult;
}

// ============================================================================
// External Functions - Peer Context Memory Access
// ============================================================================

extern "C" {
    pub fn cuDeviceCanAccessPeer(
        canAccessPeer: *mut c_int,
        dev: CUdevice,
        peerDev: CUdevice,
    ) -> CUresult;
    pub fn cuCtxEnablePeerAccess(peerContext: CUcontext, Flags: c_uint) -> CUresult;
    pub fn cuCtxDisablePeerAccess(peerContext: CUcontext) -> CUresult;
    pub fn cuMemcpyPeer(
        dstDevice: CUdeviceptr,
        dstContext: CUcontext,
        srcDevice: CUdeviceptr,
        srcContext: CUcontext,
        ByteCount: size_t,
    ) -> CUresult;
    pub fn cuMemcpyPeerAsync(
        dstDevice: CUdeviceptr,
        dstContext: CUcontext,
        srcDevice: CUdeviceptr,
        srcContext: CUcontext,
        ByteCount: size_t,
        hStream: CUstream,
    ) -> CUresult;
}

// ============================================================================
// External Functions - Error Handling
// ============================================================================

extern "C" {
    pub fn cuGetErrorName(error: CUresult, pStr: *mut *const c_char) -> CUresult;
    pub fn cuGetErrorString(error: CUresult, pStr: *mut *const c_char) -> CUresult;
}

// ============================================================================
// CUDA 12.3+ Features
// ============================================================================

#[cfg(feature = "cuda-12-3")]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUgraph_st {
    _unused: [u8; 0],
}
#[cfg(feature = "cuda-12-3")]
pub type CUgraph = *mut CUgraph_st;

#[cfg(feature = "cuda-12-3")]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUgraphExec_st {
    _unused: [u8; 0],
}
#[cfg(feature = "cuda-12-3")]
pub type CUgraphExec = *mut CUgraphExec_st;

#[cfg(feature = "cuda-12-3")]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUlib_st {
    _unused: [u8; 0],
}
#[cfg(feature = "cuda-12-3")]
pub type CUlibrary = *mut CUlib_st;

#[cfg(feature = "cuda-12-3")]
extern "C" {
    pub fn cuLibraryLoadData(
        library: *mut CUlibrary,
        code: *const c_void,
        jitOptions: *mut CUjit_option,
        jitOptionsValues: *mut *mut c_void,
        numJitOptions: c_uint,
        libraryOptions: *mut c_void,
        libraryOptionValues: *mut *mut c_void,
        numLibraryOptions: c_uint,
    ) -> CUresult;
    pub fn cuLibraryUnload(library: CUlibrary) -> CUresult;
    pub fn cuLibraryGetKernel(pKernel: *mut CUfunction, library: CUlibrary, name: *const c_char) -> CUresult;
}
