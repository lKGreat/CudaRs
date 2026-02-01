//! Raw FFI bindings to CUDA Runtime API.
//!
//! This crate provides unsafe, low-level bindings to the CUDA Runtime API.
//! For a safer interface, use the `cuda-runtime` crate.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(clippy::all)]

use libc::{c_char, c_int, c_uint, c_void, size_t};

// ============================================================================
// Error Types
// ============================================================================

pub type cudaError_t = c_int;

pub const cudaSuccess: cudaError_t = 0;
pub const cudaErrorInvalidValue: cudaError_t = 1;
pub const cudaErrorMemoryAllocation: cudaError_t = 2;
pub const cudaErrorInitializationError: cudaError_t = 3;
pub const cudaErrorCudartUnloading: cudaError_t = 4;
pub const cudaErrorProfilerDisabled: cudaError_t = 5;
pub const cudaErrorInvalidConfiguration: cudaError_t = 9;
pub const cudaErrorInvalidPitchValue: cudaError_t = 12;
pub const cudaErrorInvalidSymbol: cudaError_t = 13;
pub const cudaErrorInvalidDevicePointer: cudaError_t = 17;
pub const cudaErrorInvalidMemcpyDirection: cudaError_t = 21;
pub const cudaErrorInsufficientDriver: cudaError_t = 35;
pub const cudaErrorNoDevice: cudaError_t = 100;
pub const cudaErrorInvalidDevice: cudaError_t = 101;
pub const cudaErrorDeviceNotLicensed: cudaError_t = 102;
pub const cudaErrorSoftwareValidityNotEstablished: cudaError_t = 103;
pub const cudaErrorStartupFailure: cudaError_t = 127;
pub const cudaErrorInvalidKernelImage: cudaError_t = 200;
pub const cudaErrorDeviceUninitialized: cudaError_t = 201;
pub const cudaErrorNotReady: cudaError_t = 600;
pub const cudaErrorIllegalAddress: cudaError_t = 700;
pub const cudaErrorLaunchOutOfResources: cudaError_t = 701;
pub const cudaErrorLaunchTimeout: cudaError_t = 702;
pub const cudaErrorPeerAccessAlreadyEnabled: cudaError_t = 704;
pub const cudaErrorPeerAccessNotEnabled: cudaError_t = 705;
pub const cudaErrorAssert: cudaError_t = 710;
pub const cudaErrorHostMemoryAlreadyRegistered: cudaError_t = 712;
pub const cudaErrorHostMemoryNotRegistered: cudaError_t = 713;
pub const cudaErrorHardwareStackError: cudaError_t = 714;
pub const cudaErrorIllegalInstruction: cudaError_t = 715;
pub const cudaErrorMisalignedAddress: cudaError_t = 716;
pub const cudaErrorInvalidAddressSpace: cudaError_t = 717;
pub const cudaErrorInvalidPc: cudaError_t = 718;
pub const cudaErrorLaunchFailure: cudaError_t = 719;
pub const cudaErrorCooperativeLaunchTooLarge: cudaError_t = 720;
pub const cudaErrorSystemNotReady: cudaError_t = 802;
pub const cudaErrorSystemDriverMismatch: cudaError_t = 803;
pub const cudaErrorStreamCaptureUnsupported: cudaError_t = 900;
pub const cudaErrorStreamCaptureInvalidated: cudaError_t = 901;
pub const cudaErrorStreamCaptureMerge: cudaError_t = 902;
pub const cudaErrorStreamCaptureUnmatched: cudaError_t = 903;
pub const cudaErrorStreamCaptureUnjoined: cudaError_t = 904;
pub const cudaErrorStreamCaptureIsolation: cudaError_t = 905;
pub const cudaErrorStreamCaptureImplicit: cudaError_t = 906;
pub const cudaErrorCapturedEvent: cudaError_t = 907;
pub const cudaErrorStreamCaptureWrongThread: cudaError_t = 908;
pub const cudaErrorTimeout: cudaError_t = 909;
pub const cudaErrorGraphExecUpdateFailure: cudaError_t = 910;
pub const cudaErrorUnknown: cudaError_t = 999;

// ============================================================================
// Memory Copy Direction
// ============================================================================

pub type cudaMemcpyKind = c_int;

pub const cudaMemcpyHostToHost: cudaMemcpyKind = 0;
pub const cudaMemcpyHostToDevice: cudaMemcpyKind = 1;
pub const cudaMemcpyDeviceToHost: cudaMemcpyKind = 2;
pub const cudaMemcpyDeviceToDevice: cudaMemcpyKind = 3;
pub const cudaMemcpyDefault: cudaMemcpyKind = 4;

// ============================================================================
// Device Properties
// ============================================================================

pub type cudaDeviceAttr = c_int;

pub const cudaDevAttrMaxThreadsPerBlock: cudaDeviceAttr = 1;
pub const cudaDevAttrMaxBlockDimX: cudaDeviceAttr = 2;
pub const cudaDevAttrMaxBlockDimY: cudaDeviceAttr = 3;
pub const cudaDevAttrMaxBlockDimZ: cudaDeviceAttr = 4;
pub const cudaDevAttrMaxGridDimX: cudaDeviceAttr = 5;
pub const cudaDevAttrMaxGridDimY: cudaDeviceAttr = 6;
pub const cudaDevAttrMaxGridDimZ: cudaDeviceAttr = 7;
pub const cudaDevAttrMaxSharedMemoryPerBlock: cudaDeviceAttr = 8;
pub const cudaDevAttrTotalConstantMemory: cudaDeviceAttr = 9;
pub const cudaDevAttrWarpSize: cudaDeviceAttr = 10;
pub const cudaDevAttrMaxRegistersPerBlock: cudaDeviceAttr = 12;
pub const cudaDevAttrClockRate: cudaDeviceAttr = 13;
pub const cudaDevAttrMultiProcessorCount: cudaDeviceAttr = 16;
pub const cudaDevAttrComputeCapabilityMajor: cudaDeviceAttr = 75;
pub const cudaDevAttrComputeCapabilityMinor: cudaDeviceAttr = 76;
pub const cudaDevAttrMaxBlocksPerMultiProcessor: cudaDeviceAttr = 106;

// ============================================================================
// Opaque Types
// ============================================================================

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUstream_st {
    _unused: [u8; 0],
}
pub type cudaStream_t = *mut CUstream_st;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUevent_st {
    _unused: [u8; 0],
}
pub type cudaEvent_t = *mut CUevent_st;

// ============================================================================
// Structures
// ============================================================================

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct dim3 {
    pub x: c_uint,
    pub y: c_uint,
    pub z: c_uint,
}

impl Default for dim3 {
    fn default() -> Self {
        Self { x: 1, y: 1, z: 1 }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudaDeviceProp {
    pub name: [c_char; 256],
    pub uuid: [c_char; 16],
    pub totalGlobalMem: size_t,
    pub sharedMemPerBlock: size_t,
    pub regsPerBlock: c_int,
    pub warpSize: c_int,
    pub memPitch: size_t,
    pub maxThreadsPerBlock: c_int,
    pub maxThreadsDim: [c_int; 3],
    pub maxGridSize: [c_int; 3],
    pub clockRate: c_int,
    pub totalConstMem: size_t,
    pub major: c_int,
    pub minor: c_int,
    pub textureAlignment: size_t,
    pub texturePitchAlignment: size_t,
    pub deviceOverlap: c_int,
    pub multiProcessorCount: c_int,
    pub kernelExecTimeoutEnabled: c_int,
    pub integrated: c_int,
    pub canMapHostMemory: c_int,
    pub computeMode: c_int,
    pub maxTexture1D: c_int,
    pub maxTexture1DMipmap: c_int,
    pub maxTexture1DLinear: c_int,
    pub maxTexture2D: [c_int; 2],
    pub maxTexture2DMipmap: [c_int; 2],
    pub maxTexture2DLinear: [c_int; 3],
    pub maxTexture2DGather: [c_int; 2],
    pub maxTexture3D: [c_int; 3],
    pub maxTexture3DAlt: [c_int; 3],
    pub maxTextureCubemap: c_int,
    pub maxTexture1DLayered: [c_int; 2],
    pub maxTexture2DLayered: [c_int; 3],
    pub maxTextureCubemapLayered: [c_int; 2],
    pub maxSurface1D: c_int,
    pub maxSurface2D: [c_int; 2],
    pub maxSurface3D: [c_int; 3],
    pub maxSurface1DLayered: [c_int; 2],
    pub maxSurface2DLayered: [c_int; 3],
    pub maxSurfaceCubemap: c_int,
    pub maxSurfaceCubemapLayered: [c_int; 2],
    pub surfaceAlignment: size_t,
    pub concurrentKernels: c_int,
    pub ECCEnabled: c_int,
    pub pciBusID: c_int,
    pub pciDeviceID: c_int,
    pub pciDomainID: c_int,
    pub tccDriver: c_int,
    pub asyncEngineCount: c_int,
    pub unifiedAddressing: c_int,
    pub memoryClockRate: c_int,
    pub memoryBusWidth: c_int,
    pub l2CacheSize: c_int,
    pub persistingL2CacheMaxSize: c_int,
    pub maxThreadsPerMultiProcessor: c_int,
    pub streamPrioritiesSupported: c_int,
    pub globalL1CacheSupported: c_int,
    pub localL1CacheSupported: c_int,
    pub sharedMemPerMultiprocessor: size_t,
    pub regsPerMultiprocessor: c_int,
    pub managedMemory: c_int,
    pub isMultiGpuBoard: c_int,
    pub multiGpuBoardGroupID: c_int,
    pub hostNativeAtomicSupported: c_int,
    pub singleToDoublePrecisionPerfRatio: c_int,
    pub pageableMemoryAccess: c_int,
    pub concurrentManagedAccess: c_int,
    pub computePreemptionSupported: c_int,
    pub canUseHostPointerForRegisteredMem: c_int,
    pub cooperativeLaunch: c_int,
    pub cooperativeMultiDeviceLaunch: c_int,
    pub sharedMemPerBlockOptin: size_t,
    pub pageableMemoryAccessUsesHostPageTables: c_int,
    pub directManagedMemAccessFromHost: c_int,
    pub maxBlocksPerMultiProcessor: c_int,
    pub accessPolicyMaxWindowSize: c_int,
    pub reservedSharedMemPerBlock: size_t,
}

// ============================================================================
// External Functions - Device Management
// ============================================================================

#[cfg(not(feature = "stub"))]
extern "C" {
    pub fn cudaGetDeviceCount(count: *mut c_int) -> cudaError_t;
    pub fn cudaGetDevice(device: *mut c_int) -> cudaError_t;
    pub fn cudaSetDevice(device: c_int) -> cudaError_t;
    pub fn cudaGetDeviceProperties(prop: *mut cudaDeviceProp, device: c_int) -> cudaError_t;
    pub fn cudaDeviceGetAttribute(
        value: *mut c_int,
        attr: cudaDeviceAttr,
        device: c_int,
    ) -> cudaError_t;
    pub fn cudaDeviceSynchronize() -> cudaError_t;
    pub fn cudaDeviceReset() -> cudaError_t;
    pub fn cudaDeviceSetCacheConfig(cacheConfig: c_int) -> cudaError_t;
    pub fn cudaDeviceGetCacheConfig(pCacheConfig: *mut c_int) -> cudaError_t;
    pub fn cudaDeviceSetLimit(limit: c_int, value: size_t) -> cudaError_t;
    pub fn cudaDeviceGetLimit(pValue: *mut size_t, limit: c_int) -> cudaError_t;
}

// ============================================================================
// External Functions - Memory Management
// ============================================================================

#[cfg(not(feature = "stub"))]
extern "C" {
    pub fn cudaMalloc(devPtr: *mut *mut c_void, size: size_t) -> cudaError_t;
    pub fn cudaMallocHost(ptr: *mut *mut c_void, size: size_t) -> cudaError_t;
    pub fn cudaMallocManaged(devPtr: *mut *mut c_void, size: size_t, flags: c_uint) -> cudaError_t;
    pub fn cudaMallocPitch(
        devPtr: *mut *mut c_void,
        pitch: *mut size_t,
        width: size_t,
        height: size_t,
    ) -> cudaError_t;
    pub fn cudaMalloc3D(
        pitchedDevPtr: *mut cudaPitchedPtr,
        extent: cudaExtent,
    ) -> cudaError_t;
    pub fn cudaFree(devPtr: *mut c_void) -> cudaError_t;
    pub fn cudaFreeHost(ptr: *mut c_void) -> cudaError_t;
    pub fn cudaMemset(devPtr: *mut c_void, value: c_int, count: size_t) -> cudaError_t;
    pub fn cudaMemsetAsync(
        devPtr: *mut c_void,
        value: c_int,
        count: size_t,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn cudaMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        count: size_t,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;
    pub fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: size_t,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn cudaMemcpy2D(
        dst: *mut c_void,
        dpitch: size_t,
        src: *const c_void,
        spitch: size_t,
        width: size_t,
        height: size_t,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;
    pub fn cudaMemcpy2DAsync(
        dst: *mut c_void,
        dpitch: size_t,
        src: *const c_void,
        spitch: size_t,
        width: size_t,
        height: size_t,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn cudaMemcpyToSymbol(
        symbol: *const c_void,
        src: *const c_void,
        count: size_t,
        offset: size_t,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;
    pub fn cudaMemcpyFromSymbol(
        dst: *mut c_void,
        symbol: *const c_void,
        count: size_t,
        offset: size_t,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;
    pub fn cudaMemGetInfo(free: *mut size_t, total: *mut size_t) -> cudaError_t;
    pub fn cudaHostAlloc(
        pHost: *mut *mut c_void,
        size: size_t,
        flags: c_uint,
    ) -> cudaError_t;
    pub fn cudaHostRegister(
        ptr: *mut c_void,
        size: size_t,
        flags: c_uint,
    ) -> cudaError_t;
    pub fn cudaHostUnregister(ptr: *mut c_void) -> cudaError_t;
    pub fn cudaHostGetDevicePointer(
        pDevice: *mut *mut c_void,
        pHost: *mut c_void,
        flags: c_uint,
    ) -> cudaError_t;
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudaExtent {
    pub width: size_t,
    pub height: size_t,
    pub depth: size_t,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudaPitchedPtr {
    pub ptr: *mut c_void,
    pub pitch: size_t,
    pub xsize: size_t,
    pub ysize: size_t,
}

// ============================================================================
// External Functions - Stream Management
// ============================================================================

#[cfg(not(feature = "stub"))]
extern "C" {
    pub fn cudaStreamCreate(pStream: *mut cudaStream_t) -> cudaError_t;
    pub fn cudaStreamCreateWithFlags(pStream: *mut cudaStream_t, flags: c_uint) -> cudaError_t;
    pub fn cudaStreamCreateWithPriority(
        pStream: *mut cudaStream_t,
        flags: c_uint,
        priority: c_int,
    ) -> cudaError_t;
    pub fn cudaStreamDestroy(stream: cudaStream_t) -> cudaError_t;
    pub fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t;
    pub fn cudaStreamQuery(stream: cudaStream_t) -> cudaError_t;
    pub fn cudaStreamWaitEvent(
        stream: cudaStream_t,
        event: cudaEvent_t,
        flags: c_uint,
    ) -> cudaError_t;
    pub fn cudaStreamGetPriority(stream: cudaStream_t, priority: *mut c_int) -> cudaError_t;
    pub fn cudaStreamGetFlags(stream: cudaStream_t, flags: *mut c_uint) -> cudaError_t;
}

pub const cudaStreamDefault: c_uint = 0x00;
pub const cudaStreamNonBlocking: c_uint = 0x01;

// ============================================================================
// External Functions - Event Management
// ============================================================================

#[cfg(not(feature = "stub"))]
extern "C" {
    pub fn cudaEventCreate(event: *mut cudaEvent_t) -> cudaError_t;
    pub fn cudaEventCreateWithFlags(event: *mut cudaEvent_t, flags: c_uint) -> cudaError_t;
    pub fn cudaEventDestroy(event: cudaEvent_t) -> cudaError_t;
    pub fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> cudaError_t;
    pub fn cudaEventSynchronize(event: cudaEvent_t) -> cudaError_t;
    pub fn cudaEventQuery(event: cudaEvent_t) -> cudaError_t;
    pub fn cudaEventElapsedTime(
        ms: *mut f32,
        start: cudaEvent_t,
        end: cudaEvent_t,
    ) -> cudaError_t;
}

pub const cudaEventDefault: c_uint = 0x00;
pub const cudaEventBlockingSync: c_uint = 0x01;
pub const cudaEventDisableTiming: c_uint = 0x02;
pub const cudaEventInterprocess: c_uint = 0x04;

// ============================================================================
// External Functions - Error Handling
// ============================================================================

#[cfg(not(feature = "stub"))]
extern "C" {
    pub fn cudaGetLastError() -> cudaError_t;
    pub fn cudaPeekAtLastError() -> cudaError_t;
    pub fn cudaGetErrorName(error: cudaError_t) -> *const c_char;
    pub fn cudaGetErrorString(error: cudaError_t) -> *const c_char;
}

// ============================================================================
// External Functions - Version Information
// ============================================================================

#[cfg(not(feature = "stub"))]
extern "C" {
    pub fn cudaDriverGetVersion(driverVersion: *mut c_int) -> cudaError_t;
    pub fn cudaRuntimeGetVersion(runtimeVersion: *mut c_int) -> cudaError_t;
}

// ============================================================================
// External Functions - Unified Memory
// ============================================================================

pub const cudaMemAttachGlobal: c_uint = 0x01;
pub const cudaMemAttachHost: c_uint = 0x02;
pub const cudaMemAttachSingle: c_uint = 0x04;

#[cfg(not(feature = "stub"))]
extern "C" {
    pub fn cudaMemPrefetchAsync(
        devPtr: *const c_void,
        count: size_t,
        dstDevice: c_int,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn cudaMemAdvise(
        devPtr: *const c_void,
        count: size_t,
        advice: c_int,
        device: c_int,
    ) -> cudaError_t;
}

// ============================================================================
// External Functions - Peer Access
// ============================================================================

#[cfg(not(feature = "stub"))]
extern "C" {
    pub fn cudaDeviceCanAccessPeer(
        canAccessPeer: *mut c_int,
        device: c_int,
        peerDevice: c_int,
    ) -> cudaError_t;
    pub fn cudaDeviceEnablePeerAccess(peerDevice: c_int, flags: c_uint) -> cudaError_t;
    pub fn cudaDeviceDisablePeerAccess(peerDevice: c_int) -> cudaError_t;
    pub fn cudaMemcpyPeer(
        dst: *mut c_void,
        dstDevice: c_int,
        src: *const c_void,
        srcDevice: c_int,
        count: size_t,
    ) -> cudaError_t;
    pub fn cudaMemcpyPeerAsync(
        dst: *mut c_void,
        dstDevice: c_int,
        src: *const c_void,
        srcDevice: c_int,
        count: size_t,
        stream: cudaStream_t,
    ) -> cudaError_t;
}

// ============================================================================
// Kernel Launch (using Driver API style for flexibility)
// ============================================================================

#[cfg(not(feature = "stub"))]
extern "C" {
    pub fn cudaLaunchKernel(
        func: *const c_void,
        gridDim: dim3,
        blockDim: dim3,
        args: *mut *mut c_void,
        sharedMem: size_t,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn cudaLaunchCooperativeKernel(
        func: *const c_void,
        gridDim: dim3,
        blockDim: dim3,
        args: *mut *mut c_void,
        sharedMem: size_t,
        stream: cudaStream_t,
    ) -> cudaError_t;
}

// ============================================================================
// Host Function Callback
// ============================================================================

pub type cudaHostFn_t = Option<unsafe extern "C" fn(userData: *mut c_void)>;

#[cfg(not(feature = "stub"))]
extern "C" {
    pub fn cudaLaunchHostFunc(
        stream: cudaStream_t,
        fn_: cudaHostFn_t,
        userData: *mut c_void,
    ) -> cudaError_t;
}

// ============================================================================
// Occupancy
// ============================================================================

#[cfg(not(feature = "stub"))]
extern "C" {
    pub fn cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        numBlocks: *mut c_int,
        func: *const c_void,
        blockSize: c_int,
        dynamicSMemSize: size_t,
    ) -> cudaError_t;
    pub fn cudaOccupancyMaxPotentialBlockSize(
        minGridSize: *mut c_int,
        blockSize: *mut c_int,
        func: *const c_void,
        dynamicSMemSize: size_t,
        blockSizeLimit: c_int,
    ) -> cudaError_t;
}

// ============================================================================
// CUDA 12.3+ Features (conditional compilation)
// ============================================================================

#[cfg(all(feature = "cuda-12-3", not(feature = "stub")))]
extern "C" {
    pub fn cudaGraphInstantiateWithParams(
        pGraphExec: *mut cudaGraphExec_t,
        graph: cudaGraph_t,
        instantiateParams: *mut cudaGraphInstantiateParams,
    ) -> cudaError_t;
}

#[cfg(feature = "cuda-12-3")]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUgraph_st {
    _unused: [u8; 0],
}
#[cfg(feature = "cuda-12-3")]
pub type cudaGraph_t = *mut CUgraph_st;

#[cfg(feature = "cuda-12-3")]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUgraphExec_st {
    _unused: [u8; 0],
}
#[cfg(feature = "cuda-12-3")]
pub type cudaGraphExec_t = *mut CUgraphExec_st;

#[cfg(feature = "cuda-12-3")]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudaGraphInstantiateParams {
    pub flags: c_uint,
    pub uploadStream: cudaStream_t,
}

// ============================================================================
// Stub Implementations (no CUDA libraries linked)
// ============================================================================

#[cfg(feature = "stub")]
pub use stub::*;

#[cfg(feature = "stub")]
#[allow(non_snake_case)]
mod stub {
    use super::*;

    static ERROR_STR: &[u8] = b"CUDA runtime not available\0";

    pub unsafe fn cudaGetErrorString(_error: cudaError_t) -> *const c_char {
        ERROR_STR.as_ptr() as *const c_char
    }

    pub unsafe fn cudaGetDeviceCount(count: *mut c_int) -> cudaError_t {
        if !count.is_null() {
            *count = 0;
        }
        cudaErrorNoDevice
    }

    pub unsafe fn cudaGetDevice(device: *mut c_int) -> cudaError_t {
        if !device.is_null() {
            *device = 0;
        }
        cudaErrorInitializationError
    }

    pub unsafe fn cudaSetDevice(_device: c_int) -> cudaError_t {
        cudaErrorInitializationError
    }

    pub unsafe fn cudaDeviceSynchronize() -> cudaError_t {
        cudaErrorInitializationError
    }

    pub unsafe fn cudaDeviceReset() -> cudaError_t {
        cudaErrorInitializationError
    }

    pub unsafe fn cudaStreamCreate(pStream: *mut cudaStream_t) -> cudaError_t {
        if !pStream.is_null() {
            *pStream = std::ptr::null_mut();
        }
        cudaErrorInitializationError
    }

    pub unsafe fn cudaStreamCreateWithFlags(pStream: *mut cudaStream_t, _flags: c_uint) -> cudaError_t {
        if !pStream.is_null() {
            *pStream = std::ptr::null_mut();
        }
        cudaErrorInitializationError
    }

    pub unsafe fn cudaStreamSynchronize(_stream: cudaStream_t) -> cudaError_t {
        cudaErrorInitializationError
    }

    pub unsafe fn cudaStreamDestroy(_stream: cudaStream_t) -> cudaError_t {
        cudaErrorInitializationError
    }

    pub unsafe fn cudaEventCreate(event: *mut cudaEvent_t) -> cudaError_t {
        if !event.is_null() {
            *event = std::ptr::null_mut();
        }
        cudaErrorInitializationError
    }

    pub unsafe fn cudaEventCreateWithFlags(event: *mut cudaEvent_t, _flags: c_uint) -> cudaError_t {
        if !event.is_null() {
            *event = std::ptr::null_mut();
        }
        cudaErrorInitializationError
    }

    pub unsafe fn cudaEventRecord(_event: cudaEvent_t, _stream: cudaStream_t) -> cudaError_t {
        cudaErrorInitializationError
    }

    pub unsafe fn cudaEventSynchronize(_event: cudaEvent_t) -> cudaError_t {
        cudaErrorInitializationError
    }

    pub unsafe fn cudaEventElapsedTime(ms: *mut f32, _start: cudaEvent_t, _end: cudaEvent_t) -> cudaError_t {
        if !ms.is_null() {
            *ms = 0.0;
        }
        cudaErrorInitializationError
    }

    pub unsafe fn cudaEventDestroy(_event: cudaEvent_t) -> cudaError_t {
        cudaErrorInitializationError
    }

    pub unsafe fn cudaMalloc(devPtr: *mut *mut c_void, _size: size_t) -> cudaError_t {
        if !devPtr.is_null() {
            *devPtr = std::ptr::null_mut();
        }
        cudaErrorInitializationError
    }

    pub unsafe fn cudaMemcpy(
        _dst: *mut c_void,
        _src: *const c_void,
        _count: size_t,
        _kind: cudaMemcpyKind,
    ) -> cudaError_t {
        cudaErrorInitializationError
    }

    pub unsafe fn cudaMemcpyAsync(
        _dst: *mut c_void,
        _src: *const c_void,
        _count: size_t,
        _kind: cudaMemcpyKind,
        _stream: cudaStream_t,
    ) -> cudaError_t {
        cudaErrorInitializationError
    }

    pub unsafe fn cudaMemset(_devPtr: *mut c_void, _value: c_int, _count: size_t) -> cudaError_t {
        cudaErrorInitializationError
    }

    pub unsafe fn cudaFree(_devPtr: *mut c_void) -> cudaError_t {
        cudaErrorInitializationError
    }

    pub unsafe fn cudaGetLastError() -> cudaError_t {
        cudaErrorInitializationError
    }

    pub unsafe fn cudaPeekAtLastError() -> cudaError_t {
        cudaErrorInitializationError
    }
}
