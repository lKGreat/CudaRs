//! GPU memory pool management (stub implementation).

use crate::CudaRsResult;
use cuda_runtime::{self};
use cuda_runtime_sys::{cudaFree, cudaMalloc, cudaMemGetInfo};
use libc::{c_char, c_uint, c_ulonglong, c_void, size_t};
use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::{Mutex, MutexGuard};
use std::sync::atomic::{AtomicU64, Ordering};

#[repr(C)]
pub enum CudaRsOomPolicy {
    Fail = 0,
    Wait = 1,
    Skip = 2,
    FallbackCpu = 3,
}

#[repr(C)]
pub struct CudaRsMemoryQuota {
    pub max_bytes: c_ulonglong,
    pub preallocate_bytes: c_ulonglong,
    pub allow_fallback_to_shared: bool,
    pub oom_policy: CudaRsOomPolicy,
}

#[repr(C)]
pub struct CudaRsMemoryPoolStats {
    pub quota: c_ulonglong,
    pub used: c_ulonglong,
    pub peak: c_ulonglong,
    pub allocation_count: c_uint,
    pub fragmentation_rate: f32,
}

#[repr(C)]
pub struct CudaRsGpuMemoryStats {
    pub device_id: i32,
    pub total: c_ulonglong,
    pub free: c_ulonglong,
    pub used: c_ulonglong,
    pub fragmentation_rate: f32,
}

#[derive(Clone, Copy)]
struct Block {
    ptr: *mut c_void,
    size: u64,
    in_use: bool,
}

struct MemoryPool {
    #[allow(dead_code)]
    id: String,
    device_id: i32,
    quota: CudaRsMemoryQuota,
    used: u64,
    peak: u64,
    allocation_count: u32,
    free_count: u32,
    allocations: HashMap<u64, Block>,
    free_list: Vec<Block>,
}

unsafe impl Send for MemoryPool {}
unsafe impl Sync for MemoryPool {}

impl MemoryPool {
    fn new(id: String, device_id: i32, quota: CudaRsMemoryQuota) -> Self {
        Self {
            id,
            device_id,
            quota,
            used: 0,
            peak: 0,
            allocation_count: 0,
            free_count: 0,
            allocations: HashMap::new(),
            free_list: Vec::new(),
        }
    }

    fn reserved_bytes(&self) -> u64 {
        self.used + self.free_list.iter().map(|b| b.size).sum::<u64>()
    }

    fn fragmentation_rate(&self) -> f32 {
        let reserved = self.reserved_bytes();
        if reserved == 0 {
            0.0
        } else {
            let free = self.free_list.iter().map(|b| b.size).sum::<u64>() as f32;
            free / reserved as f32
        }
    }

    fn try_reuse_block(&mut self, size: u64) -> Option<u64> {
        let idx = self.free_list.iter().position(|b| b.size >= size)?;
        let mut block = self.free_list.swap_remove(idx);
        block.in_use = true;
        self.used = self.used.saturating_add(block.size);
        self.peak = self.peak.max(self.used);
        let ptr = block.ptr as u64;
        self.allocations.insert(ptr, block);
        Some(ptr)
    }

    fn allocate(&mut self, size: u64) -> Result<u64, CudaRsResult> {
        if self.quota.max_bytes > 0 && self.used + size > self.quota.max_bytes {
            return Err(CudaRsResult::ErrorOutOfMemory);
        }

        if let Some(ptr) = self.try_reuse_block(size) {
            return Ok(ptr);
        }

        unsafe {
            let mut dev_ptr: *mut c_void = std::ptr::null_mut();
            let code = cudaMalloc(&mut dev_ptr as *mut *mut c_void, size as size_t);
            if code != 0 || dev_ptr.is_null() {
                return Err(CudaRsResult::ErrorOutOfMemory);
            }

            let block = Block { ptr: dev_ptr, size, in_use: true };
            let key = dev_ptr as u64;
            self.used = self.used.saturating_add(size);
            self.peak = self.peak.max(self.used);
            self.allocation_count = self.allocation_count.saturating_add(1);
            self.allocations.insert(key, block);
            Ok(key)
        }
    }

    fn free(&mut self, ptr: u64) -> Result<(), CudaRsResult> {
        match self.allocations.get_mut(&ptr) {
            Some(block) if block.in_use => {
                block.in_use = false;
                self.used = self.used.saturating_sub(block.size);
                self.free_list.push(*block);
                self.free_count = self.free_count.saturating_add(1);
                Ok(())
            }
            _ => Err(CudaRsResult::ErrorInvalidHandle),
        }
    }

    fn defragment(&mut self) -> Result<(), CudaRsResult> {
        let mut freed = Vec::new();

        for block in self.free_list.drain(..) {
            unsafe {
                let _ = cudaFree(block.ptr);
            }
            freed.push(block.ptr as u64);
        }

        for ptr in freed {
            self.allocations.remove(&ptr);
        }

        Ok(())
    }
}

lazy_static::lazy_static! {
    static ref POOLS: Mutex<HashMap<u64, MemoryPool>> = Mutex::new(HashMap::new());
}

static NEXT_POOL_ID: AtomicU64 = AtomicU64::new(1);

fn pools() -> MutexGuard<'static, HashMap<u64, MemoryPool>> {
    POOLS.lock().unwrap()
}

/// Create a memory pool for a model (stub).
#[no_mangle]
pub extern "C" fn cudars_memory_pool_create(
    _pool_id: *const c_char,
    _quota: CudaRsMemoryQuota,
    out_handle: *mut c_ulonglong,
) -> CudaRsResult {
    if out_handle.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let device_id = match cuda_runtime::get_device() {
        Ok(d) => d,
        Err(_) => return CudaRsResult::ErrorUnknown,
    };

    let id = unsafe {
        if !_is_valid_pool_id(_pool_id) {
            return CudaRsResult::ErrorInvalidValue;
        }
        CStr::from_ptr(_pool_id).to_string_lossy().to_string()
    };

    let handle = NEXT_POOL_ID.fetch_add(1, Ordering::SeqCst);
    let mut pools = pools();
    pools.insert(handle, MemoryPool::new(id, device_id, _quota));

    unsafe { *out_handle = handle; }
    CudaRsResult::Success
}

/// Create a memory pool on a specific device.
#[no_mangle]
pub extern "C" fn cudars_memory_pool_create_with_device(
    pool_id: *const c_char,
    device_id: i32,
    quota: CudaRsMemoryQuota,
    out_handle: *mut c_ulonglong,
) -> CudaRsResult {
    if out_handle.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    if !_is_valid_pool_id(pool_id) {
        return CudaRsResult::ErrorInvalidValue;
    }

    if cuda_runtime::set_device(device_id).is_err() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let id = unsafe { CStr::from_ptr(pool_id).to_string_lossy().to_string() };
    let handle = NEXT_POOL_ID.fetch_add(1, Ordering::SeqCst);

    let mut pools = pools();
    pools.insert(handle, MemoryPool::new(id, device_id, quota));

    unsafe { *out_handle = handle; }
    CudaRsResult::Success
}

/// Destroy a memory pool (stub).
#[no_mangle]
pub extern "C" fn cudars_memory_pool_destroy(_handle: c_ulonglong) -> CudaRsResult {
    let mut pools = pools();
    if let Some(mut pool) = pools.remove(&_handle) {
        let _ = pool.defragment();
        for block in pool.allocations.values() {
            if block.in_use {
                unsafe { let _ = cudaFree(block.ptr); }
            }
        }
        CudaRsResult::Success
    } else {
        CudaRsResult::ErrorInvalidHandle
    }
}

/// Allocate memory from a pool (stub).
#[no_mangle]
pub extern "C" fn cudars_memory_pool_allocate(
    _handle: c_ulonglong,
    _size: c_ulonglong,
    out_ptr: *mut c_ulonglong,
) -> CudaRsResult {
    if out_ptr.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let mut pools = pools();
    let pool = match pools.get_mut(&_handle) {
        Some(pool) => pool,
        None => return CudaRsResult::ErrorInvalidHandle,
    };

    if cuda_runtime::set_device(pool.device_id).is_err() {
        return CudaRsResult::ErrorInvalidValue;
    }

    match pool.allocate(_size) {
        Ok(ptr) => {
            unsafe { *out_ptr = ptr; }
            CudaRsResult::Success
        }
        Err(code) => code,
    }
}

/// Free memory in a pool (stub).
#[no_mangle]
pub extern "C" fn cudars_memory_pool_free(_handle: c_ulonglong, _ptr: c_ulonglong) -> CudaRsResult {
    let mut pools = pools();
    let pool = match pools.get_mut(&_handle) {
        Some(pool) => pool,
        None => return CudaRsResult::ErrorInvalidHandle,
    };

    if cuda_runtime::set_device(pool.device_id).is_err() {
        return CudaRsResult::ErrorInvalidValue;
    }

    match pool.free(_ptr) {
        Ok(()) => CudaRsResult::Success,
        Err(code) => code,
    }
}

/// Get memory pool stats (stub).
#[no_mangle]
pub extern "C" fn cudars_memory_pool_get_stats(
    _handle: c_ulonglong,
    out_stats: *mut CudaRsMemoryPoolStats,
) -> CudaRsResult {
    if out_stats.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    let pools = pools();
    let pool = match pools.get(&_handle) {
        Some(pool) => pool,
        None => return CudaRsResult::ErrorInvalidHandle,
    };

    unsafe {
        *out_stats = CudaRsMemoryPoolStats {
            quota: pool.quota.max_bytes,
            used: pool.used,
            peak: pool.peak,
            allocation_count: pool.allocation_count,
            fragmentation_rate: pool.fragmentation_rate(),
        };
    }

    CudaRsResult::Success
}

/// Trigger defragmentation (stub).
#[no_mangle]
pub extern "C" fn cudars_memory_pool_defragment(_handle: c_ulonglong) -> CudaRsResult {
    let mut pools = pools();
    let pool = match pools.get_mut(&_handle) {
        Some(pool) => pool,
        None => return CudaRsResult::ErrorInvalidHandle,
    };

    if cuda_runtime::set_device(pool.device_id).is_err() {
        return CudaRsResult::ErrorInvalidValue;
    }

    match pool.defragment() {
        Ok(()) => CudaRsResult::Success,
        Err(code) => code,
    }
}

/// Get GPU memory stats (stub).
#[no_mangle]
pub extern "C" fn cudars_gpu_get_memory_stats(
    _device_id: i32,
    out_stats: *mut CudaRsGpuMemoryStats,
) -> CudaRsResult {
    if out_stats.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }

    if cuda_runtime::set_device(_device_id).is_err() {
        return CudaRsResult::ErrorInvalidValue;
    }

    unsafe {
        let mut free: size_t = 0;
        let mut total: size_t = 0;
        let code = cudaMemGetInfo(&mut free, &mut total);
        if code != 0 {
            return CudaRsResult::ErrorUnknown;
        }

        let pools = pools();
        let mut reserved: u64 = 0;
        let mut free_in_pools: u64 = 0;
        for pool in pools.values() {
            if pool.device_id == _device_id {
                reserved += pool.reserved_bytes();
                free_in_pools += pool.free_list.iter().map(|b| b.size).sum::<u64>();
            }
        }

        let fragmentation_rate = if reserved > 0 {
            free_in_pools as f32 / reserved as f32
        } else {
            0.0
        };

        *out_stats = CudaRsGpuMemoryStats {
            device_id: _device_id,
            total: total as u64,
            free: free as u64,
            used: (total - free) as u64,
            fragmentation_rate,
        };
    }

    CudaRsResult::Success
}

/// Helper to validate pool id string.
fn _is_valid_pool_id(pool_id: *const c_char) -> bool {
    if pool_id.is_null() {
        return false;
    }

    unsafe {
        if let Ok(s) = CStr::from_ptr(pool_id).to_str() {
            !s.trim().is_empty()
        } else {
            false
        }
    }
}
