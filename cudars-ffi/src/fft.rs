//! cuFFT FFI exports.

use super::CudaRsResult;
use cufft::{Plan1dC2C, Plan2dC2C};
use libc::c_int;
use std::collections::HashMap;
use std::sync::Mutex;

lazy_static::lazy_static! {
    static ref FFT_PLANS_1D: Mutex<HandleManager<Plan1dC2C>> = Mutex::new(HandleManager::new());
    static ref FFT_PLANS_2D: Mutex<HandleManager<Plan2dC2C>> = Mutex::new(HandleManager::new());
}

struct HandleManager<T> {
    handles: HashMap<u64, T>,
    next_id: u64,
}

impl<T> HandleManager<T> {
    fn new() -> Self {
        Self {
            handles: HashMap::new(),
            next_id: 1,
        }
    }

    fn insert(&mut self, value: T) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.handles.insert(id, value);
        id
    }

    fn remove(&mut self, id: u64) -> Option<T> {
        self.handles.remove(&id)
    }
}

pub type CudaRsFftPlan = u64;

/// Create a 1D FFT plan (C2C).
#[no_mangle]
pub extern "C" fn cudars_fft_plan_1d_c2c(plan: *mut CudaRsFftPlan, nx: c_int) -> CudaRsResult {
    if plan.is_null() || nx <= 0 {
        return CudaRsResult::ErrorInvalidValue;
    }
    match Plan1dC2C::new(nx) {
        Ok(p) => {
            let mut plans = FFT_PLANS_1D.lock().unwrap();
            let id = plans.insert(p);
            unsafe { *plan = id };
            CudaRsResult::Success
        }
        Err(_) => CudaRsResult::ErrorUnknown,
    }
}

/// Create a 2D FFT plan (C2C).
#[no_mangle]
pub extern "C" fn cudars_fft_plan_2d_c2c(
    plan: *mut CudaRsFftPlan,
    nx: c_int,
    ny: c_int,
) -> CudaRsResult {
    if plan.is_null() || nx <= 0 || ny <= 0 {
        return CudaRsResult::ErrorInvalidValue;
    }
    match Plan2dC2C::new(nx, ny) {
        Ok(p) => {
            let mut plans = FFT_PLANS_2D.lock().unwrap();
            let id = plans.insert(p);
            unsafe { *plan = id };
            CudaRsResult::Success
        }
        Err(_) => CudaRsResult::ErrorUnknown,
    }
}

/// Destroy a 1D FFT plan.
#[no_mangle]
pub extern "C" fn cudars_fft_plan_1d_destroy(plan: CudaRsFftPlan) -> CudaRsResult {
    let mut plans = FFT_PLANS_1D.lock().unwrap();
    match plans.remove(plan) {
        Some(_) => CudaRsResult::Success,
        None => CudaRsResult::ErrorInvalidHandle,
    }
}

/// Destroy a 2D FFT plan.
#[no_mangle]
pub extern "C" fn cudars_fft_plan_2d_destroy(plan: CudaRsFftPlan) -> CudaRsResult {
    let mut plans = FFT_PLANS_2D.lock().unwrap();
    match plans.remove(plan) {
        Some(_) => CudaRsResult::Success,
        None => CudaRsResult::ErrorInvalidHandle,
    }
}
