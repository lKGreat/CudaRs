//! cuRAND FFI exports.

use super::CudaRsResult;
use curand::{Generator, RngType};
use libc::{c_int, size_t};
use std::collections::HashMap;
use std::sync::Mutex;

lazy_static::lazy_static! {
    static ref GENERATORS: Mutex<HandleManager<Generator>> = Mutex::new(HandleManager::new());
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

    fn get(&self, id: u64) -> Option<&T> {
        self.handles.get(&id)
    }

    fn remove(&mut self, id: u64) -> Option<T> {
        self.handles.remove(&id)
    }
}

pub type CudaRsRng = u64;

/// RNG type constants.
pub const CUDARS_RNG_PSEUDO_XORWOW: c_int = 0;
pub const CUDARS_RNG_PSEUDO_MRG32K3A: c_int = 1;
pub const CUDARS_RNG_PSEUDO_PHILOX: c_int = 2;
pub const CUDARS_RNG_QUASI_SOBOL32: c_int = 3;

/// Create a random number generator.
#[no_mangle]
pub extern "C" fn cudars_rand_create(rng: *mut CudaRsRng, rng_type: c_int) -> CudaRsResult {
    if rng.is_null() {
        return CudaRsResult::ErrorInvalidValue;
    }
    
    let rt = match rng_type {
        CUDARS_RNG_PSEUDO_XORWOW => RngType::PseudoXorwow,
        CUDARS_RNG_PSEUDO_MRG32K3A => RngType::PseudoMrg32k3a,
        CUDARS_RNG_PSEUDO_PHILOX => RngType::PseudoPhilox4_32_10,
        CUDARS_RNG_QUASI_SOBOL32 => RngType::QuasiSobol32,
        _ => return CudaRsResult::ErrorInvalidValue,
    };
    
    match Generator::with_type(rt) {
        Ok(g) => {
            let mut gens = GENERATORS.lock().unwrap();
            let id = gens.insert(g);
            unsafe { *rng = id };
            CudaRsResult::Success
        }
        Err(_) => CudaRsResult::ErrorUnknown,
    }
}

/// Destroy a random number generator.
#[no_mangle]
pub extern "C" fn cudars_rand_destroy(rng: CudaRsRng) -> CudaRsResult {
    let mut gens = GENERATORS.lock().unwrap();
    match gens.remove(rng) {
        Some(_) => CudaRsResult::Success,
        None => CudaRsResult::ErrorInvalidHandle,
    }
}

/// Set the seed for a random number generator.
#[no_mangle]
pub extern "C" fn cudars_rand_set_seed(rng: CudaRsRng, seed: u64) -> CudaRsResult {
    let gens = GENERATORS.lock().unwrap();
    match gens.get(rng) {
        Some(g) => match g.set_seed(seed) {
            Ok(()) => CudaRsResult::Success,
            Err(_) => CudaRsResult::ErrorUnknown,
        },
        None => CudaRsResult::ErrorInvalidHandle,
    }
}

/// Generate uniform random numbers (float).
#[no_mangle]
pub extern "C" fn cudars_rand_generate_uniform(
    rng: CudaRsRng,
    output: *mut f32,
    n: size_t,
) -> CudaRsResult {
    if output.is_null() || n == 0 {
        return CudaRsResult::ErrorInvalidValue;
    }
    
    let gens = GENERATORS.lock().unwrap();
    match gens.get(rng) {
        Some(g) => match g.generate_uniform(output, n) {
            Ok(()) => CudaRsResult::Success,
            Err(_) => CudaRsResult::ErrorUnknown,
        },
        None => CudaRsResult::ErrorInvalidHandle,
    }
}

/// Generate normal random numbers (float).
#[no_mangle]
pub extern "C" fn cudars_rand_generate_normal(
    rng: CudaRsRng,
    output: *mut f32,
    n: size_t,
    mean: f32,
    stddev: f32,
) -> CudaRsResult {
    if output.is_null() || n == 0 {
        return CudaRsResult::ErrorInvalidValue;
    }
    
    let gens = GENERATORS.lock().unwrap();
    match gens.get(rng) {
        Some(g) => match g.generate_normal(output, n, mean, stddev) {
            Ok(()) => CudaRsResult::Success,
            Err(_) => CudaRsResult::ErrorUnknown,
        },
        None => CudaRsResult::ErrorInvalidHandle,
    }
}
