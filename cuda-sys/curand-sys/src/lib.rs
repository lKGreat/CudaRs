//! Raw FFI bindings to cuRAND.
//!
//! cuRAND is NVIDIA's GPU-accelerated library for random number generation.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use cuda_runtime_sys::cudaStream_t;
use libc::{c_double, c_int, c_uint, c_ulong, size_t};

// ============================================================================
// Types
// ============================================================================

pub type curandStatus_t = c_int;

pub const CURAND_STATUS_SUCCESS: curandStatus_t = 0;
pub const CURAND_STATUS_VERSION_MISMATCH: curandStatus_t = 100;
pub const CURAND_STATUS_NOT_INITIALIZED: curandStatus_t = 101;
pub const CURAND_STATUS_ALLOCATION_FAILED: curandStatus_t = 102;
pub const CURAND_STATUS_TYPE_ERROR: curandStatus_t = 103;
pub const CURAND_STATUS_OUT_OF_RANGE: curandStatus_t = 104;
pub const CURAND_STATUS_LENGTH_NOT_MULTIPLE: curandStatus_t = 105;
pub const CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: curandStatus_t = 106;
pub const CURAND_STATUS_LAUNCH_FAILURE: curandStatus_t = 201;
pub const CURAND_STATUS_PREEXISTING_FAILURE: curandStatus_t = 202;
pub const CURAND_STATUS_INITIALIZATION_FAILED: curandStatus_t = 203;
pub const CURAND_STATUS_ARCH_MISMATCH: curandStatus_t = 204;
pub const CURAND_STATUS_INTERNAL_ERROR: curandStatus_t = 999;

pub type curandRngType_t = c_int;

pub const CURAND_RNG_TEST: curandRngType_t = 0;
pub const CURAND_RNG_PSEUDO_DEFAULT: curandRngType_t = 100;
pub const CURAND_RNG_PSEUDO_XORWOW: curandRngType_t = 101;
pub const CURAND_RNG_PSEUDO_MRG32K3A: curandRngType_t = 121;
pub const CURAND_RNG_PSEUDO_MTGP32: curandRngType_t = 141;
pub const CURAND_RNG_PSEUDO_MT19937: curandRngType_t = 142;
pub const CURAND_RNG_PSEUDO_PHILOX4_32_10: curandRngType_t = 161;
pub const CURAND_RNG_QUASI_DEFAULT: curandRngType_t = 200;
pub const CURAND_RNG_QUASI_SOBOL32: curandRngType_t = 201;
pub const CURAND_RNG_QUASI_SCRAMBLED_SOBOL32: curandRngType_t = 202;
pub const CURAND_RNG_QUASI_SOBOL64: curandRngType_t = 203;
pub const CURAND_RNG_QUASI_SCRAMBLED_SOBOL64: curandRngType_t = 204;

pub type curandOrdering_t = c_int;

pub const CURAND_ORDERING_PSEUDO_BEST: curandOrdering_t = 100;
pub const CURAND_ORDERING_PSEUDO_DEFAULT: curandOrdering_t = 101;
pub const CURAND_ORDERING_PSEUDO_SEEDED: curandOrdering_t = 102;
pub const CURAND_ORDERING_PSEUDO_LEGACY: curandOrdering_t = 103;
pub const CURAND_ORDERING_PSEUDO_DYNAMIC: curandOrdering_t = 104;
pub const CURAND_ORDERING_QUASI_DEFAULT: curandOrdering_t = 201;

pub type curandDirectionVectorSet_t = c_int;

pub const CURAND_DIRECTION_VECTORS_32_JOEKUO6: curandDirectionVectorSet_t = 101;
pub const CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6: curandDirectionVectorSet_t = 102;
pub const CURAND_DIRECTION_VECTORS_64_JOEKUO6: curandDirectionVectorSet_t = 103;
pub const CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6: curandDirectionVectorSet_t = 104;

// ============================================================================
// Opaque Types
// ============================================================================

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct curandGenerator_st {
    _unused: [u8; 0],
}
pub type curandGenerator_t = *mut curandGenerator_st;

// ============================================================================
// External Functions - Generator Management
// ============================================================================

extern "C" {
    pub fn curandCreateGenerator(
        generator: *mut curandGenerator_t,
        rng_type: curandRngType_t,
    ) -> curandStatus_t;

    pub fn curandCreateGeneratorHost(
        generator: *mut curandGenerator_t,
        rng_type: curandRngType_t,
    ) -> curandStatus_t;

    pub fn curandDestroyGenerator(generator: curandGenerator_t) -> curandStatus_t;

    pub fn curandSetStream(
        generator: curandGenerator_t,
        stream: cudaStream_t,
    ) -> curandStatus_t;

    pub fn curandSetPseudoRandomGeneratorSeed(
        generator: curandGenerator_t,
        seed: c_ulong,
    ) -> curandStatus_t;

    pub fn curandSetGeneratorOffset(
        generator: curandGenerator_t,
        offset: c_ulong,
    ) -> curandStatus_t;

    pub fn curandSetGeneratorOrdering(
        generator: curandGenerator_t,
        order: curandOrdering_t,
    ) -> curandStatus_t;

    pub fn curandSetQuasiRandomGeneratorDimensions(
        generator: curandGenerator_t,
        num_dimensions: c_uint,
    ) -> curandStatus_t;
}

// ============================================================================
// External Functions - Generation
// ============================================================================

extern "C" {
    // Uniform distribution [0, 1)
    pub fn curandGenerateUniform(
        generator: curandGenerator_t,
        outputPtr: *mut f32,
        num: size_t,
    ) -> curandStatus_t;

    pub fn curandGenerateUniformDouble(
        generator: curandGenerator_t,
        outputPtr: *mut f64,
        num: size_t,
    ) -> curandStatus_t;

    // Normal distribution (mean=0, stddev=1)
    pub fn curandGenerateNormal(
        generator: curandGenerator_t,
        outputPtr: *mut f32,
        n: size_t,
        mean: f32,
        stddev: f32,
    ) -> curandStatus_t;

    pub fn curandGenerateNormalDouble(
        generator: curandGenerator_t,
        outputPtr: *mut f64,
        n: size_t,
        mean: c_double,
        stddev: c_double,
    ) -> curandStatus_t;

    // Log-normal distribution
    pub fn curandGenerateLogNormal(
        generator: curandGenerator_t,
        outputPtr: *mut f32,
        n: size_t,
        mean: f32,
        stddev: f32,
    ) -> curandStatus_t;

    pub fn curandGenerateLogNormalDouble(
        generator: curandGenerator_t,
        outputPtr: *mut f64,
        n: size_t,
        mean: c_double,
        stddev: c_double,
    ) -> curandStatus_t;

    // Poisson distribution
    pub fn curandGeneratePoisson(
        generator: curandGenerator_t,
        outputPtr: *mut c_uint,
        n: size_t,
        lambda: c_double,
    ) -> curandStatus_t;

    // Raw unsigned integers
    pub fn curandGenerate(
        generator: curandGenerator_t,
        outputPtr: *mut c_uint,
        num: size_t,
    ) -> curandStatus_t;

    pub fn curandGenerateLongLong(
        generator: curandGenerator_t,
        outputPtr: *mut c_ulong,
        num: size_t,
    ) -> curandStatus_t;
}

// ============================================================================
// External Functions - Version
// ============================================================================

extern "C" {
    pub fn curandGetVersion(version: *mut c_int) -> curandStatus_t;
    pub fn curandGetProperty(type_: c_int, value: *mut c_int) -> curandStatus_t;
}
