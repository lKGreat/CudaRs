//! Safe Rust wrapper for cuRAND.

use curand_sys::*;
use cuda_runtime::Stream;
use std::ptr;
use thiserror::Error;

#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
#[error("cuRAND Error: {0}")]
pub struct CurandError(pub i32);

pub type Result<T> = std::result::Result<T, CurandError>;

#[inline]
fn check(code: curandStatus_t) -> Result<()> {
    if code == CURAND_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(CurandError(code))
    }
}

/// Random number generator type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RngType {
    /// XORWOW pseudo-random generator
    PseudoXorwow,
    /// MRG32k3a pseudo-random generator
    PseudoMrg32k3a,
    /// Mersenne Twister pseudo-random generator
    PseudoMtgp32,
    /// Philox pseudo-random generator
    PseudoPhilox4_32_10,
    /// Sobol quasi-random generator
    QuasiSobol32,
    /// Scrambled Sobol quasi-random generator
    QuasiScrambledSobol32,
    /// Sobol64 quasi-random generator
    QuasiSobol64,
    /// Scrambled Sobol64 quasi-random generator
    QuasiScrambledSobol64,
}

impl RngType {
    fn to_curand(self) -> curandRngType_t {
        match self {
            RngType::PseudoXorwow => CURAND_RNG_PSEUDO_XORWOW,
            RngType::PseudoMrg32k3a => CURAND_RNG_PSEUDO_MRG32K3A,
            RngType::PseudoMtgp32 => CURAND_RNG_PSEUDO_MTGP32,
            RngType::PseudoPhilox4_32_10 => CURAND_RNG_PSEUDO_PHILOX4_32_10,
            RngType::QuasiSobol32 => CURAND_RNG_QUASI_SOBOL32,
            RngType::QuasiScrambledSobol32 => CURAND_RNG_QUASI_SCRAMBLED_SOBOL32,
            RngType::QuasiSobol64 => CURAND_RNG_QUASI_SOBOL64,
            RngType::QuasiScrambledSobol64 => CURAND_RNG_QUASI_SCRAMBLED_SOBOL64,
        }
    }
}

/// cuRAND Generator wrapper with automatic resource management.
pub struct Generator {
    handle: curandGenerator_t,
}

impl Generator {
    /// Create a new generator with the default type (XORWOW).
    pub fn new() -> Result<Self> {
        Self::with_type(RngType::PseudoXorwow)
    }

    /// Create a new generator with the specified type.
    pub fn with_type(rng_type: RngType) -> Result<Self> {
        let mut handle = ptr::null_mut();
        unsafe { check(curandCreateGenerator(&mut handle, rng_type.to_curand()))? };
        Ok(Self { handle })
    }

    /// Set the seed for pseudo-random generators.
    pub fn set_seed(&self, seed: u64) -> Result<()> {
        let seed = u32::try_from(seed).map_err(|_| CurandError(CURAND_STATUS_OUT_OF_RANGE))?;
        unsafe { check(curandSetPseudoRandomGeneratorSeed(self.handle, seed)) }
    }

    /// Set the offset for the generator.
    pub fn set_offset(&self, offset: u64) -> Result<()> {
        let offset = u32::try_from(offset).map_err(|_| CurandError(CURAND_STATUS_OUT_OF_RANGE))?;
        unsafe { check(curandSetGeneratorOffset(self.handle, offset)) }
    }

    /// Set the stream for this generator.
    pub fn set_stream(&self, stream: &Stream) -> Result<()> {
        unsafe { check(curandSetStream(self.handle, stream.as_raw())) }
    }

    /// Generate uniformly distributed floats in (0, 1].
    pub fn generate_uniform(&self, output: *mut f32, n: usize) -> Result<()> {
        unsafe { check(curandGenerateUniform(self.handle, output, n)) }
    }

    /// Generate uniformly distributed doubles in (0, 1].
    pub fn generate_uniform_double(&self, output: *mut f64, n: usize) -> Result<()> {
        unsafe { check(curandGenerateUniformDouble(self.handle, output, n)) }
    }

    /// Generate normally distributed floats.
    pub fn generate_normal(&self, output: *mut f32, n: usize, mean: f32, stddev: f32) -> Result<()> {
        unsafe { check(curandGenerateNormal(self.handle, output, n, mean, stddev)) }
    }

    /// Generate normally distributed doubles.
    pub fn generate_normal_double(
        &self,
        output: *mut f64,
        n: usize,
        mean: f64,
        stddev: f64,
    ) -> Result<()> {
        unsafe { check(curandGenerateNormalDouble(self.handle, output, n, mean, stddev)) }
    }

    /// Generate log-normally distributed floats.
    pub fn generate_log_normal(
        &self,
        output: *mut f32,
        n: usize,
        mean: f32,
        stddev: f32,
    ) -> Result<()> {
        unsafe { check(curandGenerateLogNormal(self.handle, output, n, mean, stddev)) }
    }

    /// Generate log-normally distributed doubles.
    pub fn generate_log_normal_double(
        &self,
        output: *mut f64,
        n: usize,
        mean: f64,
        stddev: f64,
    ) -> Result<()> {
        unsafe { check(curandGenerateLogNormalDouble(self.handle, output, n, mean, stddev)) }
    }

    /// Generate 32-bit unsigned integers.
    pub fn generate(&self, output: *mut u32, n: usize) -> Result<()> {
        unsafe { check(curandGenerate(self.handle, output, n)) }
    }

    /// Generate Poisson-distributed unsigned integers.
    pub fn generate_poisson(&self, output: *mut u32, n: usize, lambda: f64) -> Result<()> {
        unsafe { check(curandGeneratePoisson(self.handle, output, n, lambda)) }
    }

    /// Get the raw handle.
    pub fn as_raw(&self) -> curandGenerator_t {
        self.handle
    }
}

impl Drop for Generator {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { curandDestroyGenerator(self.handle) };
        }
    }
}

unsafe impl Send for Generator {}
unsafe impl Sync for Generator {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_generator() {
        // This will fail if no CUDA device is available
        let _ = Generator::new();
    }
}
