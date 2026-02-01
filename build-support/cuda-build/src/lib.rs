//! Build utilities for CUDA FFI crates.
//!
//! Provides functions to detect CUDA installation paths and configure linking.

use std::env;
use std::path::PathBuf;

/// CUDA library information
#[derive(Debug, Clone)]
pub struct CudaLib {
    pub name: &'static str,
    pub header: &'static str,
    pub lib_name: &'static str,
}

/// Common CUDA libraries
pub mod libs {
    use super::CudaLib;

    pub const CUDA_RUNTIME: CudaLib = CudaLib {
        name: "CUDA Runtime",
        header: "cuda_runtime.h",
        lib_name: "cudart",
    };

    pub const CUDA_DRIVER: CudaLib = CudaLib {
        name: "CUDA Driver",
        header: "cuda.h",
        lib_name: "cuda",
    };

    pub const CUBLAS: CudaLib = CudaLib {
        name: "cuBLAS",
        header: "cublas_v2.h",
        lib_name: "cublas",
    };

    pub const CUFFT: CudaLib = CudaLib {
        name: "cuFFT",
        header: "cufft.h",
        lib_name: "cufft",
    };

    pub const CURAND: CudaLib = CudaLib {
        name: "cuRAND",
        header: "curand.h",
        lib_name: "curand",
    };

    pub const CUSPARSE: CudaLib = CudaLib {
        name: "cuSPARSE",
        header: "cusparse.h",
        lib_name: "cusparse",
    };

    pub const CUSOLVER: CudaLib = CudaLib {
        name: "cusolverDn",
        header: "cusolverDn.h",
        lib_name: "cusolver",
    };

    pub const CUDNN: CudaLib = CudaLib {
        name: "cuDNN",
        header: "cudnn.h",
        lib_name: "cudnn",
    };

    pub const NVRTC: CudaLib = CudaLib {
        name: "NVRTC",
        header: "nvrtc.h",
        lib_name: "nvrtc",
    };

    pub const NVJPEG: CudaLib = CudaLib {
        name: "nvJPEG",
        header: "nvjpeg.h",
        lib_name: "nvjpeg",
    };

    pub const NPP: CudaLib = CudaLib {
        name: "NPP",
        header: "npp.h",
        lib_name: "nppc",
    };

    pub const CUPTI: CudaLib = CudaLib {
        name: "CUPTI",
        header: "cupti.h",
        lib_name: "cupti",
    };

    pub const NVML: CudaLib = CudaLib {
        name: "NVML",
        header: "nvml.h",
        lib_name: "nvml",
    };
}

/// Detects the CUDA installation path.
///
/// Checks in order:
/// 1. `CUDA_PATH` environment variable
/// 2. `CUDA_HOME` environment variable
/// 3. Default Windows path: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`
/// 4. Default Linux path: `/usr/local/cuda`
pub fn detect_cuda_path() -> Option<PathBuf> {
    // Check environment variables first
    if let Ok(path) = env::var("CUDA_PATH") {
        let p = PathBuf::from(&path);
        if p.exists() {
            return Some(p);
        }
    }

    if let Ok(path) = env::var("CUDA_HOME") {
        let p = PathBuf::from(&path);
        if p.exists() {
            return Some(p);
        }
    }

    // Platform-specific defaults
    #[cfg(target_os = "windows")]
    {
        // Try common Windows paths
        let versions = ["v12.6", "v12.5", "v12.4", "v12.3", "v12.2", "v12.1", "v12.0", "v11.8"];
        for ver in versions {
            let path = PathBuf::from(format!(
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\{ver}"
            ));
            if path.exists() {
                return Some(path);
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        let path = PathBuf::from("/usr/local/cuda");
        if path.exists() {
            return Some(path);
        }
    }

    None
}

/// Returns the library search path for CUDA.
pub fn cuda_lib_path() -> Option<PathBuf> {
    let cuda_path = detect_cuda_path()?;

    #[cfg(target_os = "windows")]
    {
        Some(cuda_path.join("lib").join("x64"))
    }

    #[cfg(target_os = "linux")]
    {
        Some(cuda_path.join("lib64"))
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        Some(cuda_path.join("lib"))
    }
}

/// Returns the include path for CUDA headers.
pub fn cuda_include_path() -> Option<PathBuf> {
    detect_cuda_path().map(|p| p.join("include"))
}

/// Emits cargo directives to link a CUDA library.
///
/// This function should be called from a `build.rs` script.
pub fn link_cuda_lib(lib: &CudaLib) {
    if let Some(lib_path) = cuda_lib_path() {
        println!("cargo:rustc-link-search=native={}", lib_path.display());
    }
    println!("cargo:rustc-link-lib={}", lib.lib_name);
}

/// Emits cargo directives to link the CUDA runtime.
pub fn link_cuda_runtime() {
    link_cuda_lib(&libs::CUDA_RUNTIME);
}

/// Emits cargo directives to link the CUDA driver API.
pub fn link_cuda_driver() {
    link_cuda_lib(&libs::CUDA_DRIVER);
}

/// Detects and returns the CUDA version as (major, minor).
pub fn detect_cuda_version() -> Option<(u32, u32)> {
    let cuda_path = detect_cuda_path()?;

    // Try to parse from path (Windows style: vX.Y)
    if let Some(name) = cuda_path.file_name().and_then(|n| n.to_str()) {
        if let Some(ver) = name.strip_prefix('v') {
            let parts: Vec<&str> = ver.split('.').collect();
            if parts.len() >= 2 {
                if let (Ok(major), Ok(minor)) = (parts[0].parse(), parts[1].parse()) {
                    return Some((major, minor));
                }
            }
        }
    }

    // Try to read version.txt or version.json
    let version_file = cuda_path.join("version.txt");
    if version_file.exists() {
        if let Ok(content) = std::fs::read_to_string(&version_file) {
            // Format: "CUDA Version X.Y.Z"
            if let Some(ver_str) = content.strip_prefix("CUDA Version ") {
                let parts: Vec<&str> = ver_str.trim().split('.').collect();
                if parts.len() >= 2 {
                    if let (Ok(major), Ok(minor)) = (parts[0].parse(), parts[1].parse()) {
                        return Some((major, minor));
                    }
                }
            }
        }
    }

    None
}

/// Prints build information for debugging.
pub fn print_build_info() {
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    let _ = detect_cuda_path();
    let _ = detect_cuda_version();
}

/// Configuration for bindgen.
#[derive(Debug, Clone)]
pub struct BindgenConfig {
    pub header: String,
    pub allowlist_functions: Vec<String>,
    pub allowlist_types: Vec<String>,
    pub allowlist_vars: Vec<String>,
}

impl BindgenConfig {
    pub fn new(header: &str) -> Self {
        Self {
            header: header.to_string(),
            allowlist_functions: Vec::new(),
            allowlist_types: Vec::new(),
            allowlist_vars: Vec::new(),
        }
    }

    pub fn allowlist_function(mut self, pattern: &str) -> Self {
        self.allowlist_functions.push(pattern.to_string());
        self
    }

    pub fn allowlist_type(mut self, pattern: &str) -> Self {
        self.allowlist_types.push(pattern.to_string());
        self
    }

    pub fn allowlist_var(mut self, pattern: &str) -> Self {
        self.allowlist_vars.push(pattern.to_string());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_cuda_path() {
        // This test may fail if CUDA is not installed
        let path = detect_cuda_path();
        println!("Detected CUDA path: {:?}", path);
    }
}
