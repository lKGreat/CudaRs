use cuda_build::print_build_info;

fn main() {
    print_build_info();
    
    #[cfg(target_os = "windows")]
    {
        // NVML is in the driver installation, not CUDA toolkit
        if let Ok(path) = std::env::var("ProgramFiles") {
            println!("cargo:rustc-link-search=native={}\\NVIDIA Corporation\\NVSMI", path);
        }
        println!("cargo:rustc-link-lib=nvml");
    }
    
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=nvidia-ml");
    }
    
    println!("cargo:rerun-if-changed=build.rs");
}
