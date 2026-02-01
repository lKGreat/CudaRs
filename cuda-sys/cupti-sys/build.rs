use cuda_build::{detect_cuda_path, print_build_info};

fn main() {
    print_build_info();
    if let Some(cuda_path) = detect_cuda_path() {
        #[cfg(target_os = "windows")]
        println!("cargo:rustc-link-search=native={}\\extras\\CUPTI\\lib64", cuda_path.display());
        #[cfg(target_os = "linux")]
        println!("cargo:rustc-link-search=native={}/extras/CUPTI/lib64", cuda_path.display());
    }
    println!("cargo:rustc-link-lib=cupti");
    println!("cargo:rerun-if-changed=build.rs");
}
