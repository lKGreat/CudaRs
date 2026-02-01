use cuda_build::{cuda_lib_path, print_build_info};

fn main() {
    print_build_info();
    if let Some(lib_path) = cuda_lib_path() {
        println!("cargo:rustc-link-search=native={}", lib_path.display());
    }
    println!("cargo:rustc-link-lib=nppc");
    println!("cargo:rustc-link-lib=nppig");
    println!("cargo:rustc-link-lib=nppicc");
    println!("cargo:rustc-link-lib=nppidei");
    println!("cargo:rustc-link-lib=nppif");
    println!("cargo:rustc-link-lib=nppim");
    println!("cargo:rustc-link-lib=nppist");
    println!("cargo:rustc-link-lib=nppisu");
    println!("cargo:rustc-link-lib=nppitc");
    println!("cargo:rustc-link-lib=npps");
    println!("cargo:rerun-if-changed=build.rs");
}
