use cuda_build::{cuda_lib_path, print_build_info};

fn main() {
    print_build_info();

    if let Some(lib_path) = cuda_lib_path() {
        println!("cargo:rustc-link-search=native={}", lib_path.display());
    }
    println!("cargo:rustc-link-lib=cufft");

    println!("cargo:rerun-if-changed=build.rs");
}
