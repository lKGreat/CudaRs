use cuda_build::{cuda_include_path, link_cuda_driver, print_build_info};

fn main() {
    if std::env::var("CARGO_FEATURE_STUB").is_ok() {
        println!("cargo:rerun-if-env-changed=CARGO_FEATURE_STUB");
        return;
    }
    print_build_info();
    link_cuda_driver();

    if let Some(include_path) = cuda_include_path() {
        println!("cargo:include={}", include_path.display());
    }

    println!("cargo:rerun-if-changed=build.rs");
}
