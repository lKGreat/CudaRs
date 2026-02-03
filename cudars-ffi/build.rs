use std::env;
use std::path::PathBuf;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let output_dir = PathBuf::from(&crate_dir).join("include");

    std::fs::create_dir_all(&output_dir).ok();

    let mut config = cbindgen::Config::default();
    config.language = cbindgen::Language::C;
    config.include_guard = Some("CUDARS_FFI_H".to_string());
    config.pragma_once = true;
    config.cpp_compat = true;
    config.documentation = true;
    config.documentation_style = cbindgen::DocumentationStyle::C99;

    cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(output_dir.join("cudars.h"));

    if env::var("CARGO_FEATURE_TENSORRT").is_ok() {
        let mut build = cc::Build::new();
        build.cpp(true)
            .file(PathBuf::from(&crate_dir).join("src").join("tensorrt_wrapper.cpp"))
            .flag_if_supported("/std:c++17")
            .flag_if_supported("-std=c++17")
            .flag_if_supported("/EHsc");

        if let Ok(include_dir) = env::var("TENSORRT_INCLUDE") {
            build.include(include_dir);
        } else if let Ok(root) = env::var("TENSORRT_ROOT") {
            build.include(PathBuf::from(root).join("include"));
        }

        if let Ok(cuda_path) = env::var("CUDA_PATH") {
            build.include(PathBuf::from(cuda_path).join("include"));
        } else if let Ok(cuda_home) = env::var("CUDA_HOME") {
            build.include(PathBuf::from(cuda_home).join("include"));
        }

        build.compile("tensorrt_wrapper");
    }

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/tensorrt_wrapper.cpp");
}
