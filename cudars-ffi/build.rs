use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let output_dir = PathBuf::from(&crate_dir).join("include");

    std::fs::create_dir_all(&output_dir).ok();

    let mut config = cbindgen::Config::default();
    config.language = cbindgen::Language::C;
    config.include_guard = Some("CUDARS_SDK_H".to_string());
    config.pragma_once = true;
    config.cpp_compat = true;
    config.documentation = true;
    config.documentation_style = cbindgen::DocumentationStyle::C99;

    match cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
    {
        Ok(bindings) => {
            bindings.write_to_file(output_dir.join("sdk.h"));
        }
        Err(e) => {
            // Don't fail the build if header generation fails.
            // This is a dev convenience: the Rust crate can still compile and the
            // .NET P/Invoke layer does not depend on cbindgen at build time.
            println!("cargo:warning=cbindgen failed: {e}");
        }
    }

    if env::var("CARGO_FEATURE_TENSORRT").is_ok() {
        let mut tensorrt_include: Option<PathBuf> = None;
        let mut tensorrt_lib: Option<PathBuf> = None;

        if let Ok(include_dir) = env::var("TENSORRT_INCLUDE") {
            tensorrt_include = Some(PathBuf::from(include_dir));
        }
        if let Ok(lib_dir) = env::var("TENSORRT_LIB") {
            tensorrt_lib = Some(PathBuf::from(lib_dir));
        }
        if tensorrt_include.is_none() || tensorrt_lib.is_none() {
            if let Ok(root) = env::var("TENSORRT_ROOT") {
                let root = PathBuf::from(root);
                if tensorrt_include.is_none() {
                    tensorrt_include = Some(root.join("include"));
                }
                if tensorrt_lib.is_none() {
                    tensorrt_lib = Some(root.join("lib"));
                }
            }
        }

        let mut build = cc::Build::new();
        build.cpp(true)
            .file(PathBuf::from(&crate_dir).join("src").join("tensorrt_wrapper.cpp"))
            .flag_if_supported("/std:c++17")
            .flag_if_supported("-std=c++17")
            .flag_if_supported("/EHsc");

        if let Some(inc) = &tensorrt_include {
            build.include(inc);
        }

        if let Ok(cuda_path) = env::var("CUDA_PATH") {
            build.include(PathBuf::from(cuda_path).join("include"));
        } else if let Ok(cuda_home) = env::var("CUDA_HOME") {
            build.include(PathBuf::from(cuda_home).join("include"));
        }

        if let Some(lib) = &tensorrt_lib {
            println!("cargo:rustc-link-search=native={}", lib.display());

            // TensorRT libraries (Windows: .lib import libraries; runtime requires DLLs on PATH)
            println!("cargo:rustc-link-lib=nvinfer");
            println!("cargo:rustc-link-lib=nvonnxparser");

            // Plugin library name differs by TensorRT major version on Windows.
            // Try common candidates in priority order.
            let plugin_candidates = [
                ("nvinfer_plugin", "nvinfer_plugin.lib"),
                ("nvinfer_plugin_10", "nvinfer_plugin_10.lib"),
                ("nvinfer_plugin_9", "nvinfer_plugin_9.lib"),
            ];

            let mut linked_plugin = false;
            for (link_name, file_name) in plugin_candidates {
                if fs::metadata(lib.join(file_name)).is_ok() {
                    println!("cargo:rustc-link-lib={}", link_name);
                    linked_plugin = true;
                    break;
                }
            }

            // Some Windows TensorRT distributions also ship a VC plugin import lib.
            // Link it if present (harmless if unused, required if referenced).
            if fs::metadata(lib.join("nvinfer_vc_plugin_10.lib")).is_ok() {
                println!("cargo:rustc-link-lib=nvinfer_vc_plugin_10");
            }

            if !linked_plugin {
                // Best effort: keep previous behavior for non-standard layouts.
                println!("cargo:rustc-link-lib=nvinfer_plugin");
            }
        }

        build.compile("tensorrt_wrapper");
    }

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/tensorrt_wrapper.cpp");
}
