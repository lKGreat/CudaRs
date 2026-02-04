use std::env;
use std::fs;
use std::path::PathBuf;
use std::path::Path;

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

    if env::var("CARGO_FEATURE_PADDLEOCR").is_ok() {
        if let Err(err) = build_paddleocr(&crate_dir) {
            println!("cargo:warning=paddleocr build skipped: {err}");
        }
    }

    if env::var("CARGO_FEATURE_OPENVINO").is_ok() {
        configure_openvino_link(&crate_dir);
    }

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/tensorrt_wrapper.cpp");
    println!("cargo:rerun-if-changed=src/paddleocr_wrapper.cpp");
    println!("cargo:rerun-if-env-changed=PADDLE_INFERENCE_ROOT");
    println!("cargo:rerun-if-env-changed=PADDLE_LIB");
    println!("cargo:rerun-if-env-changed=OPENCV_DIR");
    println!("cargo:rerun-if-env-changed=OPENCV_ROOT");
    println!("cargo:rerun-if-env-changed=PADDLE_OCR_ROOT");
    println!("cargo:rerun-if-env-changed=PADDLE_OCR_CPP_DIR");
    println!("cargo:rerun-if-env-changed=ABSL_INCLUDE");
    println!("cargo:rerun-if-env-changed=ABSL_ROOT");
    println!("cargo:rerun-if-env-changed=OPENVINO_LIB");
    println!("cargo:rerun-if-env-changed=OPENVINO_ROOT");
    println!("cargo:rerun-if-env-changed=OPENVINO_DIR");
    println!("cargo:rerun-if-env-changed=VIRTUAL_ENV");
}

fn configure_openvino_link(crate_dir: &str) {
    let mut candidates: Vec<PathBuf> = Vec::new();

    if let Ok(lib_dir) = env::var("OPENVINO_LIB") {
        candidates.push(PathBuf::from(lib_dir));
    }

    if let Ok(root) = env::var("OPENVINO_ROOT").or_else(|_| env::var("OPENVINO_DIR")) {
        let root = PathBuf::from(root);
        candidates.push(root.join("libs"));
        candidates.push(root.join("lib"));
        candidates.push(root.join("runtime").join("lib"));
        candidates.push(root.join("runtime").join("lib").join("intel64"));
        candidates.push(
            root.join("Lib")
                .join("site-packages")
                .join("openvino")
                .join("libs"),
        );
    }

    if let Ok(venv) = env::var("VIRTUAL_ENV") {
        candidates.push(
            PathBuf::from(venv)
                .join("Lib")
                .join("site-packages")
                .join("openvino")
                .join("libs"),
        );
    }

    candidates.push(
        PathBuf::from(crate_dir)
            .join("..")
            .join("openvino_env")
            .join("Lib")
            .join("site-packages")
            .join("openvino")
            .join("libs"),
    );

    for candidate in candidates {
        if candidate.join("openvino.lib").exists() || candidate.join("openvino_c.lib").exists() {
            println!("cargo:rustc-link-search=native={}", candidate.display());
            break;
        }
    }
}

fn build_paddleocr(crate_dir: &str) -> Result<(), String> {
    let paddle_infer_root = env::var("PADDLE_INFERENCE_ROOT")
        .or_else(|_| env::var("PADDLE_LIB"))
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            let candidate = PathBuf::from(crate_dir).join("..").join("paddle_inference");
            if candidate.exists() { Some(candidate) } else { None }
        })
        .ok_or_else(|| "PADDLE_INFERENCE_ROOT or PADDLE_LIB not set and ../paddle_inference not found".to_string())?;

    let opencv_root = env::var("OPENCV_DIR")
        .or_else(|_| env::var("OPENCV_ROOT"))
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            let candidate = PathBuf::from(crate_dir).join("..").join("opencv").join("build");
            if candidate.exists() { Some(candidate) } else { None }
        })
        .ok_or_else(|| "OPENCV_DIR or OPENCV_ROOT not set and ../opencv/build not found".to_string())?;

    let paddleocr_cpp_dir = if let Ok(dir) = env::var("PADDLE_OCR_CPP_DIR") {
        PathBuf::from(dir)
    } else if let Ok(root) = env::var("PADDLE_OCR_ROOT") {
        PathBuf::from(root).join("deploy").join("cpp_infer")
    } else {
        let candidate = PathBuf::from(crate_dir).join("..").join("PaddleOCR-3.3.2").join("deploy").join("cpp_infer");
        if candidate.exists() {
            candidate
        } else {
            return Err("PADDLE_OCR_CPP_DIR or PADDLE_OCR_ROOT not set and ../PaddleOCR-3.3.2/deploy/cpp_infer not found".to_string());
        }
    };

    let paddleocr_src = paddleocr_cpp_dir.join("src");
    if !paddleocr_src.exists() {
        return Err(format!("PaddleOCR cpp_infer src not found at {}", paddleocr_src.display()));
    }

    let mut build = cc::Build::new();
    build.cpp(true);
    build.flag_if_supported("/std:c++17");
    build.flag_if_supported("-std=c++17");
    build.flag_if_supported("/EHsc");

    build.file(PathBuf::from(crate_dir).join("src").join("paddleocr_wrapper.cpp"));

    let mut cpp_sources = Vec::new();
    collect_cc_files(&paddleocr_src, &mut cpp_sources)?;
    for file in cpp_sources {
        build.file(file);
    }

    // Includes: PaddleOCR sources + local shims + Paddle Inference + OpenCV + third_party
    build.include(&paddleocr_cpp_dir);
    build.include(PathBuf::from(crate_dir));
    build.include(&paddle_infer_root.join("paddle").join("include"));
    build.include(&paddle_infer_root.join("third_party").join("install").join("protobuf").join("include"));
    build.include(&paddle_infer_root.join("third_party").join("install").join("glog").join("include"));
    build.include(&paddle_infer_root.join("third_party").join("install").join("gflags").join("include"));
    build.include(&paddle_infer_root.join("third_party").join("install").join("xxhash").join("include"));
    build.include(&paddle_infer_root.join("third_party").join("install").join("zlib").join("include"));
    build.include(&paddle_infer_root.join("third_party").join("install").join("onnxruntime").join("include"));
    build.include(&paddle_infer_root.join("third_party").join("install").join("paddle2onnx").join("include"));
    build.include(&paddle_infer_root.join("third_party").join("install").join("yaml-cpp").join("include"));
    build.include(&paddle_infer_root.join("third_party").join("install").join("openvino").join("include"));
    build.include(&paddle_infer_root.join("third_party").join("install").join("tbb").join("include"));
    build.include(&paddle_infer_root.join("third_party").join("boost"));
    build.include(&paddle_infer_root.join("third_party").join("eigen3"));

    if let Ok(absl_inc) = env::var("ABSL_INCLUDE") {
        build.include(absl_inc);
    } else if let Ok(absl_root) = env::var("ABSL_ROOT") {
        build.include(PathBuf::from(absl_root).join("include"));
    }

    // Abseil + vcpkg libs (status/strings are required by PaddleOCR)
    if let Ok(absl_lib) = env::var("ABSL_LIB") {
        println!("cargo:rustc-link-search=native={}", absl_lib);
    } else if let Ok(absl_root) = env::var("ABSL_ROOT") {
        println!(
            "cargo:rustc-link-search=native={}",
            PathBuf::from(absl_root).join("lib").display()
        );
    } else {
        let vcpkg_lib = PathBuf::from("C:\\vcpkg\\installed\\x64-windows\\lib");
        if vcpkg_lib.exists() {
            println!("cargo:rustc-link-search=native={}", vcpkg_lib.display());
        }
    }

    // Clipper (polyclipping) is used by PaddleOCR post-process
    let vcpkg_lib = PathBuf::from("C:\\vcpkg\\installed\\x64-windows\\lib");
    if vcpkg_lib.exists() {
        println!("cargo:rustc-link-search=native={}", vcpkg_lib.display());
    }

    // OpenCV include
    build.include(&opencv_root.join("include"));

    // Windows-specific defines used by PaddleOCR demo build.
    build.define("GOOGLE_GLOG_DLL_DECL", Some(""));
    if env::var("PADDLEOCR_WITH_MKL").unwrap_or_else(|_| "1".to_string()) == "1" {
        build.define("USE_MKL", None);
    }

    // Link search paths for Paddle Inference and OpenCV
    println!("cargo:rustc-link-search=native={}", paddle_infer_root.join("paddle").join("lib").display());
    println!("cargo:rustc-link-search=native={}", paddle_infer_root.join("third_party").join("install").join("protobuf").join("lib").display());
    println!("cargo:rustc-link-search=native={}", paddle_infer_root.join("third_party").join("install").join("glog").join("lib").display());
    println!("cargo:rustc-link-search=native={}", paddle_infer_root.join("third_party").join("install").join("gflags").join("lib").display());
    println!("cargo:rustc-link-search=native={}", paddle_infer_root.join("third_party").join("install").join("xxhash").join("lib").display());
    println!("cargo:rustc-link-search=native={}", paddle_infer_root.join("third_party").join("install").join("zlib").join("lib").display());
    println!("cargo:rustc-link-search=native={}", paddle_infer_root.join("third_party").join("install").join("onnxruntime").join("lib").display());
    println!("cargo:rustc-link-search=native={}", paddle_infer_root.join("third_party").join("install").join("paddle2onnx").join("lib").display());
    println!("cargo:rustc-link-search=native={}", paddle_infer_root.join("third_party").join("install").join("openvino").join("intel64").display());
    println!("cargo:rustc-link-search=native={}", paddle_infer_root.join("third_party").join("install").join("tbb").join("lib").display());

    let opencv_lib_dir = if cfg!(windows) {
        opencv_root.join("x64").join("vc16").join("lib")
    } else {
        opencv_root.join("lib64")
    };
    println!("cargo:rustc-link-search=native={}", opencv_lib_dir.display());

    // Link Paddle Inference and common helper libs.
    if cfg!(windows) {
        println!("cargo:rustc-link-lib=paddle_inference");
        println!("cargo:rustc-link-lib=common");
        println!("cargo:rustc-link-lib=yaml-cpp");
        println!("cargo:rustc-link-lib=polyclipping");
        println!("cargo:rustc-link-lib=abseil_dll");
    } else {
        println!("cargo:rustc-link-lib=paddle_inference");
    }

    // Try linking OpenCV world if available.
    if cfg!(windows) {
        if let Some(name) = find_opencv_world(&opencv_lib_dir) {
            println!("cargo:rustc-link-lib={name}");
        } else {
            println!("cargo:warning=OpenCV world library not found in {}", opencv_lib_dir.display());
        }
    } else {
        println!("cargo:rustc-link-lib=opencv_core");
        println!("cargo:rustc-link-lib=opencv_imgcodecs");
        println!("cargo:rustc-link-lib=opencv_imgproc");
    }

    build.compile("paddleocr_wrapper");
    Ok(())
}

fn collect_cc_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = std::fs::read_dir(dir).map_err(|e| format!("read_dir failed: {e}"))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("read_dir entry failed: {e}"))?;
        let path = entry.path();
        if path.is_dir() {
            collect_cc_files(&path, out)?;
        } else if let Some(ext) = path.extension() {
            if ext == "cc" {
                out.push(path);
            }
        }
    }
    Ok(())
}

fn find_opencv_world(dir: &Path) -> Option<String> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if let Some(name) = path.file_stem().and_then(|n| n.to_str()) {
            if name.starts_with("opencv_world") {
                return Some(name.to_string());
            }
        }
    }
    None
}
