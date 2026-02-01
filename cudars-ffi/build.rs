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

    println!("cargo:rerun-if-changed=src/lib.rs");
}
