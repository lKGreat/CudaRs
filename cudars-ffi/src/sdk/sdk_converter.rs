//! Model conversion APIs - PaddleOCR to OpenVINO IR conversion

use libc::c_char;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, UNIX_EPOCH};
use std::{fs, io};
use std::io::Read;

use cudars_core::SdkErr;
use wait_timeout::ChildExt;

use super::sdk_error::{clear_last_error, set_last_error, with_panic_boundary_err};
use super::sdk_strings::read_utf8;

/// Options for paddle OCR to IR conversion (passed as JSON)
#[derive(Debug, serde::Deserialize)]
#[serde(default)]
pub struct ConversionOptions {
    /// ONNX opset version (default: 11)
    pub opset_version: i32,
    /// Compress weights to FP16 (default: false)
    pub compress_to_fp16: bool,
    /// Enable ONNX validation (default: true)
    pub enable_validation: bool,
    /// Force reconversion even if cached (default: false)
    pub force_reconvert: bool,
    /// Timeout in seconds (default: 300, 0 = no timeout)
    pub timeout_secs: u64,
    /// Cache directory (empty = use temp)
    pub cache_dir: String,
}

impl Default for ConversionOptions {
    fn default() -> Self {
        Self {
            opset_version: 11,
            compress_to_fp16: false,
            enable_validation: true,
            force_reconvert: false,
            timeout_secs: 300,
            cache_dir: String::new(),
        }
    }
}

#[derive(Debug)]
enum ConvertError {
    NotFound(String),
    MissingDependency(String),
    Timeout(String),
    Io(String),
    Runtime(String),
}

impl ConvertError {
    fn message(&self) -> &str {
        match self {
            ConvertError::NotFound(msg)
            | ConvertError::MissingDependency(msg)
            | ConvertError::Timeout(msg)
            | ConvertError::Io(msg)
            | ConvertError::Runtime(msg) => msg,
        }
    }

    fn sdk_err(&self) -> SdkErr {
        match self {
            ConvertError::NotFound(_) => SdkErr::NotFound,
            ConvertError::MissingDependency(_) => SdkErr::MissingDependency,
            ConvertError::Timeout(_) => SdkErr::Timeout,
            ConvertError::Io(_) => SdkErr::Io,
            ConvertError::Runtime(_) => SdkErr::Runtime,
        }
    }
}

impl From<io::Error> for ConvertError {
    fn from(err: io::Error) -> Self {
        ConvertError::Io(err.to_string())
    }
}

type ConvResult<T> = Result<T, ConvertError>;

/// Get the directory containing the current dynamic library
#[cfg(target_os = "windows")]
fn get_library_directory() -> ConvResult<PathBuf> {
    use std::ffi::OsString;
    use std::os::windows::ffi::OsStringExt;
    use winapi::shared::minwindef::{DWORD, HMODULE};
    use winapi::um::libloaderapi::{GetModuleFileNameW, GetModuleHandleExW};
    use winapi::um::winnt::WCHAR;

    unsafe {
        let mut module: HMODULE = std::ptr::null_mut();
        let flags = 0x00000004 | 0x00000002; // GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT

        let addr = get_library_directory as *const () as *const WCHAR;
        if GetModuleHandleExW(flags, addr, &mut module) == 0
        {
            return Err(ConvertError::Runtime("GetModuleHandleExW failed".to_string()));
        }

        let mut buffer = vec![0u16; 32768];
        let len = GetModuleFileNameW(module, buffer.as_mut_ptr(), buffer.len() as DWORD);

        if len == 0 {
            return Err(ConvertError::Runtime("GetModuleFileNameW failed".to_string()));
        }

        buffer.truncate(len as usize);
        let path = PathBuf::from(OsString::from_wide(&buffer));

        path.parent()
            .map(|p| p.to_path_buf())
            .ok_or_else(|| ConvertError::Runtime("Failed to get parent directory".to_string()))
    }
}

/// Get the directory containing the current dynamic library
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn get_library_directory() -> ConvResult<PathBuf> {
    use libc::{dladdr, Dl_info};
    use std::ffi::CStr;

    unsafe {
        let mut info: Dl_info = std::mem::zeroed();
        if dladdr(get_library_directory as *const _, &mut info) == 0 {
            return Err(ConvertError::Runtime("dladdr failed".to_string()));
        }

        if info.dli_fname.is_null() {
            return Err(ConvertError::Runtime("dli_fname is null".to_string()));
        }

        let path_cstr = CStr::from_ptr(info.dli_fname);
        let path_str = path_cstr
            .to_str()
            .map_err(|_| ConvertError::Runtime("Invalid UTF-8 in library path".to_string()))?;
        let path = PathBuf::from(path_str);

        path.parent()
            .map(|p| p.to_path_buf())
            .ok_or_else(|| ConvertError::Runtime("Failed to get parent directory".to_string()))
    }
}

/// Locate Python runtime bundled with the library
fn find_bundled_python() -> ConvResult<PathBuf> {
    let lib_dir = get_library_directory()?;

    #[cfg(target_os = "windows")]
    let python_exe = lib_dir.join("python").join("python.exe");

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    let python_exe = lib_dir.join("python").join("bin").join("python3");

    if python_exe.exists() {
        Ok(python_exe)
    } else {
        Err(ConvertError::MissingDependency(format!(
            "Bundled Python not found at {:?}",
            python_exe
        )))
    }
}

/// Locate conversion scripts bundled with the library
fn find_conversion_scripts() -> ConvResult<PathBuf> {
    let lib_dir = get_library_directory()?;
    let scripts_dir = lib_dir.join("scripts");

    if !scripts_dir.exists() {
        return Err(ConvertError::NotFound(format!(
            "Scripts directory not found at {:?}",
            scripts_dir
        )));
    }

    let paddle2onnx_script = scripts_dir.join("paddle2onnx_converter.py");
    let onnx2ir_script = scripts_dir.join("onnx_to_openvino_ir.py");

    if !paddle2onnx_script.exists() {
        return Err(ConvertError::NotFound(format!(
            "paddle2onnx_converter.py not found at {:?}",
            paddle2onnx_script
        )));
    }
    if !onnx2ir_script.exists() {
        return Err(ConvertError::NotFound(format!(
            "onnx_to_openvino_ir.py not found at {:?}",
            onnx2ir_script
        )));
    }

    Ok(scripts_dir)
}

/// Setup environment for bundled Python
fn setup_python_env() -> ConvResult<(PathBuf, Vec<(String, String)>)> {
    let lib_dir = get_library_directory()?;
    let python_exe = find_bundled_python()?;
    let python_home = lib_dir.join("python");
    let site_packages = lib_dir.join("site-packages");

    let mut env_vars = Vec::new();
    env_vars.push((
        "PYTHONHOME".to_string(),
        python_home.to_string_lossy().to_string(),
    ));
    env_vars.push(("PYTHONNOUSERSITE".to_string(), "1".to_string()));
    env_vars.push(("PYTHONDONTWRITEBYTECODE".to_string(), "1".to_string()));
    env_vars.push(("PYTHONUTF8".to_string(), "1".to_string()));

    if site_packages.exists() {
        let sep = if cfg!(windows) { ';' } else { ':' };
        let mut python_path = site_packages.to_string_lossy().to_string();
        if let Ok(existing) = std::env::var("PYTHONPATH") {
            if !existing.is_empty() {
                python_path.push(sep);
                python_path.push_str(&existing);
            }
        }
        env_vars.push(("PYTHONPATH".to_string(), python_path));
    }

    Ok((python_exe, env_vars))
}

fn timeout_from_options(options: &ConversionOptions) -> Option<Duration> {
    if options.timeout_secs == 0 {
        None
    } else {
        Some(Duration::from_secs(options.timeout_secs))
    }
}

/// Run Python script with timeout
fn run_python_script(
    python_exe: &Path,
    script_path: &Path,
    args: &[&str],
    env_vars: &[(String, String)],
    timeout: Option<Duration>,
) -> ConvResult<String> {
    let mut cmd = Command::new(python_exe);
    cmd.arg(script_path);
    cmd.args(args);
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    for (key, value) in env_vars {
        cmd.env(key, value);
    }

    let mut child = cmd.spawn().map_err(|e| {
        ConvertError::MissingDependency(format!("Failed to execute Python: {}", e))
    })?;

    let output = match timeout {
        Some(timeout) => match child.wait_timeout(timeout).map_err(|e| {
            ConvertError::Runtime(format!("Failed to wait for Python: {}", e))
        })? {
            Some(status) => {
                let mut stdout_buf = Vec::new();
                let mut stderr_buf = Vec::new();
                if let Some(mut stdout) = child.stdout.take() {
                    let _ = stdout.read_to_end(&mut stdout_buf);
                }
                if let Some(mut stderr) = child.stderr.take() {
                    let _ = stderr.read_to_end(&mut stderr_buf);
                }
                std::process::Output {
                    status,
                    stdout: stdout_buf,
                    stderr: stderr_buf,
                }
            }
            None => {
                let _ = child.kill();
                let _ = child.wait();
                let mut stdout_buf = Vec::new();
                let mut stderr_buf = Vec::new();
                if let Some(mut stdout) = child.stdout.take() {
                    let _ = stdout.read_to_end(&mut stdout_buf);
                }
                if let Some(mut stderr) = child.stderr.take() {
                    let _ = stderr.read_to_end(&mut stderr_buf);
                }
                let stdout = String::from_utf8_lossy(&stdout_buf);
                let stderr = String::from_utf8_lossy(&stderr_buf);
                return Err(ConvertError::Timeout(format!(
                    "Python script timed out after {}s\nStdout: {}\nStderr: {}",
                    timeout.as_secs(),
                    stdout,
                    stderr
                )));
            }
        },
        None => child
            .wait_with_output()
            .map_err(|e| ConvertError::Runtime(format!("Failed to capture Python output: {}", e)))?,
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !output.status.success() {
        return Err(ConvertError::Runtime(format!(
            "Python script failed with exit code {:?}\nStdout: {}\nStderr: {}",
            output.status.code(),
            stdout,
            stderr
        )));
    }

    Ok(stdout.to_string())
}

fn ensure_model_files(model_dir: &Path) -> ConvResult<(PathBuf, PathBuf)> {
    if !model_dir.exists() {
        return Err(ConvertError::NotFound(format!(
            "Model directory not found: {:?}",
            model_dir
        )));
    }

    let json_path = model_dir.join("inference.json");
    let params_path = model_dir.join("inference.pdiparams");

    if !json_path.exists() {
        return Err(ConvertError::NotFound(format!(
            "Model file not found: {:?}",
            json_path
        )));
    }
    if !params_path.exists() {
        return Err(ConvertError::NotFound(format!(
            "Parameters file not found: {:?}",
            params_path
        )));
    }

    Ok((json_path, params_path))
}

fn modified_secs(meta: &fs::Metadata) -> u64 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn cache_key_for_model(
    json_path: &Path,
    params_path: &Path,
    options: &ConversionOptions,
) -> ConvResult<String> {
    let json_meta = fs::metadata(json_path)?;
    let params_meta = fs::metadata(params_path)?;

    let json_ts = modified_secs(&json_meta);
    let params_ts = modified_secs(&params_meta);
    let json_size = json_meta.len();
    let params_size = params_meta.len();

    Ok(format!(
        "j{}_s{}_p{}_s{}_op{}_val{}",
        json_ts,
        json_size,
        params_ts,
        params_size,
        options.opset_version,
        if options.enable_validation { 1 } else { 0 }
    ))
}

/// Convert PaddlePaddle model to ONNX
fn convert_paddle_to_onnx(
    model_dir: &Path,
    output_path: &Path,
    options: &ConversionOptions,
) -> ConvResult<PathBuf> {
    let (python_exe, env_vars) = setup_python_env()?;
    let scripts_dir = find_conversion_scripts()?;
    let script = scripts_dir.join("paddle2onnx_converter.py");

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let model_dir_arg = model_dir.to_string_lossy().to_string();
    let output_arg = output_path.to_string_lossy().to_string();
    let opset_arg = options.opset_version.to_string();
    let mut args = vec![
        "--model_dir".to_string(),
        model_dir_arg,
        "--output".to_string(),
        output_arg,
        "--opset_version".to_string(),
        opset_arg,
    ];

    if !options.enable_validation {
        args.push("--no-validation".to_string());
    }

    let args_ref: Vec<&str> = args.iter().map(|v| v.as_str()).collect();
    run_python_script(
        &python_exe,
        &script,
        &args_ref,
        &env_vars,
        timeout_from_options(options),
    )?;

    if !output_path.exists() {
        return Err(ConvertError::Runtime(format!(
            "ONNX output not found at {:?}",
            output_path
        )));
    }

    Ok(output_path.to_path_buf())
}

/// Convert ONNX model to OpenVINO IR
fn convert_onnx_to_ir(
    onnx_path: &Path,
    output_dir: &Path,
    model_name: &str,
    options: &ConversionOptions,
) -> ConvResult<PathBuf> {
    let (python_exe, env_vars) = setup_python_env()?;
    let scripts_dir = find_conversion_scripts()?;
    let script = scripts_dir.join("onnx_to_openvino_ir.py");

    fs::create_dir_all(output_dir)?;

    let input_arg = onnx_path.to_string_lossy().to_string();
    let output_arg = output_dir.to_string_lossy().to_string();
    let mut args = vec![
        "--input".to_string(),
        input_arg,
        "--output_dir".to_string(),
        output_arg,
        "--model_name".to_string(),
        model_name.to_string(),
    ];

    if options.compress_to_fp16 {
        args.push("--compress_to_fp16".to_string());
    }

    let args_ref: Vec<&str> = args.iter().map(|v| v.as_str()).collect();
    run_python_script(
        &python_exe,
        &script,
        &args_ref,
        &env_vars,
        timeout_from_options(options),
    )?;

    let xml_path = output_dir.join(format!("{}.xml", model_name));
    let bin_path = output_dir.join(format!("{}.bin", model_name));

    if !xml_path.exists() || !bin_path.exists() {
        return Err(ConvertError::Runtime(format!(
            "IR output files not found at {:?} / {:?}",
            xml_path, bin_path
        )));
    }

    Ok(xml_path)
}

fn copy_ir_outputs(
    src_xml: &Path,
    src_bin: &Path,
    output_dir: &Path,
    output_xml: &Path,
    output_bin: &Path,
) -> ConvResult<()> {
    fs::create_dir_all(output_dir)?;

    if src_xml != output_xml {
        fs::copy(src_xml, output_xml)?;
    }
    if src_bin != output_bin {
        fs::copy(src_bin, output_bin)?;
    }

    if !output_xml.exists() || !output_bin.exists() {
        return Err(ConvertError::Runtime(format!(
            "IR output files not found after copy at {:?} / {:?}",
            output_xml, output_bin
        )));
    }

    Ok(())
}

fn convert_model_to_ir(
    model_dir: &Path,
    model_name: &str,
    output_dir: &Path,
    cache_root: &Path,
    options: &ConversionOptions,
) -> ConvResult<PathBuf> {
    let (json_path, params_path) = ensure_model_files(model_dir)?;
    let onnx_key = cache_key_for_model(&json_path, &params_path, options)?;

    let onnx_cache_dir = cache_root.join("onnx");
    let ir_cache_root = cache_root.join("ir");
    fs::create_dir_all(&onnx_cache_dir)?;
    fs::create_dir_all(&ir_cache_root)?;

    let onnx_path = onnx_cache_dir.join(format!("{model_name}_{onnx_key}.onnx"));
    let ir_key = format!(
        "{onnx_key}_fp{}",
        if options.compress_to_fp16 { "16" } else { "32" }
    );
    let ir_cache_dir = ir_cache_root.join(format!("{model_name}_{ir_key}"));
    let ir_xml = ir_cache_dir.join(format!("{model_name}.xml"));
    let ir_bin = ir_cache_dir.join(format!("{model_name}.bin"));

    let output_xml = output_dir.join(format!("{model_name}.xml"));
    let output_bin = output_dir.join(format!("{model_name}.bin"));

    if !options.force_reconvert && output_xml.exists() && output_bin.exists() {
        return Ok(output_xml);
    }

    if !options.force_reconvert && ir_xml.exists() && ir_bin.exists() {
        copy_ir_outputs(&ir_xml, &ir_bin, output_dir, &output_xml, &output_bin)?;
        return Ok(output_xml);
    }

    if options.force_reconvert || !onnx_path.exists() {
        convert_paddle_to_onnx(model_dir, &onnx_path, options)?;
    }

    convert_onnx_to_ir(&onnx_path, &ir_cache_dir, model_name, options)?;
    copy_ir_outputs(&ir_xml, &ir_bin, output_dir, &output_xml, &output_bin)?;

    Ok(output_xml)
}

/// Convert PaddleOCR models (detection + recognition) to OpenVINO IR format
///
/// # Safety
/// All pointers must be valid and buffers must have sufficient capacity
#[no_mangle]
pub extern "C" fn sdk_convert_paddle_ocr_to_ir(
    det_model_dir_ptr: *const c_char,
    det_model_dir_len: usize,
    rec_model_dir_ptr: *const c_char,
    rec_model_dir_len: usize,
    output_dir_ptr: *const c_char,
    output_dir_len: usize,
    options_json_ptr: *const c_char,
    options_json_len: usize,
    det_xml_buf: *mut c_char,
    det_xml_cap: usize,
    det_xml_written: *mut usize,
    rec_xml_buf: *mut c_char,
    rec_xml_cap: usize,
    rec_xml_written: *mut usize,
) -> SdkErr {
    with_panic_boundary_err("sdk_convert_paddle_ocr_to_ir", || {
        // Validate pointers
        if det_xml_buf.is_null()
            || det_xml_written.is_null()
            || rec_xml_buf.is_null()
            || rec_xml_written.is_null()
        {
            set_last_error("Output buffer pointers are null");
            return SdkErr::InvalidArg;
        }

        // Read input strings
        let det_model_dir = match read_utf8(det_model_dir_ptr, det_model_dir_len, "det_model_dir")
        {
            Ok(s) => s,
            Err(e) => return e,
        };

        let rec_model_dir = match read_utf8(rec_model_dir_ptr, rec_model_dir_len, "rec_model_dir")
        {
            Ok(s) => s,
            Err(e) => return e,
        };

        let output_dir = match read_utf8(output_dir_ptr, output_dir_len, "output_dir") {
            Ok(s) => s,
            Err(e) => return e,
        };

        if output_dir.is_empty() {
            set_last_error("output_dir is empty");
            return SdkErr::InvalidArg;
        }

        // Parse options
        let options = if options_json_len > 0 {
            let options_json = match read_utf8(options_json_ptr, options_json_len, "options_json") {
                Ok(s) => s,
                Err(e) => return e,
            };
            match serde_json::from_str::<ConversionOptions>(&options_json) {
                Ok(opts) => opts,
                Err(e) => {
                    set_last_error(&format!("Failed to parse options JSON: {}", e));
                    return SdkErr::InvalidArg;
                }
            }
        } else {
            ConversionOptions::default()
        };

        // Determine cache directory
        let cache_dir = if options.cache_dir.is_empty() {
            std::env::temp_dir().join("cudars_conversion_cache")
        } else {
            PathBuf::from(&options.cache_dir)
        };

        if let Err(err) = fs::create_dir_all(&cache_dir) {
            set_last_error(&format!("Failed to create cache directory: {}", err));
            return SdkErr::Io;
        }

        let output_dir = PathBuf::from(output_dir);
        if let Err(err) = fs::create_dir_all(&output_dir) {
            set_last_error(&format!("Failed to create output directory: {}", err));
            return SdkErr::Io;
        }

        // Convert detection model
        let det_dir = PathBuf::from(det_model_dir);
        let det_xml_path = match convert_model_to_ir(&det_dir, "det_model", &output_dir, &cache_dir, &options) {
            Ok(path) => path,
            Err(e) => {
                set_last_error(&format!("Detection model conversion failed: {}", e.message()));
                return e.sdk_err();
            }
        };

        // Convert recognition model
        let rec_dir = PathBuf::from(rec_model_dir);
        let rec_xml_path = match convert_model_to_ir(&rec_dir, "rec_model", &output_dir, &cache_dir, &options) {
            Ok(path) => path,
            Err(e) => {
                set_last_error(&format!("Recognition model conversion failed: {}", e.message()));
                return e.sdk_err();
            }
        };

        // Write output paths
        let det_xml_string = det_xml_path.to_string_lossy().to_string();
        let rec_xml_string = rec_xml_path.to_string_lossy().to_string();
        let det_bytes = det_xml_string.as_bytes();
        let rec_bytes = rec_xml_string.as_bytes();

        unsafe {
            *det_xml_written = det_bytes.len();
            *rec_xml_written = rec_bytes.len();
        }

        if det_bytes.len() > det_xml_cap || rec_bytes.len() > rec_xml_cap {
            set_last_error(&format!(
                "XML path buffer too small: det requires {} bytes, rec requires {} bytes",
                det_bytes.len(),
                rec_bytes.len()
            ));
            return SdkErr::OutOfMemory;
        }

        unsafe {
            std::ptr::copy_nonoverlapping(
                det_bytes.as_ptr(),
                det_xml_buf as *mut u8,
                det_bytes.len(),
            );
            std::ptr::copy_nonoverlapping(
                rec_bytes.as_ptr(),
                rec_xml_buf as *mut u8,
                rec_bytes.len(),
            );
        }

        clear_last_error();
        SdkErr::Ok
    })
}
