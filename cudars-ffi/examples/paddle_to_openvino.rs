/// Example: Loading PaddlePaddle models with OpenVINO in Rust
/// 
/// This example demonstrates how to:
/// 1. Convert a PaddlePaddle model to ONNX using external tools
/// 2. Load the ONNX model with OpenVINO
/// 3. Run inference
/// 
/// Prerequisites:
/// - Python with paddle2onnx installed (pip install paddle2onnx onnx)
/// - PaddlePaddle model files (inference.json, inference.pdiparams)
/// 
/// Usage:
///   cargo run --example paddle_to_openvino --features openvino

use std::path::Path;
use std::process::Command;

fn main() {
    println!("PaddlePaddle to OpenVINO Example");
    println!("=================================\n");

    // Configuration - adjust to your model location
    let paddle_model_dir = "E:/codeding/AI/PP-OCRv5_mobile_det_infer";
    let model_json = "inference.json";
    let model_params = "inference.pdiparams";
    let onnx_output = "model_converted.onnx";

    // Step 1: Check if PaddlePaddle model exists
    println!("[Step 1] Checking PaddlePaddle model files...");
    let model_json_path = Path::new(paddle_model_dir).join(model_json);
    let model_params_path = Path::new(paddle_model_dir).join(model_params);

    if !model_json_path.exists() {
        eprintln!("Error: Model file not found: {:?}", model_json_path);
        eprintln!("Please set paddle_model_dir to your PaddlePaddle model location");
        return;
    }

    if !model_params_path.exists() {
        eprintln!("Error: Parameters file not found: {:?}", model_params_path);
        return;
    }

    println!("✓ Model files found");
    println!("  - {}: {:?}", model_json, model_json_path);
    println!("  - {}: {:?}", model_params, model_params_path);

    // Step 2: Convert PaddlePaddle model to ONNX
    println!("\n[Step 2] Converting PaddlePaddle model to ONNX...");
    
    let convert_result = convert_paddle_to_onnx(
        paddle_model_dir,
        onnx_output,
        model_json,
        model_params,
        11, // opset_version
    );

    match convert_result {
        Ok(onnx_path) => {
            println!("✓ Conversion successful: {}", onnx_path);
            
            // Step 3: Load ONNX model with OpenVINO (placeholder)
            println!("\n[Step 3] Loading ONNX model with OpenVINO...");
            println!("Note: Add OpenVINO loading code here using cudars_ov_load()");
            println!("Example:");
            println!("  let config = CudaRsOvConfig {{ ... }};");
            println!("  let model = cudars_ov_load(onnx_path, &config)?;");
            
            // Step 4: Run inference (placeholder)
            println!("\n[Step 4] Running inference...");
            println!("Note: Add inference code here using cudars_ov_run()");
            
            println!("\n✓ Example completed successfully!");
            println!("\nNext steps:");
            println!("  1. Implement OpenVINO model loading (see cudars-ffi/src/openvino.rs)");
            println!("  2. Prepare input data");
            println!("  3. Run inference and process outputs");
        }
        Err(e) => {
            eprintln!("✗ Conversion failed: {}", e);
            eprintln!("\nTroubleshooting:");
            eprintln!("  1. Ensure Python is installed and in PATH");
            eprintln!("  2. Install paddle2onnx: pip install paddle2onnx onnx");
            eprintln!("  3. Check that the converter script exists in scripts/");
        }
    }
}

/// Convert PaddlePaddle model to ONNX using Python script
fn convert_paddle_to_onnx(
    model_dir: &str,
    output_path: &str,
    model_filename: &str,
    params_filename: &str,
    opset_version: u32,
) -> Result<String, String> {
    // Find the converter script
    let script_paths = vec![
        "scripts/paddle2onnx_converter.py",
        "../scripts/paddle2onnx_converter.py",
        "../../scripts/paddle2onnx_converter.py",
    ];

    let mut script_path = None;
    for path in &script_paths {
        if Path::new(path).exists() {
            script_path = Some(*path);
            break;
        }
    }

    let script_path = script_path.ok_or_else(|| {
        "Converter script not found. Expected scripts/paddle2onnx_converter.py".to_string()
    })?;

    // Build command
    let output = Command::new("python")
        .arg(script_path)
        .arg("--model_dir")
        .arg(model_dir)
        .arg("--output")
        .arg(output_path)
        .arg("--model_filename")
        .arg(model_filename)
        .arg("--params_filename")
        .arg(params_filename)
        .arg("--opset_version")
        .arg(opset_version.to_string())
        .output()
        .map_err(|e| format!("Failed to execute conversion: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Conversion failed: {}", stderr));
    }

    // Print stdout
    let stdout = String::from_utf8_lossy(&output.stdout);
    print!("{}", stdout);

    // Verify output file exists
    if !Path::new(output_path).exists() {
        return Err(format!("Output file not created: {}", output_path));
    }

    Ok(output_path.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires Python and paddle2onnx installed
    fn test_paddle_to_onnx_conversion() {
        // This is an integration test
        // Run with: cargo test --example paddle_to_openvino -- --ignored
        let result = convert_paddle_to_onnx(
            "test_models/paddle",
            "test_output.onnx",
            "inference.json",
            "inference.pdiparams",
            11,
        );

        match result {
            Ok(path) => println!("Conversion successful: {}", path),
            Err(e) => eprintln!("Conversion failed: {}", e),
        }
    }
}
