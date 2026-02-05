#!/usr/bin/env python3
"""
PaddlePaddle to ONNX Converter
Converts PaddlePaddle models (inference.json format) to ONNX format for use with OpenVINO
"""

import os
import sys
import argparse
import json
from pathlib import Path


def check_dependencies():
    """Check if required packages are installed"""
    try:
        import paddle2onnx
        import onnx
        return True
    except ImportError as e:
        print(f"Error: Missing required package - {e}")
        print("\nPlease install required packages:")
        print("  pip install paddle2onnx onnx onnxruntime")
        return False


def validate_model_files(model_dir, model_filename, params_filename):
    """Validate that model files exist"""
    model_path = Path(model_dir) / model_filename
    params_path = Path(model_dir) / params_filename
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not params_path.exists():
        raise FileNotFoundError(f"Parameters file not found: {params_path}")
    
    return str(model_path), str(params_path)


def convert_paddle_to_onnx(model_dir, output_path, model_filename="inference.json", 
                           params_filename="inference.pdiparams", opset_version=11,
                           enable_onnx_checker=True, enable_auto_update_opset=True):
    """
    Convert PaddlePaddle model to ONNX format
    
    Args:
        model_dir: Directory containing the PaddlePaddle model files
        output_path: Path to save the ONNX model
        model_filename: Name of the model structure file (default: inference.json)
        params_filename: Name of the parameters file (default: inference.pdiparams)
        opset_version: ONNX opset version (default: 11)
        enable_onnx_checker: Enable ONNX model validation (default: True)
        enable_auto_update_opset: Auto update opset version if needed (default: True)
    
    Returns:
        Path to the generated ONNX model
    """
    import paddle2onnx
    
    # Validate input files
    model_path, params_path = validate_model_files(model_dir, model_filename, params_filename)
    
    print(f"Converting PaddlePaddle model to ONNX...")
    print(f"  Model: {model_path}")
    print(f"  Params: {params_path}")
    print(f"  Output: {output_path}")
    print(f"  Opset version: {opset_version}")
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to ONNX using paddle2onnx.convert.export
    try:
        paddle2onnx.convert.export(
            model_filename=model_path,
            params_filename=params_path,
            save_file=output_path,
            opset_version=opset_version,
            auto_upgrade_opset=enable_auto_update_opset,
            enable_onnx_checker=enable_onnx_checker,
            verbose=True
        )
        
        print(f"OK Conversion successful!")
        print(f"  ONNX model saved to: {output_path}")
        
        # Validate ONNX model (non-blocking)
        if enable_onnx_checker:
            try:
                validate_onnx_model(output_path)
            except Exception as e:
                print(f"WARNING: ONNX validation failed: {e}")
                print(f"  Model file was still created and may be usable")
        
        return output_path
        
    except Exception as e:
        print(f"FAIL Conversion failed: {e}")
        raise


def validate_onnx_model(onnx_path):
    """Validate the generated ONNX model"""
    import onnx
    
    print(f"\nValidating ONNX model...")
    try:
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        
        # Print model info
        print(f"OK ONNX model is valid")
        print(f"  IR version: {model.ir_version}")
        print(f"  Producer: {model.producer_name}")
        print(f"  Opset version: {model.opset_import[0].version}")
        
        # Print input/output info
        print(f"\n  Inputs ({len(model.graph.input)}):")
        for inp in model.graph.input:
            dims = [str(d.dim_value) if d.dim_value > 0 else '?' for d in inp.type.tensor_type.shape.dim]
            print(f"    - {inp.name}: {dims}")
        
        print(f"\n  Outputs ({len(model.graph.output)}):")
        for out in model.graph.output:
            dims = [str(d.dim_value) if d.dim_value > 0 else '?' for d in out.type.tensor_type.shape.dim]
            print(f"    - {out.name}: {dims}")
        
    except Exception as e:
        print(f"WARNING: ONNX validation failed: {e}")
        # Try to still print basic info even if validation fails
        try:
            model = onnx.load(onnx_path)
            print(f"  Model file exists and can be loaded")
            print(f"  IR version: {getattr(model, 'ir_version', 'N/A')}")
            if model.opset_import:
                print(f"  Opset version: {model.opset_import[0].version}")
        except:
            pass
        raise


def batch_convert(input_dir, output_dir, model_filename="inference.json", 
                  params_filename="inference.pdiparams", opset_version=11):
    """
    Batch convert multiple PaddlePaddle models to ONNX
    
    Args:
        input_dir: Root directory containing PaddlePaddle model directories
        output_dir: Root directory to save ONNX models
        model_filename: Name of the model structure file
        params_filename: Name of the parameters file
        opset_version: ONNX opset version
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Find all model directories
    model_dirs = []
    for item in input_path.rglob(model_filename):
        model_dirs.append(item.parent)
    
    if not model_dirs:
        print(f"No PaddlePaddle models found in {input_dir}")
        return
    
    print(f"Found {len(model_dirs)} model(s) to convert:")
    for model_dir in model_dirs:
        print(f"  - {model_dir.relative_to(input_path)}")
    
    print("\n" + "="*60)
    
    # Convert each model
    success_count = 0
    for i, model_dir in enumerate(model_dirs, 1):
        print(f"\n[{i}/{len(model_dirs)}] Converting: {model_dir.name}")
        print("-" * 60)
        
        try:
            rel_path = model_dir.relative_to(input_path)
            output_model_dir = output_path / rel_path
            output_model_path = output_model_dir / "model.onnx"
            
            convert_paddle_to_onnx(
                str(model_dir),
                str(output_model_path),
                model_filename=model_filename,
                params_filename=params_filename,
                opset_version=opset_version
            )
            success_count += 1
            
        except Exception as e:
            print(f"Failed to convert {model_dir.name}: {e}")
            continue
    
    print("\n" + "="*60)
    print(f"\nConversion complete: {success_count}/{len(model_dirs)} successful")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PaddlePaddle models to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a single model
  python paddle2onnx_converter.py \\
    --model_dir ./PP-OCRv5_mobile_det_infer \\
    --output model.onnx

  # Convert with custom filenames
  python paddle2onnx_converter.py \\
    --model_dir ./model \\
    --model_filename inference.json \\
    --params_filename inference.pdiparams \\
    --output model.onnx \\
    --opset_version 13

  # Batch convert multiple models
  python paddle2onnx_converter.py \\
    --batch \\
    --input_dir ./models \\
    --output_dir ./onnx_models
        """
    )
    
    parser.add_argument('--model_dir', type=str, 
                       help='Directory containing PaddlePaddle model files')
    parser.add_argument('--output', type=str,
                       help='Path to save the ONNX model')
    parser.add_argument('--model_filename', type=str, default='inference.json',
                       help='Name of the model structure file (default: inference.json)')
    parser.add_argument('--params_filename', type=str, default='inference.pdiparams',
                       help='Name of the parameters file (default: inference.pdiparams)')
    parser.add_argument('--opset_version', type=int, default=11,
                       help='ONNX opset version (default: 11)')
    parser.add_argument('--batch', action='store_true',
                       help='Batch convert multiple models')
    parser.add_argument('--input_dir', type=str,
                       help='Root directory for batch conversion')
    parser.add_argument('--output_dir', type=str,
                       help='Output root directory for batch conversion')
    parser.add_argument('--no-validation', action='store_true',
                       help='Skip ONNX model validation')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    try:
        if args.batch:
            # Batch conversion
            if not args.input_dir or not args.output_dir:
                parser.error("--input_dir and --output_dir are required for batch conversion")
            
            batch_convert(
                args.input_dir,
                args.output_dir,
                model_filename=args.model_filename,
                params_filename=args.params_filename,
                opset_version=args.opset_version
            )
        else:
            # Single model conversion
            if not args.model_dir or not args.output:
                parser.error("--model_dir and --output are required for single model conversion")
            
            convert_paddle_to_onnx(
                args.model_dir,
                args.output,
                model_filename=args.model_filename,
                params_filename=args.params_filename,
                opset_version=args.opset_version,
                enable_onnx_checker=not args.no_validation
            )
        
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
