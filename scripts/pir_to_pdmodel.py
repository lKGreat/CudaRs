#!/usr/bin/env python3
"""
Convert PIR format (inference.json) to pdmodel format for paddle2onnx compatibility
"""

import os
import sys
import argparse
from pathlib import Path


def convert_pir_to_pdmodel(model_dir, output_dir=None):
    """
    Convert PIR format model (inference.json) to pdmodel format
    
    Args:
        model_dir: Directory containing inference.json and inference.pdiparams
        output_dir: Output directory (default: same as model_dir)
    """
    try:
        import paddle
        from paddle import inference
    except ImportError:
        print("Error: PaddlePaddle is not installed")
        print("Please install: pip install paddlepaddle")
        return False
    
    model_dir = Path(model_dir)
    json_path = model_dir / "inference.json"
    params_path = model_dir / "inference.pdiparams"
    
    if not json_path.exists():
        print(f"Error: {json_path} not found")
        return False
    if not params_path.exists():
        print(f"Error: {params_path} not found")
        return False
    
    output_dir = Path(output_dir) if output_dir else model_dir
    
    print(f"Loading PIR model from {model_dir}...")
    
    try:
        # Create config
        config = inference.Config(
            str(json_path),
            str(params_path)
        )
        
        # Try to save as pdmodel format
        # Note: This might not work directly, as PaddlePaddle may not support
        # converting PIR back to pdmodel easily
        
        # Alternative: Try using paddle.jit.save or paddle.static.save_inference_model
        # But these require loading the model first
        
        print("Warning: Direct PIR to pdmodel conversion may not be supported")
        print("Trying alternative approach...")
        
        # Try loading with inference API and see if we can get the model
        predictor = inference.create_predictor(config)
        
        # Unfortunately, there's no direct way to convert PIR to pdmodel
        # The best approach is to use the model with paddle2onnx's newer versions
        # or use paddle's onnx export directly
        
        print("PIR format conversion to pdmodel is not straightforward.")
        print("Consider using paddle.onnx.export() directly instead.")
        
        return False
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False


def try_paddle_onnx_export(model_dir, output_path):
    """
    Try using paddle.onnx.export directly
    """
    try:
        import paddle
        from paddle import inference
        
        model_dir = Path(model_dir)
        json_path = model_dir / "inference.json"
        params_path = model_dir / "inference.pdiparams"
        
        print(f"Attempting direct ONNX export using paddle.onnx.export...")
        
        # Load model
        config = inference.Config(str(json_path), str(params_path))
        predictor = inference.create_predictor(config)
        
        # Unfortunately paddle.onnx.export requires a Layer object, not a predictor
        # This approach won't work directly
        
        print("Direct export not possible with inference API")
        return False
        
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert PIR format to pdmodel (may not be fully supported)"
    )
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing inference.json and inference.pdiparams')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory for pdmodel files')
    
    args = parser.parse_args()
    
    if not convert_pir_to_pdmodel(args.model_dir, args.output_dir):
        print("\nNote: PIR format (inference.json) conversion is limited.")
        print("Consider:")
        print("1. Using a newer version of paddle2onnx that supports PIR")
        print("2. Re-exporting the model in pdmodel format from source")
        print("3. Using PaddlePaddle's native ONNX export if available")
        sys.exit(1)
