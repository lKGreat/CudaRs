#!/usr/bin/env python3
"""
Direct conversion from PIR format (inference.json) to ONNX using paddle.onnx.export
"""

import os
import sys
import argparse
from pathlib import Path


def convert_pir_to_onnx_direct(model_dir, output_path, opset_version=11):
    """
    Convert PIR format model directly to ONNX using paddle.onnx.export
    
    Args:
        model_dir: Directory containing inference.json and inference.pdiparams
        output_path: Path to save ONNX model
        opset_version: ONNX opset version
    """
    try:
        import paddle
        from paddle import inference
        import numpy as np
    except ImportError as e:
        print(f"Error: Missing required package - {e}")
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
    
    print(f"Loading PIR model from {model_dir}...")
    
    try:
        # Create config and predictor
        config = inference.Config(str(json_path), str(params_path))
        # Note: enable_memory_optim() may not work with PIR format
        # config.enable_memory_optim()
        predictor = inference.create_predictor(config)
        
        # Get input/output info
        input_names = predictor.get_input_names()
        output_names = predictor.get_output_names()
        
        print(f"Model inputs: {input_names}")
        print(f"Model outputs: {output_names}")
        
        # Get input shapes
        input_specs = []
        for name in input_names:
            input_tensor = predictor.get_input_handle(name)
            shape = input_tensor.shape()
            dtype = input_tensor.type()
            print(f"  Input {name}: shape={shape}, dtype={dtype}")
            
            # Create InputSpec for paddle.onnx.export
            # Note: paddle.onnx.export requires a Layer, not a predictor
            # This approach won't work directly with inference API
        
        print("\nWarning: paddle.onnx.export requires a Layer object, not a predictor.")
        print("For inference models, we need to use paddle2onnx or convert the model format.")
        print("\nTrying alternative: Check if model can be loaded as Layer...")
        
        # Try loading as static model
        try:
            # This might work if we can load the model differently
            # But inference.json is PIR format, which is different from static graph
            print("PIR format models cannot be directly loaded as Layer objects.")
            print("Consider using the original training code to export in pdmodel format,")
            print("or wait for paddle2onnx to support PIR format.")
        except Exception as e2:
            print(f"Alternative approach also failed: {e2}")
        
        return False
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False


def try_workaround_conversion(model_dir, output_path):
    """
    Try workaround: Use paddle's program to convert
    """
    try:
        import paddle
        from paddle import static
        
        print("\nAttempting workaround conversion...")
        
        # This approach also won't work because inference.json is PIR format
        # which is different from the old ProgramDesc format
        
        print("Workaround not available for PIR format.")
        return False
        
    except Exception as e:
        print(f"Workaround failed: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert PIR format to ONNX (experimental)"
    )
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing inference.json and inference.pdiparams')
    parser.add_argument('--output', type=str, required=True,
                       help='Output ONNX file path')
    parser.add_argument('--opset_version', type=int, default=11,
                       help='ONNX opset version')
    
    args = parser.parse_args()
    
    if not convert_pir_to_onnx_direct(args.model_dir, args.output, args.opset_version):
        print("\n" + "="*60)
        print("CONVERSION NOT POSSIBLE WITH CURRENT TOOLS")
        print("="*60)
        print("\nPIR format (inference.json) is not yet fully supported by:")
        print("1. paddle2onnx (command line tool)")
        print("2. paddle.onnx.export (requires Layer object, not inference model)")
        print("\nRecommended solutions:")
        print("1. Re-export the model from training code in pdmodel format")
        print("2. Use the model with PaddlePaddle inference API directly")
        print("3. Wait for paddle2onnx to add full PIR support")
        print("4. Check if there's a pdmodel version available")
        sys.exit(1)
