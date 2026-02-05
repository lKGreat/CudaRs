#!/usr/bin/env python
import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert ONNX to OpenVINO IR (xml/bin) using ovc.")
    parser.add_argument("--input", required=True, help="Path to ONNX model")
    parser.add_argument("--output_dir", required=True, help="Output directory for IR files")
    parser.add_argument("--model_name", required=False, help="Base name for IR files (default: ONNX filename)")
    parser.add_argument("--compress_to_fp16", action="store_true", help="Compress weights to FP16")
    args = parser.parse_args()

    onnx_path = os.path.abspath(args.input)
    if not os.path.isfile(onnx_path):
        print(f"ERROR: ONNX model not found: {onnx_path}", file=sys.stderr)
        return 1

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    model_name = args.model_name or os.path.splitext(os.path.basename(onnx_path))[0]
    xml_path = os.path.join(output_dir, f"{model_name}.xml")

    try:
        from openvino.tools.ovc import convert_model
        import openvino as ov
    except Exception as exc:
        print("ERROR: OpenVINO Python package not available. Install with: pip install openvino", file=sys.stderr)
        print(f"DETAIL: {exc}", file=sys.stderr)
        return 2

    try:
        model = convert_model(onnx_path)
        if hasattr(ov, "save_model"):
            ov.save_model(model, xml_path, compress_to_fp16=args.compress_to_fp16)
        else:
            from openvino.runtime import serialize
            serialize(model, xml_path)
    except Exception as exc:
        print(f"ERROR: Conversion failed: {exc}", file=sys.stderr)
        return 3

    bin_path = os.path.splitext(xml_path)[0] + ".bin"
    if not os.path.isfile(xml_path) or not os.path.isfile(bin_path):
        print("ERROR: IR conversion finished but output files are missing.", file=sys.stderr)
        print(f"  xml: {xml_path}", file=sys.stderr)
        print(f"  bin: {bin_path}", file=sys.stderr)
        return 4

    print(f"OK: {xml_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
