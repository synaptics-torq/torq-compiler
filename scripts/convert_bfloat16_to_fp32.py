#!/usr/bin/env python3
"""
Convert ONNX BF16 model back to FP32 format.

This is useful for creating a working FP32 test model from a known-good BF16 model.

Usage:
    python convert_bfloat16_to_fp32.py input_bf16.onnx output_fp32.onnx
"""

import argparse
import numpy as np
import onnx
from onnx import TensorProto
from pathlib import Path


def bfloat16_to_float32(arr_uint16: np.ndarray) -> np.ndarray:
    """Convert bfloat16 (stored as uint16) back to float32."""
    arr_uint32 = arr_uint16.astype(np.uint32) << 16
    return arr_uint32.view(np.float32)


def convert_bf16_to_fp32(model: onnx.ModelProto) -> onnx.ModelProto:
    """Convert BF16 ONNX model to FP32 format."""
    import copy
    model = copy.deepcopy(model)  # Don't modify original

    # Convert initializers (weights)
    converted_count = 0
    for init in model.graph.initializer:
        if init.data_type != TensorProto.BFLOAT16:
            continue

        # Get BF16 data
        if init.raw_data:
            bf16_data = np.frombuffer(init.raw_data, dtype=np.uint16).copy()
        else:
            continue

        # Convert to FP32
        fp32_data = bfloat16_to_float32(bf16_data)

        # Replace data
        init.raw_data = fp32_data.tobytes()
        init.data_type = TensorProto.FLOAT
        converted_count += 1

    print(f"[FP32] Converted {converted_count} tensors from BF16 to FP32")

    # Run shape inference
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass

    # Update input/output types
    for value_info in list(model.graph.input) + list(model.graph.output):
        if value_info.type.tensor_type.elem_type == TensorProto.BFLOAT16:
            value_info.type.tensor_type.elem_type = TensorProto.FLOAT

    # Update intermediate value_info types
    for value_info in model.graph.value_info:
        if value_info.type.tensor_type.elem_type == TensorProto.BFLOAT16:
            value_info.type.tensor_type.elem_type = TensorProto.FLOAT

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Convert ONNX BF16 model back to FP32 format"
    )
    parser.add_argument("input", help="Input ONNX model path (BF16)")
    parser.add_argument("output", help="Output ONNX model path (FP32)")

    args = parser.parse_args()

    print(f"Loading BF16 model: {args.input}")
    model = onnx.load(str(args.input))

    # Check if actually BF16
    is_bf16 = any(
        init.data_type == TensorProto.BFLOAT16
        for init in model.graph.initializer
    )
    if not is_bf16:
        print("Warning: Model does not appear to be BF16, may already be FP32")

    print("Converting to FP32...")
    fp32_model = convert_bf16_to_fp32(model)

    # Save
    print(f"Saving FP32 model to: {args.output}")
    onnx.save(fp32_model, str(args.output))

    # Report sizes
    from pathlib import Path
    input_size = Path(args.input).stat().st_size / (1024 * 1024)
    output_size = Path(args.output).stat().st_size / (1024 * 1024)

    print(f"\nFile sizes:")
    print(f"  BF16: {input_size:.1f} MB")
    print(f"  FP32: {output_size:.1f} MB")
    print(f"  Increase: {(output_size/input_size - 1)*100:.1f}%")

    print("\nConversion complete!")
    return 0


if __name__ == "__main__":
    exit(main())
