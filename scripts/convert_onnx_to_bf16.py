#!/usr/bin/env python3
"""
Convert ONNX FP32 model to BF16 format with accuracy reporting.

This script converts all float32 tensors in an ONNX model to bfloat16 format
and reports the numerical accuracy of the conversion.

Usage:
    python convert_onnx_to_bf16.py input_model.onnx output_model.onnx

Output:
    - Converted BF16 model
    - Accuracy report (max error, mean error per layer)
"""

import sys
import argparse
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from typing import Dict, List, Tuple, Optional
from pathlib import Path


def float32_to_bfloat16(arr: np.ndarray) -> np.ndarray:
    """
    Convert float32 numpy array to bfloat16 (stored as uint16).
    
    BF16 has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
    We truncate the lower 16 bits of FP32 to get BF16.
    """
    arr_uint32 = arr.view(np.uint32)
    arr_bf16 = (arr_uint32 >> 16).astype(np.uint16)
    return arr_bf16


def bfloat16_to_float32(arr_uint16: np.ndarray) -> np.ndarray:
    """
    Convert bfloat16 (stored as uint16) back to float32.
    """
    arr_uint32 = arr_uint16.astype(np.uint32) << 16
    return arr_uint32.view(np.float32)


def compute_error_metrics(fp32_arr: np.ndarray, bf16_arr: np.ndarray) -> Dict[str, float]:
    """
    Compute error metrics between FP32 and BF16 arrays.
    
    Returns:
        Dictionary with max_error, mean_error, rmse, max_rel_error
    """
    # Convert BF16 back to FP32 for comparison
    if bf16_arr.dtype == np.uint16:
        bf16_as_fp32 = bfloat16_to_float32(bf16_arr)
    else:
        bf16_as_fp32 = bf16_arr.astype(np.float32)
    
    fp32_flat = fp32_arr.flatten()
    bf16_flat = bf16_as_fp32.flatten()
    
    abs_diff = np.abs(fp32_flat - bf16_flat)
    rel_diff = abs_diff / (np.abs(fp32_flat) + 1e-7)
    
    return {
        'count': len(fp32_flat),
        'max_error': float(np.max(abs_diff)),
        'mean_error': float(np.mean(abs_diff)),
        'rmse': float(np.sqrt(np.mean(abs_diff ** 2))),
        'max_rel_error': float(np.max(rel_diff)),
        'mean_rel_error': float(np.mean(rel_diff)),
    }


def print_error_table(errors: List[Tuple[str, Dict[str, float]]], overall: Dict[str, float]):
    """Print formatted error table."""
    print("\n" + "="*100)
    print("CONVERSION ACCURACY REPORT")
    print("="*100)
    print(f"{'Tensor Name':<40} {'Count':>10} {'Max Error':>12} {'Mean Error':>12} {'RMSE':>12} {'Max Rel %':>10}")
    print("-"*100)
    
    for name, err in errors:
        print(f"{name:<40} {err['count']:>10,} {err['max_error']:>12.6f} {err['mean_error']:>12.6f} {err['rmse']:>12.6f} {err['max_rel_error']*100:>9.2f}%")
    
    print("-"*100)
    print(f"{'OVERALL':<40} {overall['count']:>10,} {overall['max_error']:>12.6f} {overall['mean_error']:>12.6f} {overall['rmse']:>12.6f} {overall['max_rel_error']*100:>9.2f}%")
    print("="*100)
    
    # Interpretation guide
    print("\nInterpretation:")
    print("  - Max Error < 0.01    : Excellent (typical for BF16)")
    print("  - Max Error < 0.1     : Good (acceptable for most inference)")
    print("  - Max Error < 1.0     : Fair (may affect some layers)")
    print("  - Max Error >= 1.0    : Poor (significant accuracy loss)")
    print("\nNote: BF16 has ~0.1% relative precision vs FP32's ~0.00001%")


def fix_batch_dimension_to_one(model: onnx.ModelProto) -> int:
    """
    Fix dynamic batch dimensions (?, -1) to 1 for all inputs.
    
    Returns:
        Number of input shapes modified
    """
    modified_count = 0
    
    for input_tensor in model.graph.input:
        tensor_type = input_tensor.type.tensor_type
        
        # Check if shape has a dynamic batch dimension
        if tensor_type.HasField('shape') and len(tensor_type.shape.dim) > 0:
            first_dim = tensor_type.shape.dim[0]
            
            # Check if batch is dynamic (dim_value not set or is -1/0)
            is_dynamic = False
            if first_dim.HasField('dim_param'):
                # Has a symbolic dimension like "batch" or "N"
                is_dynamic = True
            elif first_dim.HasField('dim_value'):
                # Has a numeric value - check if it's invalid (0 or -1)
                if first_dim.dim_value <= 0 or first_dim.dim_value == 1:
                    # Could be dynamic or already 1
                    pass
            else:
                # Neither dim_param nor dim_value set - treat as dynamic
                is_dynamic = True
            
            # If first dim has dim_param, it's dynamic - fix it
            if first_dim.HasField('dim_param'):
                print(f"  Fixing batch dimension for '{input_tensor.name}': {first_dim.dim_param} -> 1")
                first_dim.ClearField('dim_param')
                first_dim.dim_value = 1
                modified_count += 1
            elif not first_dim.HasField('dim_value'):
                # Unspecified dimension - set to 1
                print(f"  Setting batch dimension for '{input_tensor.name}': ? -> 1")
                first_dim.dim_value = 1
                modified_count += 1
    
    return modified_count


def convert_and_compare(model) -> Tuple[onnx.ModelProto, List[Tuple[str, Dict]], Dict]:
    """
    Convert model to BF16 and compute error metrics.
    
    Returns:
        (converted_model, per_tensor_errors, overall_error)
    """
    all_errors = []
    total_abs_error = 0.0
    total_count = 0
    max_error_overall = 0.0
    max_rel_error_overall = 0.0
    
    # Fix batch dimension from dynamic (?, -1) to 1
    print("Checking and fixing batch dimensions...")
    batch_fixed = fix_batch_dimension_to_one(model)
    if batch_fixed > 0:
        print(f"  Fixed {batch_fixed} input(s) to have batch=1")
    else:
        print("  No dynamic batch dimensions found or already fixed")
    
    # Process initializers (weights and biases)
    for init in model.graph.initializer:
        if init.data_type != TensorProto.FLOAT:
            continue
        
        # Get FP32 data
        if init.raw_data:
            fp32_data = np.frombuffer(init.raw_data, dtype=np.float32).copy()
        elif init.float_data:
            fp32_data = np.array(init.float_data, dtype=np.float32)
        else:
            continue
        
        # Convert to BF16
        bf16_data = float32_to_bfloat16(fp32_data)
        
        # Compute error
        error = compute_error_metrics(fp32_data, bf16_data)
        all_errors.append((init.name, error))
        
        # Update overall stats
        total_abs_error += error['mean_error'] * error['count']
        total_count += error['count']
        max_error_overall = max(max_error_overall, error['max_error'])
        max_rel_error_overall = max(max_rel_error_overall, error['max_rel_error'])
        
        # Replace data in initializer
        init.raw_data = bf16_data.tobytes()
        init.float_data[:] = []
        init.data_type = TensorProto.BFLOAT16
    
    # Run shape inference to populate value_info (intermediate tensor types)
    # This is needed for Netron to display dtypes for all tensors
    print("Running shape inference to populate intermediate tensor types...")
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"  Warning: Shape inference failed: {e}")
    
    # Update input/output types
    for value_info in list(model.graph.input) + list(model.graph.output):
        if value_info.type.tensor_type.elem_type == TensorProto.FLOAT:
            value_info.type.tensor_type.elem_type = TensorProto.BFLOAT16
    
    # Update intermediate value_info types
    value_info_count = 0
    for value_info in model.graph.value_info:
        if value_info.type.tensor_type.elem_type == TensorProto.FLOAT:
            value_info.type.tensor_type.elem_type = TensorProto.BFLOAT16
            value_info_count += 1
    print(f"  Updated {value_info_count} intermediate tensor types to BF16")
    
    # Compute overall error
    overall_error = {
        'count': total_count,
        'max_error': max_error_overall,
        'mean_error': total_abs_error / total_count if total_count > 0 else 0,
        'rmse': np.sqrt(total_abs_error / total_count) if total_count > 0 else 0,
        'max_rel_error': max_rel_error_overall,
        'mean_rel_error': max_rel_error_overall,  # Approximation
    }
    
    return model, all_errors, overall_error


def create_random_input(model, batch_size=1):
    """Create random input tensor matching model's input shape."""
    input_shape = [d.dim_value if d.dim_value > 0 else batch_size 
                   for d in model.graph.input[0].type.tensor_type.shape.dim]
    return np.random.randn(*input_shape).astype(np.float32)


def run_inference_ort(model_path, input_data):
    """Run inference using ONNX Runtime."""
    try:
        session = ort.InferenceSession(str(model_path))
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_data})
        return outputs
    except Exception as e:
        print(f"  Warning: ORT inference failed: {e}")
        return None


def compare_inference_outputs(fp32_outputs, bf16_outputs):
    """Compare inference outputs and return metrics."""
    if fp32_outputs is None or bf16_outputs is None:
        return None
    
    metrics = []
    for i, (fp32, bf16) in enumerate(zip(fp32_outputs, bf16_outputs)):
        # Convert BF16 output to FP32 for comparison
        if bf16.dtype == np.uint16:
            bf16_as_fp32 = (bf16.astype(np.uint32) << 16).view(np.float32)
        else:
            bf16_as_fp32 = bf16.astype(np.float32)
        
        fp32_flat = fp32.flatten()
        bf16_flat = bf16_as_fp32.flatten()
        
        abs_diff = np.abs(fp32_flat - bf16_flat)
        rel_diff = abs_diff / (np.abs(fp32_flat) + 1e-7)
        
        metrics.append({
            'output_idx': i,
            'shape': fp32.shape,
            'max_abs_error': float(np.max(abs_diff)),
            'mean_abs_error': float(np.mean(abs_diff)),
            'rmse': float(np.sqrt(np.mean(abs_diff ** 2))),
            'max_rel_error': float(np.max(rel_diff)),
            'mean_rel_error': float(np.mean(rel_diff)),
        })
    
    return metrics


def compare_inference_accuracy(fp32_path, bf16_path, num_samples=5):
    """Compare inference accuracy between FP32 and BF16 models."""
    print("\n" + "="*100)
    print("INFERENCE ACCURACY COMPARISON")
    print("="*100)
    print(f"Running {num_samples} random input comparisons using ONNX Runtime...")
    
    fp32_model = onnx.load(str(fp32_path))
    all_metrics = []
    
    for i in range(num_samples):
        # Create random input
        input_data = create_random_input(fp32_model)
        
        # Run FP32 inference
        fp32_outputs = run_inference_ort(fp32_path, input_data)
        
        # Run BF16 inference (input needs BF16 dtype)
        input_bf16 = input_data.astype(np.float32).view(np.uint32) >> 16
        input_bf16 = input_bf16.astype(np.uint16)
        bf16_outputs = run_inference_ort(bf16_path, input_bf16)
        
        # Compare
        metrics = compare_inference_outputs(fp32_outputs, bf16_outputs)
        if metrics:
            all_metrics.append(metrics)
    
    if not all_metrics:
        print("  Failed to run inference comparison")
        return
    
    # Print per-sample results
    print(f"\n{'Sample':<8} {'Output':<8} {'Shape':<20} {'Max Error':<12} {'Mean Error':<12} {'RMSE':<12}")
    print("-" * 100)
    
    for sample_idx, metrics in enumerate(all_metrics):
        for m in metrics:
            shape_str = str(m['shape'])
            print(f"{sample_idx+1:<8} {m['output_idx']:<8} {shape_str:<20} "
                  f"{m['max_abs_error']:<12.6f} {m['mean_abs_error']:<12.6f} {m['rmse']:<12.6f}")
    
    # Aggregate statistics
    print(f"\n{'='*100}")
    print("AGGREGATE INFERENCE STATS:")
    
    max_errors = [m[0]['max_abs_error'] for m in all_metrics]
    mean_errors = [m[0]['mean_abs_error'] for m in all_metrics]
    rmses = [m[0]['rmse'] for m in all_metrics]
    max_rel_errors = [m[0]['max_rel_error'] for m in all_metrics]
    
    print(f"  Max error across all samples:     {max(max_errors):.6f}")
    print(f"  Mean of max errors:               {np.mean(max_errors):.6f}")
    print(f"  Mean of mean errors:              {np.mean(mean_errors):.6f}")
    print(f"  Mean RMSE:                        {np.mean(rmses):.6f}")
    print(f"  Max relative error:               {max(max_rel_errors)*100:.2f}%")
    print(f"{'='*100}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ONNX FP32 model to BF16 with accuracy report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic conversion
    python convert_onnx_to_bf16.py model.onnx model_bf16.onnx
    
    # Conversion with inference comparison
    python convert_onnx_to_bf16.py model.onnx model_bf16.onnx --compare-inference
    
    # More inference samples for comparison
    python convert_onnx_to_bf16.py model.onnx model_bf16.onnx --compare-inference --num-samples 10
        """
    )
    parser.add_argument("input", help="Input ONNX model path (FP32)")
    parser.add_argument("output", help="Output ONNX model path (BF16)")
    parser.add_argument("--compare-inference", action="store_true",
                       help="Run inference comparison between FP32 and BF16 models")
    parser.add_argument("--num-samples", type=int, default=5,
                       help="Number of random inputs for inference comparison (default: 5)")
    
    args = parser.parse_args()
    
    print(f"Loading FP32 model: {args.input}")
    model = onnx.load(str(args.input))
    
    print("Converting to BF16 and computing accuracy metrics...")
    converted_model, errors, overall = convert_and_compare(model)
    
    # Print accuracy report
    print_error_table(errors, overall)
    
    # Save converted model
    print(f"\nSaving BF16 model to: {args.output}")
    onnx.save(converted_model, str(args.output))
    
    # Print summary
    from pathlib import Path
    input_size = Path(args.input).stat().st_size / (1024 * 1024)
    output_size = Path(args.output).stat().st_size / (1024 * 1024)
    
    print(f"\nFile sizes:")
    print(f"  FP32: {input_size:.1f} MB")
    print(f"  BF16: {output_size:.1f} MB")
    print(f"  Reduction: {(1 - output_size/input_size)*100:.1f}%")
    
    # Quality assessment
    print(f"\nQuality Assessment:")
    if overall['max_error'] < 0.01:
        print("  EXCELLENT - BF16 conversion has minimal precision loss")
    elif overall['max_error'] < 0.1:
        print("  GOOD - BF16 conversion is acceptable for inference")
    elif overall['max_error'] < 1.0:
        print("  FAIR - Some layers may have noticeable accuracy impact")
    else:
        print("  POOR - Significant accuracy loss, review conversion")
    
    # Run inference comparison if requested
    if args.compare_inference:
        compare_inference_accuracy(args.input, args.output, args.num_samples)
    
    print(f"\nConversion complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
