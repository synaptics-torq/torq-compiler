"""Verify ONNX to MLIR import preserves ordering.

This script validates the assumption that torch-mlir imports ONNX nodes:
1. In topological order (same as ONNX node order)
2. Without fusion (one ONNX node = one torch.operator)
3. Without reordering

Usage:
    python scripts/verify_onnx_import_order.py --model-path=./model.onnx
    python scripts/verify_onnx_import_order.py --model-dir=./tests/testdata/onnx_models

If this verification fails, the executor discovery mapping logic needs to be updated.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import onnx


def extract_onnx_op_sequence(model: onnx.ModelProto) -> list:
    """Extract ordered list of (op_type, name, output) from ONNX model."""
    return [
        (node.op_type, node.name, node.output[0] if node.output else None)
        for node in model.graph.node
        if node.op_type != "Constant"  # Constants are handled separately
    ]


def extract_mlir_op_sequence(mlir_file: Path) -> list:
    """Extract ordered list of (op_type, location_name) from MLIR."""
    content = mlir_file.read_text()
    ops = []

    for line_num, line_content in enumerate(content.split("\n"), start=1):
        match = re.search(r'torch\.operator\s+"onnx\.([A-Za-z]+)"', line_content, re.IGNORECASE)
        if not match:
            continue

        op_type = match.group(1)
        if op_type == "Constant":
            continue

        # Extract name from location
        node_name = None
        loc_match = re.search(r'loc\(([^)]+)\)', line_content)
        if loc_match:
            loc_str = loc_match.group(1)
            name_match = re.search(r'"([^"]+)"', loc_str)
            if name_match:
                candidate = name_match.group(1)
                if "/" not in candidate and not candidate.endswith(".mlir"):
                    node_name = candidate

        ops.append((op_type, node_name, line_num))

    return ops


def verify_import_order(onnx_model_path: Path, mlir_file: Path) -> dict:
    """Verify ONNX to MLIR import preserves ordering.

    Returns dict with:
        - 'onnx_ops': list of ONNX ops
        - 'mlir_ops': list of MLIR ops
        - 'matches': bool if sequences match
        - 'mismatches': list of discrepancies
        - 'counts_match': bool if op counts match
    """
    # Load ONNX
    model = onnx.load(str(onnx_model_path))
    onnx_ops = extract_onnx_op_sequence(model)

    # Generate MLIR if needed
    if not mlir_file.exists():
        mlir_file.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, "-m", "iree.compiler.tools.import_onnx",
             str(onnx_model_path), "-o", str(mlir_file), "--data-prop"],
            check=True, capture_output=True,
        )

    mlir_ops = extract_mlir_op_sequence(mlir_file)

    # Compare sequences
    mismatches = []
    min_len = min(len(onnx_ops), len(mlir_ops))

    for i in range(min_len):
        onnx_op, onnx_name, onnx_output = onnx_ops[i]
        mlir_op, mlir_name, mlir_line = mlir_ops[i]

        if onnx_op != mlir_op:
            mismatches.append({
                "position": i,
                "type": "op_type_mismatch",
                "onnx": f"{onnx_op} (name='{onnx_name}')",
                "mlir": f"{mlir_op} (name='{mlir_name}') at line {mlir_line}",
            })
        elif onnx_name and mlir_name and onnx_name != mlir_name:
            # Name mismatch but type matches - warning only
            mismatches.append({
                "position": i,
                "type": "name_mismatch",
                "onnx": f"{onnx_op} (name='{onnx_name}')",
                "mlir": f"{mlir_op} (name='{mlir_name}') at line {mlir_line}",
            })

    # Check counts
    if len(onnx_ops) != len(mlir_ops):
        mismatches.append({
            "position": min_len,
            "type": "count_mismatch",
            "onnx": f"{len(onnx_ops)} ops",
            "mlir": f"{len(mlir_ops)} ops",
        })

    return {
        "onnx_ops": onnx_ops,
        "mlir_ops": mlir_ops,
        "matches": len(mismatches) == 0,
        "mismatches": mismatches,
        "counts_match": len(onnx_ops) == len(mlir_ops),
    }


def verify_single_model(model_path: Path, mlir_file: Path = None) -> bool:
    """Verify ONNX to MLIR import ordering for a single model.

    Returns True if verification passes, False otherwise.
    """
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}", file=sys.stderr)
        return False

    if mlir_file is None:
        mlir_file = Path("/tmp") / f"verify_{model_path.stem}.mlir"

    print(f"\nVerifying ONNX to MLIR import for: {model_path.name}")
    print("-" * 60)

    try:
        result = verify_import_order(model_path, mlir_file)
    except Exception as e:
        print(f"ERROR: Failed to verify: {e}", file=sys.stderr)
        return False

    # Print summary
    print(f"ONNX ops: {len(result['onnx_ops'])}")
    print(f"MLIR ops: {len(result['mlir_ops'])}")
    print(f"Counts match: {'YES' if result['counts_match'] else 'NO'}")
    print(f"Full match: {'YES' if result['matches'] else 'NO'}")

    if result['mismatches']:
        print(f"\nMismatches ({len(result['mismatches'])}):")
        for m in result['mismatches'][:10]:  # Show first 10
            print(f"  [{m['type']}] pos={m['position']}: ONNX={m['onnx']} vs MLIR={m['mlir']}")

    # Determine pass/fail
    type_mismatches = [m for m in result['mismatches'] if m['type'] == 'op_type_mismatch']

    if not result['counts_match']:
        print("\nFAILED: Op count mismatch", file=sys.stderr)
        return False

    if type_mismatches:
        print(f"\nFAILED: {len(type_mismatches)} op type mismatches found", file=sys.stderr)
        return False

    print("\nPASSED: Import ordering is correct")
    return True


def verify_model_directory(model_dir: Path, limit: int = None) -> bool:
    """Verify all ONNX models in a directory.

    Returns True if all verifications pass, False otherwise.
    """
    models = list(model_dir.glob("*.onnx"))
    if not models:
        print(f"No ONNX models found in: {model_dir}")
        return True

    if limit:
        models = models[:limit]

    print(f"\nVerifying {len(models)} models from: {model_dir}")
    print("=" * 60)

    passed = 0
    failed = 0

    for model_path in models:
        mlir_file = Path("/tmp") / f"verify_{model_path.stem}.mlir"

        try:
            result = verify_import_order(model_path, mlir_file)
            type_mismatches = [m for m in result['mismatches'] if m['type'] == 'op_type_mismatch']

            if result['counts_match'] and len(type_mismatches) == 0:
                print(f"OK {model_path.name}: {len(result['onnx_ops'])} ops")
                passed += 1
            else:
                print(f"FAILED {model_path.name}: {len(result['onnx_ops'])} ops", file=sys.stderr)
                failed += 1
        except Exception as e:
            print(f"ERROR {model_path.name}: {e}", file=sys.stderr)
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")

    return failed == 0


def main():
    parser = argparse.ArgumentParser(
        description="Verify ONNX to MLIR import ordering"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Path to single ONNX model to verify"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Directory containing ONNX models to verify"
    )
    parser.add_argument(
        "--mlir-output",
        type=Path,
        help="Path to save generated MLIR (optional)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of models to verify (for --model-dir)"
    )

    args = parser.parse_args()

    if args.model_path:
        success = verify_single_model(args.model_path, args.mlir_output)
    elif args.model_dir:
        success = verify_model_directory(args.model_dir, args.limit)
    else:
        # Default: verify test models
        testdata_dir = Path(__file__).parent.parent / "tests" / "testdata" / "onnx_models"
        if testdata_dir.exists():
            success = verify_model_directory(testdata_dir, limit=5)
        else:
            print("Error: No model specified and default testdata not found", file=sys.stderr)
            print("Usage: python scripts/verify_onnx_import_order.py --model-path=./model.onnx")
            sys.exit(1)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
