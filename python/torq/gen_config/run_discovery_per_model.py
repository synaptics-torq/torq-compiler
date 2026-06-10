#!/usr/bin/env python3
"""
Run ONNX executor discovery model-by-model.

This script iterates over ONNX models and runs pytest discovery for each
model in isolation.  This guarantees that one model finishes completely
before the next starts — no shared discovery state, no cross-model
pollution, and easy to resume if a single model fails.

Usage:
    # Discover all models in a local directory
    python python/torq/gen_config/run_discovery_per_model.py \
        --model-dir=/home/xshang/proj/models --output-dir=./result

    # Discover models from default HF repos (defined in discovery.py)
    python python/torq/gen_config/run_discovery_per_model.py --output-dir=./result

    # Filter models by name
    python python/torq/gen_config/run_discovery_per_model.py \
        --model-dir=/home/xshang/proj/models --output-dir=./result --model-filter="resnet*"

    # Extra pytest arguments
    python python/torq/gen_config/run_discovery_per_model.py \
        --model-dir=/home/xshang/proj/models --output-dir=./result -- -s --tb=short
"""

import argparse
import fnmatch
import subprocess
import sys
from pathlib import Path

from torq.gen_config.discovery import DEFAULT_HF_REPOS, _DEFAULT_ONNX_JSON_DIR


def _resolve_models(args) -> list:
    """Resolve the list of ONNX model files to process."""
    models = []

    if args.model_dir:
        folder = Path(args.model_dir)
        # Use rglob but limit depth for performance; HF cache has models under snapshots/
        models = sorted(folder.rglob("snapshots/*/*.onnx"))
        if not models:
            models = sorted(folder.glob("*.onnx"))
        if not models:
            print(f"No ONNX models found in {folder}", file=sys.stderr)
            sys.exit(1)
    elif args.model_path:
        path = Path(args.model_path)
        if not path.exists():
            print(f"--model-path does not exist: {path}", file=sys.stderr)
            sys.exit(1)
        models = [path]
    else:
        # Default: download from HF repos
        try:
            from huggingface_hub import list_repo_files, hf_hub_download
        except ImportError:
            print(
                "huggingface_hub not installed. Install with: pip install huggingface_hub",
                file=sys.stderr,
            )
            sys.exit(1)

        # Pre-filter repos before downloading to avoid unnecessary API calls
        repos_to_process = DEFAULT_HF_REPOS
        if args.model_filter:
            pattern = args.model_filter.lower()
            repos_to_process = [r for r in repos_to_process if fnmatch.fnmatch(r.lower(), f"*{pattern}*")]
            if not repos_to_process:
                print(f"No HF repos match filter: {args.model_filter}", file=sys.stderr)
                sys.exit(1)

        for repo_id in repos_to_process:
            try:
                files = [f for f in list_repo_files(repo_id) if f.endswith(".onnx")]
            except Exception as e:
                print(f"[HF] Error listing files in {repo_id}: {e}", file=sys.stderr)
                continue
            if not files:
                print(f"[HF] Warning: no *.onnx in {repo_id}", file=sys.stderr)
                continue
            try:
                local_path = hf_hub_download(repo_id=repo_id, filename=files[0])
            except Exception as e:
                print(f"[HF] Error downloading {repo_id}/{files[0]}: {e}", file=sys.stderr)
                continue
            models.append(Path(local_path))
            print(f"[HF] {repo_id}/{files[0]} -> {local_path}")

    return models


def _run_pytest_for_model(model_path: Path, output_dir: Path, extra_pytest_args: list) -> int:
    """Run pytest discovery for a single model. Returns exit code."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_onnx_gen_config.py",
        "-v",
        f"--model-path={model_path}",
        f"--output-dir={output_dir}",
        "--auto-convert-bf16",
        "--skip-executors=css",
        "--dedup-layers",
        "--skip-mode",
        '-k', 'layer',
    ] + extra_pytest_args

    print(f"\n{'='*60}")
    print(f"Discovering: {model_path.name}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")

    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Run ONNX executor discovery model-by-model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model-dir=/home/xshang/proj/models --output-dir=./result
  %(prog)s --output-dir=./result --model-filter="resnet*"
  %(prog)s --model-dir=/home/xshang/proj/models --output-dir=./result -- -s --tb=short
""",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Directory containing *.onnx models",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Single ONNX model file (overrides --model-dir)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_DEFAULT_ONNX_JSON_DIR),
        help=f"Directory to save discovery JSON output (default: {_DEFAULT_ONNX_JSON_DIR})",
    )
    parser.add_argument(
        "--model-filter",
        default=None,
        help="Filter models by fnmatch pattern (e.g., 'resnet*', '*bert*')",
    )
    parser.add_argument(
        "extra_pytest_args",
        nargs="*",
        help="Extra arguments passed to pytest (use -- before them)",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = _resolve_models(args)
    print(f"\nTotal models to discover: {len(models)}\n")

    failed = []
    passed = []

    for i, model_path in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] {model_path.name}")
        rc = _run_pytest_for_model(model_path, output_dir, args.extra_pytest_args or [])
        if rc == 0:
            passed.append(model_path.name)
        else:
            failed.append(model_path.name)

    print(f"\n{'='*60}")
    print("Discovery complete")
    print(f"{'='*60}")
    print(f"Passed: {len(passed)}  Failed: {len(failed)}")
    if failed:
        print(f"Failed models: {', '.join(failed)}")
        sys.exit(1)
    print("All models discovered successfully.")


if __name__ == "__main__":
    main()
