import os
import argparse
from huggingface_hub import snapshot_download, hf_hub_download
import shutil
from pathlib import Path
import sys

"""
This script downloads and copies specified models to a given destination directory.

- `HF_MODEL_LIST`: Hugging Face models to be downloaded.
- `TOSA_MODEL_LIST`: TOSA test case files to be copied.
- `LINALG_MODEL_LIST`: LINALG test case files to be copied.

Usage:
        python model_release.py <dest_dir>
Arguments:
        dest_dir: Destination directory to store release models.
"""

HF_MODEL_LIST = [
    ["Synaptics/MobileNetV2", "MobileNetV2_int8.tflite"],
    ["Synaptics/MobileNetV2", "MobileNetV2_int8.mlir"]
]
TOSA_MODEL_LIST = [
    "add.mlir",
    "mul-bf16.mlir",
    "clamp-bf16.mlir"
]
LINALG_MODEL_LIST = [
    "batch-matmul-in-int8-out-int16.mlir",
    "matvec-in-int16-out-int16.mlir"
]

TOPDIR = Path(__file__).parent.parent
MODELS_DIR = TOPDIR / 'tests/testdata/'
TOSA_SRC_DIR = MODELS_DIR / "tosa_ops"
LINALG_SRC_DIR = MODELS_DIR / "linalg_ops"

def download_hf_model_file(model_list, dest_dir, subfolder):
    subfolder_path = os.path.join(dest_dir, subfolder)
    os.makedirs(subfolder_path, exist_ok=True)
    for repo_id, filename in model_list:
        repo_dir_name = repo_id.replace("/", "_")
        repo_dir = os.path.join(subfolder_path, repo_dir_name)
        os.makedirs(repo_dir, exist_ok=True)
        model_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename
        )
        target_path = os.path.join(repo_dir, filename)
        shutil.copy2(model_file, target_path)
        print(f"Saved {filename} to {target_path}")

def copy_test_cases(model_list, src_dir, dest_dir, subfolder):
    subfolder_path = os.path.join(dest_dir, subfolder)
    os.makedirs(subfolder_path, exist_ok=True)
    for model_id in model_list:
        print(f"Copying {model_id} ...")
        src_path = os.path.join(src_dir, model_id)
        dest_path = os.path.join(subfolder_path, model_id)
        shutil.copy2(src_path, dest_path)
        print(f"Saved {model_id} to {dest_path}")

def main():
    parser = argparse.ArgumentParser(description="Download/copy models to a destination folder.")
    parser.add_argument("dest_dir", help="Destination directory to store models")
    args = parser.parse_args()
    os.makedirs(args.dest_dir, exist_ok=True)
    download_hf_model_file(HF_MODEL_LIST, args.dest_dir, "hf")
    copy_test_cases(TOSA_MODEL_LIST, TOSA_SRC_DIR, args.dest_dir, "tosa")
    copy_test_cases(LINALG_MODEL_LIST, LINALG_SRC_DIR, args.dest_dir, "linalg")

if __name__ == "__main__":
    main()
