import os
import argparse
from huggingface_hub import snapshot_download, hf_hub_download
import shutil
from pathlib import Path
import sys

"""
This script downloads and copies specified models to a given destination directory.

- `HF_MODEL_LIST`: Hugging Face models to be downloaded.

Usage:
        python model_release.py <dest_dir>
Arguments:
        dest_dir: Destination directory to store release models.
"""

HF_MODEL_LIST = [
    ["Synaptics/MobileNetV2", "MobileNetV2_int8.tflite"],
    ["Synaptics/MobileNetV2", "MobileNetV2_int8.mlir"]
]

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

def main():
    parser = argparse.ArgumentParser(description="Download/copy models to a destination folder.")
    parser.add_argument("dest_dir", help="Destination directory to store models")
    args = parser.parse_args()
    os.makedirs(args.dest_dir, exist_ok=True)
    download_hf_model_file(HF_MODEL_LIST, args.dest_dir, "hf")

if __name__ == "__main__":
    main()
