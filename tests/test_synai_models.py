#!/usr/bin/env python3
"""
Script to test TFLite models from synai_models through IREE.

Authentication Required:
    This script downloads models from a private Hugging Face repository.
    Authentication is automatic using the default Synaptics token.
    
    To override, use one of these methods:
    Method 1: huggingface-cli login
    Method 2: export HF_TOKEN='your_token'
    Method 3: ./test_synai_models.py --hf-token your_token
    
    Get token from: https://huggingface.co/settings/tokens

Usage:
    ./test_synai_models.py                      # Download and process all model folders (parallel, all CPU cores)
    ./test_synai_models.py -t all               # Process all model folders (explicit)
    ./test_synai_models.py -t 1,2,3             # Process only model001, model002, model003
    ./test_synai_models.py -t 1-5               # Process model001 through model005
    ./test_synai_models.py -t 1,3-5,8           # Process model001, model003, model004, model005, model008
    ./test_synai_models.py -t 40 --dry-run      # Print commands without executing for model040
    ./test_synai_models.py --timeout 300        # Set command timeout to 300 seconds (default is 180)
    ./test_synai_models.py --no-kill-qemu       # Skip killing qemu-system-riscv processes (kills by default)
    ./test_synai_models.py -j 4                 # Use 4 parallel workers
    ./test_synai_models.py -j 1                 # Run sequentially (1 worker)
    ./test_synai_models.py --parallel 8         # Use 8 parallel workers
    ./test_synai_models.py --cache-dir /path    # Specify custom cache directory for downloaded models
    ./test_synai_models.py --skip-download      # Skip downloading models (use existing cache)
    ./test_synai_models.py --hf-token TOKEN     # Provide Hugging Face token via command line
"""

import os
import glob
import re
import argparse
import subprocess
import sys
import time
import signal
import shutil
import json
import numpy as np
import datetime
import multiprocessing
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Add parent directory to path to import from python/torq
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
from torq.utils.report_generator import TestReportGenerator, ensure_log_in_common_location, get_display_stage

# Global variables to track processes and thread safety
current_process = None
is_exiting = False
active_processes = set()
process_lock = Lock()
executor = None

# Signal handler for graceful termination
def signal_handler(sig, frame):
    global is_exiting, executor
    
    print("\nInterrupt received. Cleaning up and exiting...")
    is_exiting = True
    
    # Shutdown executor if it exists
    if executor is not None:
        print("Shutting down thread pool...")
        executor.shutdown(wait=False, cancel_futures=True)
    
    # Kill all active processes
    with process_lock:
        for proc in list(active_processes):
            try:
                print(f"Terminating process (PID: {proc.pid})...")
                kill_process_and_children(proc)
            except:
                pass
        active_processes.clear()
    
    # Kill the current process if there is one
    if current_process is not None:
        try:
            print(f"Terminating currently running process (PID: {current_process.pid})...")
            kill_process_and_children(current_process)
        except:
            pass
    
    # Final cleanup of QEMU processes
    print("Performing final cleanup...")
    check_and_kill_qemu_processes()
    
    print("Exiting.")
    sys.exit(1)

# Register the signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)

def get_cache_directory(custom_cache_dir=None):
    """Get the cache directory for downloaded models."""
    if custom_cache_dir:
        cache_dir = Path(custom_cache_dir).expanduser().resolve()
    else:
        # Default to ~/.cache/synai_models
        cache_dir = Path.home() / '.cache' / 'synai_models'
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def download_models_from_huggingface(cache_dir, target_models=None, token=None):
    """
    Download models from Hugging Face repository.
    
    Args:
        cache_dir: Directory to store downloaded models
        target_models: Optional set of model numbers to download (None = all)
        token: Hugging Face authentication token (for private repos)
    
    Returns:
        Path to the models directory in cache
    """
    print("=" * 80)
    print("DOWNLOADING MODELS FROM HUGGING FACE")
    print("=" * 80)
    
    try:
        from huggingface_hub import snapshot_download, HfFolder
    except ImportError:
        print("❌ Error: huggingface_hub is not installed.")
        print("Install it with: pip install huggingface_hub")
        sys.exit(1)
    
    repo_id = "Synaptics/synai_models"
    
    # Default token for Synaptics internal use
    DEFAULT_TOKEN = "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"
    
    # Get token from multiple sources (priority order)
    if not token:
        # 1. Check environment variable
        token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
        
        # 2. Check if user is logged in via huggingface-cli
        if not token:
            try:
                token = HfFolder.get_token()
            except:
                pass
        
        # 3. Use default token as fallback
        if not token:
            token = DEFAULT_TOKEN
    
    # Inform user about authentication
    if token == DEFAULT_TOKEN:
        print("✅ Using default Synaptics authentication token")
    elif token:
        print("✅ Using Hugging Face authentication token")
    else:
        print("⚠️  No authentication token found.")
        print("For private repositories, you need to authenticate:")
        print("  Option 1: Run 'huggingface-cli login' (recommended)")
        print("  Option 2: Set HF_TOKEN environment variable")
        print("  Option 3: Use --hf-token flag")
        print("\nAttempting download without authentication...")
    
    # If specific models are requested, download only those folders
    if target_models:
        print(f"Downloading {len(target_models)} specific model(s): {sorted(target_models)}")
        allow_patterns = []
        for model_num in target_models:
            model_folder = f"model{model_num:03d}"
            allow_patterns.append(f"{model_folder}/*")
        
        print(f"Download patterns: {allow_patterns}")
    else:
        print("Downloading all models...")
        allow_patterns = ["model*/*"]
    
    print("📦 Downloading: .tflite files, .npy input/output files, and other model data")
    
    try:
        print(f"Cache directory: {cache_dir}")
        print(f"Repository: {repo_id}")
        print("Starting download...")
        
        local_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=cache_dir,
            allow_patterns=allow_patterns,
            resume_download=True,
            local_dir_use_symlinks=False,
            token=token
        )
        
        print(f"✅ Models downloaded successfully to: {local_dir}")
        
        # Verify downloaded files
        _verify_downloaded_files(cache_dir, target_models)
        
        return Path(local_dir)
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error downloading models: {error_msg}")
        
        # Provide helpful error messages for common issues
        if "401" in error_msg or "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
            print("\n" + "=" * 80)
            print("AUTHENTICATION REQUIRED")
            print("=" * 80)
            print("This is a private repository. Please authenticate using one of these methods:")
            print("\n📍 METHOD 1 - Login via CLI (Recommended):")
            print("   huggingface-cli login")
            print("   # Enter your Hugging Face token when prompted")
            print("\n📍 METHOD 2 - Environment Variable:")
            print("   export HF_TOKEN='your_token_here'")
            print("   ./run_test.py -t 1")
            print("\n📍 METHOD 3 - Command-line Flag:")
            print("   ./run_test.py --hf-token your_token_here -t 1")
            print("\n💡 Get your token from: https://huggingface.co/settings/tokens")
            print("=" * 80)
        elif "404" in error_msg or "not found" in error_msg.lower():
            print("\n⚠️  Repository not found. Check:")
            print("   - Repository name: Synaptics/synai_models")
            print("   - You have access to this private repository")
            print("   - Repository exists on Hugging Face")
        
        sys.exit(1)

def _verify_downloaded_files(cache_dir, target_models=None):
    """
    Verify that downloaded model directories contain required files.
    
    Args:
        cache_dir: Directory where models are cached
        target_models: Optional set of model numbers to verify
    """
    cache_path = Path(cache_dir)
    model_dirs = sorted(cache_path.glob("model*"))
    
    if target_models:
        model_pattern = re.compile(r'model(\d+)')
        model_dirs = [d for d in model_dirs 
                     if model_pattern.match(d.name) and 
                     int(model_pattern.match(d.name).group(1)) in target_models]
    
    print("\n📋 Verifying downloaded files...")
    for model_dir in model_dirs[:5]:  # Show first 5 as sample
        tflite_files = list(model_dir.glob("*.tflite"))
        npy_files = list(model_dir.glob("*.npy"))
        
        print(f"  {model_dir.name}: {len(tflite_files)} .tflite, {len(npy_files)} .npy files")
    
    if len(model_dirs) > 5:
        print(f"  ... and {len(model_dirs) - 5} more model directories")
    
    # Count total files
    total_tflite = sum(len(list(d.glob("*.tflite"))) for d in model_dirs)
    total_npy = sum(len(list(d.glob("*.npy"))) for d in model_dirs)
    
    print(f"\n✅ Total: {total_tflite} TFLite files, {total_npy} NPY files across {len(model_dirs)} model directories")

def verify_models_in_cache(cache_dir, target_models=None):
    """
    Verify that required models exist in cache.
    
    Args:
        cache_dir: Directory where models should be cached
        target_models: Optional set of model numbers to verify (None = verify any exist)
    
    Returns:
        True if models exist, False otherwise
    """
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        return False
    
    # Find all model directories
    model_dirs = list(cache_path.glob("model*"))
    
    if not model_dirs:
        return False
    
    # If specific models requested, verify they exist
    if target_models:
        existing_models = set()
        model_pattern = re.compile(r'model(\d+)')
        
        for model_dir in model_dirs:
            match = model_pattern.match(model_dir.name)
            if match:
                existing_models.add(int(match.group(1)))
        
        missing_models = target_models - existing_models
        if missing_models:
            print(f"Missing models in cache: {sorted(missing_models)}")
            return False
    
    return True

def parse_model_numbers(model_str):
    """Parse the comma-separated list of model numbers/ranges."""
    if not model_str:
        return []
        
    result = set()
    parts = model_str.split(',')
    
    for part in parts:
        if '-' in part:
            # It's a range
            try:
                start, end = map(int, part.split('-'))
                result.update(range(start, end + 1))
            except ValueError:
                print(f"Warning: Invalid range '{part}'. Skipping.")
        else:
            # It's a single number
            try:
                result.add(int(part))
            except ValueError:
                print(f"Warning: Invalid number '{part}'. Skipping.")
                
    return sorted(result)

def find_input_files(model_dir):
    """Find all input NPY files (input_*.npy) downloaded from Hugging Face."""
    input_npy_pattern = re.compile(r'^input_\d+\.npy$')
    input_files = []
    
    for file in os.listdir(model_dir):
        if input_npy_pattern.match(file):
            input_files.append(os.path.join(model_dir, file))
    
    # Sort by input number to ensure consistent ordering
    return sorted(input_files)

def find_output_files(model_dir):
    """Find all output NPY files (output_*.npy) downloaded from Hugging Face."""
    output_npy_pattern = re.compile(r'^output_\d+\.npy$')
    output_files = []
    
    for file in os.listdir(model_dir):
        if output_npy_pattern.match(file):
            # Skip test output files
            if not file.endswith('_test.npy'):
                output_files.append(os.path.join(model_dir, file))
    
    # Sort by output number to ensure consistent ordering
    return sorted(output_files)

def kill_process_and_children(proc):
    """Kill a process and all its child processes. Also kill any qemu-system-riscv processes."""
    try:
        # Get the process group ID
        pgid = os.getpgid(proc.pid)
        # Kill the entire process group
        os.killpg(pgid, signal.SIGTERM)
        print(f"Process timed out and was terminated (PID: {proc.pid}, PGID: {pgid})")
        
        # If it's still alive after SIGTERM, use SIGKILL
        try:
            if proc.poll() is None:
                time.sleep(1)
                if proc.poll() is None:
                    os.killpg(pgid, signal.SIGKILL)
                    print(f"Process forcefully killed with SIGKILL")
        except:
            pass
            
        # Also check for and kill any qemu-system-riscv processes
        try:
            # Find qemu-system-riscv processes
            qemu_check_cmd = "ps -ef | grep qemu-system-ris | grep -v grep"
            qemu_result = subprocess.run(qemu_check_cmd, shell=True, text=True, capture_output=True)
            
            if qemu_result.stdout.strip():
                print("Found qemu-system-riscv processes, attempting to terminate them:")
                print(qemu_result.stdout.strip())
                
                # Extract PIDs
                qemu_pids = []
                for line in qemu_result.stdout.strip().split('\n'):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            pid = int(parts[1])
                            qemu_pids.append(pid)
                        except ValueError:
                            continue
                
                # Kill each qemu process
                for pid in qemu_pids:
                    try:
                        os.kill(pid, signal.SIGTERM)
                        print(f"Sent SIGTERM to qemu process with PID: {pid}")
                        
                        # Check if it's still running after a short wait
                        time.sleep(1)
                        try:
                            os.kill(pid, 0)  # This will raise an error if the process doesn't exist
                            os.kill(pid, signal.SIGKILL)
                            print(f"Sent SIGKILL to qemu process with PID: {pid}")
                        except OSError:
                            print(f"qemu process with PID {pid} successfully terminated")
                    except OSError as e:
                        print(f"Error killing qemu process {pid}: {e}")
        except Exception as e:
            print(f"Error checking for qemu processes: {e}")
    except Exception as e:
        print(f"Error while killing process {proc.pid}: {e}")
        # Fallback: try to kill just the main process
        try:
            proc.kill()
            print(f"Killed only the main process")
        except:
            pass

def compare_npy_files(ref_file, test_file):
    """Compare two .npy files and return a string with the comparison results."""
    try:
        # Load the files
        ref_data = np.load(ref_file)
        test_data = np.load(test_file)
        
        # Check shapes match
        if ref_data.shape != test_data.shape:
            return f"Shape mismatch: {ref_data.shape} vs {test_data.shape}", False
            
        # Calculate differences
        abs_diff = np.abs(ref_data - test_data)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        diff_positions = np.where(abs_diff > 0)
        
        # Check for differences greater than 1
        large_diff_positions = np.where(abs_diff > 1)
        has_large_diffs = large_diff_positions[0].size > 0
        max_large_diff = np.max(abs_diff) if has_large_diffs else 0
        
        if max_diff == 0:
            return "Arrays are identical", True
        
        # Format a result summary
        result = f"Max difference: {max_diff}\n"
        result += f"Mean difference: {mean_diff}\n"
        
        # Flag if any difference is > 1
        if has_large_diffs:
            result += f"FAILURE: Found {large_diff_positions[0].size} values with difference > 1 (max: {max_large_diff})\n"
            
        # Show some example differences (up to 5)
        if diff_positions[0].size > 0:
            result += "Sample differences:\n"
            
            # If we have large differences, prioritize showing those
            if has_large_diffs:
                display_positions = large_diff_positions
                result += "Large differences (> 1):\n"
            else:
                display_positions = diff_positions
            
            for i in range(min(5, display_positions[0].size)):
                idx = tuple(p[i] for p in display_positions)
                result += f"  Position {idx}: {ref_data[idx]} vs {test_data[idx]} (diff: {abs_diff[idx]})\n"
        
        # Return result and success status (success = no large diffs)
        return result, not has_large_diffs
    except Exception as e:
        return f"Error comparing files: {str(e)}", False

def process_tflite_file(tflite_file, dry_run=False, timeout=180):
    """Process a single TFLite file through the IREE pipeline."""
    model_dir = os.path.dirname(tflite_file)
    base_name = os.path.splitext(tflite_file)[0]
    model_name = os.path.basename(base_name)
    tosa_file = f"{base_name}.tosa"
    mlir_file = f"{base_name}.mlir"
    vmfb_file = f"{base_name}.vmfb"
    
    # Create log directory for outputs
    log_dir = os.path.join(model_dir, "logs")
    # Always create log directory, even during dry run
    os.makedirs(log_dir, exist_ok=True)
    
    # Log file for the entire process - include model directory name to ensure uniqueness
    model_dir_name = os.path.basename(model_dir)
    tflite_basename = os.path.splitext(os.path.basename(tflite_file))[0]
    process_log_file = os.path.join(log_dir, f"{model_dir_name}_{tflite_basename}_process.log")
    
    # Find all input files for this model
    input_files = find_input_files(model_dir)
    if not input_files:
        print(f"WARNING: No input files found for {tflite_file}")
    
    # Find reference output files (JSON files converted to NPY)
    output_files = find_output_files(model_dir)
    
    # Create test output files for each output
    test_output_files = []
    for i, _ in enumerate(output_files):
        test_output_files.append(os.path.join(model_dir, f"output_{i}_test.npy"))
    
    # Prepare input arguments for iree-run-module
    input_args = []
    for input_file in input_files:
        input_args.append(f'--input="@{input_file}"')
    
    # Prepare output arguments for iree-run-module
    output_args = []
    for test_output_file in test_output_files:
        output_args.append(f'--output="@{test_output_file}"')
    
    # Define commands
    commands = [
        f"iree-import-tflite {tflite_file} -o {tosa_file}",
        f"iree-opt {tosa_file} -o {mlir_file}",
        f"torq-compile iree-compile {mlir_file} -o {vmfb_file}",
        f"iree-run-module --device=torq --module={vmfb_file} --function=main {' '.join(input_args)} {' '.join(output_args)}"
    ]
    
    # Command names for log files and stages
    cmd_names = ["import", "opt", "compile", "run"]
    stage_names = ["tflite_to_tosa", "tosa_to_mlir", "mlir_to_vmfb", "execute_module"]
    
    # Process commands
    print(f"Processing: {os.path.basename(tflite_file)}")
    process_log = []
    process_log.append(f"Processing: {tflite_file}\n")
    process_log.append(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    process_log.append("-" * 80 + "\n")
    
    # Track failure stage if any
    failure_stage = None
    
    for i, (cmd, cmd_name) in enumerate(zip(commands, cmd_names)):
        cmd_log_file = os.path.join(log_dir, f"{model_dir_name}_{tflite_basename}_{cmd_name}.log")
        
        if dry_run:
            print(f"  [DRY RUN] {cmd}")
        else:
            print(f"  Executing: {cmd}")
            process_log.append(f"Executing: {cmd}\n")
            
            try:
                # Start the process in its own process group
                global current_process
                process = subprocess.Popen(
                    cmd, 
                    shell=True, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True,
                    preexec_fn=os.setsid  # Create a new process group
                )
                current_process = process
                
                # Add to active processes (thread-safe)
                with process_lock:
                    active_processes.add(process)
                
                # Use the timeout value provided
                timed_out = False
                start_time = time.time()
                
                try:
                    stdout, stderr = process.communicate(timeout=timeout)
                    exit_code = process.returncode
                except subprocess.TimeoutExpired:
                    # Kill the process and all its children if it times out
                    timed_out = True
                    kill_process_and_children(process)
                    
                    # Try to capture any output that was produced before the timeout
                    try:
                        stdout, stderr = process.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        stdout = "Output collection timed out"
                        stderr = "Error output collection timed out"
                    
                    exit_code = -9  # Signal for killed process
                
                execution_time = time.time() - start_time
                
                # Remove from active processes (thread-safe)
                with process_lock:
                    active_processes.discard(process)
                
                # Clear the current process reference
                current_process = None
                
                # Check if we're exiting due to keyboard interrupt
                if is_exiting:
                    print("Keyboard interrupt detected, stopping processing...")
                    return (False, "keyboard_interrupt", None)
                
                # Save output to log file
                with open(cmd_log_file, 'w') as f:
                    f.write(f"COMMAND: {cmd}\n")
                    f.write(f"EXECUTED: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"EXECUTION TIME: {execution_time:.2f} seconds\n")
                    
                    if timed_out:
                        f.write(f"ERROR: Command timed out after {timeout} seconds and was terminated\n")
                    elif exit_code != 0:
                        f.write(f"ERROR: Command failed with exit code {exit_code}\n")
                    
                    f.write("-" * 80 + "\n\n")
                    f.write("STDOUT:\n")
                    f.write(stdout)
                    f.write("\n\nSTDERR:\n")
                    f.write(stderr)
                
                if stdout.strip():
                    print(stdout.strip())
                    process_log.append(f"STDOUT:\n{stdout.strip()}\n")
                
                process_log.append(f"Command output saved to: {cmd_log_file}\n")
                
                # Handle command failure (timeout or non-zero exit code)
                if timed_out or exit_code != 0:
                    error_msg = f"ERROR: Command {'timed out after '+str(timeout)+' seconds' if timed_out else f'failed with exit code {exit_code}'}"
                    print(error_msg)
                    if timed_out:
                        print(f"Process was killed after exceeding the {timeout} second timeout")
                    print(f"STDOUT: {stdout}")
                    print(f"STDERR: {stderr}")
                    
                    process_log.append(f"{error_msg}\n")
                    process_log.append(f"STDOUT: {stdout}\n")
                    process_log.append(f"STDERR: {stderr}\n")
                    
                    # Record the failure stage
                    failure_stage = "timeout" if timed_out else stage_names[i]
                    
                    # Write the process log before returning
                    if not dry_run:
                        process_log.append(f"\nProcess aborted due to failure in {stage_names[i]} stage\n")
                        process_log.append(f"Stopped: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        with open(process_log_file, 'w') as f:
                            f.writelines(process_log)
                        
                        # Ensure log is in common location
                        _, _, _ = ensure_log_in_common_location(process_log_file, content=process_log)
                        
                        print(f"  Failed at stage: {stage_names[i]} (step {i+1}/{len(commands)})")
                        print(f"  Process log saved to: {process_log_file}")
                    return (False, failure_stage, None)
                
            except Exception as e:
                error_msg = f"ERROR: Exception occurred while running command: {str(e)}"
                print(error_msg)
                
                # Save error output to log file
                with open(cmd_log_file, 'w') as f:
                    f.write(f"COMMAND: {cmd}\n")
                    f.write(f"EXECUTED: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"ERROR: Exception occurred: {str(e)}\n")
                    f.write("-" * 80 + "\n\n")
                
                process_log.append(f"{error_msg}\n")
                process_log.append(f"Error output saved to: {cmd_log_file}\n")
                
                # Record the failure stage
                failure_stage = stage_names[i]
                
                # Write the process log before returning
                process_log.append(f"\nProcess aborted due to failure in {stage_names[i]} stage\n")
                process_log.append(f"Stopped: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                with open(process_log_file, 'w') as f:
                    f.writelines(process_log)
                
                # Ensure log is in common location
                _, _, _ = ensure_log_in_common_location(process_log_file, content=process_log)
                
                print(f"  Failed at stage: {stage_names[i]} (step {i+1}/{len(commands)})")
                print(f"  Process log saved to: {process_log_file}")
                return (False, failure_stage, None)
    
    # In dry-run mode, return early without processing outputs
    if dry_run:
        return (True, None, None)
    
    # Compare outputs with references if not in dry run mode and outputs exist
    if not dry_run and output_files and test_output_files:
        print("\nComparing outputs:")
        process_log.append("\nComparing outputs:\n")
        comparison_log = []
        comparison_success = True
        model_max_diff = None
        
        for ref_file, test_file in zip(output_files, test_output_files):
            if os.path.exists(test_file):
                print(f"\nComparing {os.path.basename(ref_file)} with {os.path.basename(test_file)}:")
                process_log.append(f"\nComparing {os.path.basename(ref_file)} with {os.path.basename(test_file)}:\n")
                
                comparison_result, is_success = compare_npy_files(ref_file, test_file)
                print(comparison_result)
                process_log.append(f"{comparison_result}\n")
                comparison_log.append(f"Comparing {os.path.basename(ref_file)} with {os.path.basename(test_file)}:\n{comparison_result}\n")
                
                # Check if comparison indicates failure
                if not is_success:
                    comparison_success = False
                    if failure_stage is None:
                        if "Shape mismatch" in comparison_result:
                            failure_stage = "shape_mismatch"
                        elif "FAILURE: Found" in comparison_result and "values with difference > 1" in comparison_result:
                            max_diff_match = re.search(r'max: (\d+(\.\d+)?)', comparison_result)
                            max_diff = max_diff_match.group(1) if max_diff_match else "?"
                            failure_stage = "large_difference"
                            model_max_diff = max_diff
                        else:
                            failure_stage = "output_comparison"
            else:
                msg = f"Test output file not found: {test_file}"
                print(msg)
                process_log.append(f"{msg}\n")
                comparison_log.append(f"{msg}\n")
                comparison_success = False
                if failure_stage is None:
                    failure_stage = "missing_output"
        
        # Save comparison results to a separate log file
        if comparison_log:
            comparison_log_file = os.path.join(log_dir, f"{model_name}_comparison.log")
            with open(comparison_log_file, 'w') as f:
                f.write(f"Output Comparison for {mlir_file}\n")
                f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("-" * 80 + "\n\n")
                f.writelines(comparison_log)
            process_log.append(f"Comparison results saved to: {comparison_log_file}\n")
            
            if not comparison_success:
                print(f"  ⚠️ Output comparison check failed!")
                process_log.append("⚠️ Output comparison check failed!\n")
    
    # Determine final success or failure
    is_success = failure_stage is None
    
    if is_success:
        success_msg = f"Successfully processed {os.path.basename(mlir_file)}"
        print(success_msg)
        process_log.append(f"\n{success_msg}\n")
    else:
        failure_msg = f"Failed processing {os.path.basename(mlir_file)} at stage: {failure_stage}"
        print(failure_msg)
        process_log.append(f"\n{failure_msg}\n")
    
    process_log.append(f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Write the log file
    with open(process_log_file, 'w') as f:
        f.writelines(process_log)
            
    print(f"Complete process log saved to: {process_log_file}")
    
    # Ensure the log is also available in the common location
    common_log_path, _, _ = ensure_log_in_common_location(process_log_file, content=process_log)
    
    # Create duplicate log files with simpler naming if needed
    simple_log_path = os.path.join(log_dir, f"{os.path.splitext(os.path.basename(mlir_file))[0]}_process.log")
    if process_log_file != simple_log_path:
        print(f"Creating additional log file with simplified path for HTML reports: {simple_log_path}")
        shutil.copy2(process_log_file, simple_log_path)
    
    # Return success status, failure stage, and max difference if applicable
    return (is_success, failure_stage if not is_success else None, model_max_diff if 'model_max_diff' in locals() else None)

def check_and_kill_qemu_processes():
    """Check for and kill any qemu-system-riscv processes that might be running."""
    try:
        qemu_check_cmd = "ps -ef | grep qemu-system-ris | grep -v grep"
        qemu_result = subprocess.run(qemu_check_cmd, shell=True, text=True, capture_output=True)
        
        if qemu_result.stdout.strip():
            print("\nFound lingering qemu-system-riscv processes, cleaning up:")
            print(qemu_result.stdout.strip())
            
            qemu_pids = []
            for line in qemu_result.stdout.strip().split('\n'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        pid = int(parts[1])
                        qemu_pids.append(pid)
                    except ValueError:
                        continue
            
            for pid in qemu_pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                    print(f"Sent SIGTERM to qemu process with PID: {pid}")
                    
                    time.sleep(1)
                    try:
                        os.kill(pid, 0)
                        os.kill(pid, signal.SIGKILL)
                        print(f"Sent SIGKILL to qemu process with PID: {pid}")
                    except OSError:
                        print(f"qemu process with PID {pid} successfully terminated")
                except OSError as e:
                    print(f"Error killing qemu process {pid}: {e}")
            
            return len(qemu_pids)
        return 0
    except Exception as e:
        print(f"Error checking for qemu processes: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description='Process TFLite models from Hugging Face synai_models repository through IREE pipeline.')
    parser.add_argument('-t', '--targets', type=str, 
                       help='Comma-separated list of model numbers or ranges (e.g., "1,3-5,7"), or "all" for all models')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print commands without executing them')
    parser.add_argument('--timeout', type=int, default=180,
                       help='Maximum execution time (in seconds) for each command (default: 180)')
    parser.add_argument('--no-kill-qemu', action='store_true',
                       help='Skip checking for and killing any lingering qemu-system-riscv processes')
    parser.add_argument('-j', '--parallel', type=int, default=None,
                       help='Number of parallel workers (default: use all available CPU cores)')
    parser.add_argument('--cache-dir', type=str,
                       help='Custom cache directory for downloaded models (default: ~/.cache/synai_models)')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip downloading models from Hugging Face (use existing cache)')
    parser.add_argument('--hf-token', type=str,
                       help='Hugging Face authentication token (for private repos). Can also use HF_TOKEN env variable.')
    args = parser.parse_args()
    
    # Determine number of workers
    if args.parallel is None:
        num_workers = multiprocessing.cpu_count()
        print(f"Using {num_workers} parallel workers (all available CPU cores)")
    elif args.parallel <= 0:
        num_workers = 1
        print("Invalid parallel value, using 1 worker (sequential execution)")
    else:
        num_workers = args.parallel
        print(f"Using {num_workers} parallel worker(s)")
    
    args.num_workers = num_workers
    
    # Check for and kill any lingering qemu processes
    if not args.no_kill_qemu:
        killed_count = check_and_kill_qemu_processes()
        if killed_count > 0:
            print(f"Killed {killed_count} lingering qemu processes before starting")
            print("-" * 80)
    
    # Parse target model numbers if provided
    target_numbers = None
    if args.targets and args.targets.lower() != "all":
        target_numbers = set(parse_model_numbers(args.targets))
    
    # Get cache directory
    cache_dir = get_cache_directory(args.cache_dir)
    
    # Download models if needed
    if not args.skip_download:
        # Check if we need to download
        needs_download = not verify_models_in_cache(cache_dir, target_numbers)
        
        if needs_download:
            download_models_from_huggingface(cache_dir, target_numbers, token=args.hf_token)
        else:
            print(f"✅ Models already cached in: {cache_dir}")
            print("Use --skip-download to skip this check in the future")
            print("-" * 80)
    else:
        print(f"Skipping download, using cached models from: {cache_dir}")
        if not verify_models_in_cache(cache_dir, target_numbers):
            print("⚠️ Warning: Some or all required models may not be in cache")
        print("-" * 80)
    
    # Use cache directory as the models directory
    models_dir = str(cache_dir)
    args.base_dir = models_dir
    
    # Find all subdirectories that match 'model*' pattern
    all_model_dirs = []
    model_number_pattern = re.compile(r'^model(\d+)$')
    
    for item in os.listdir(models_dir):
        dir_path = os.path.join(models_dir, item)
        if os.path.isdir(dir_path):
            match = model_number_pattern.match(item)
            if match:
                all_model_dirs.append((item, int(match.group(1))))
    
    # Special case for "all" target
    if args.targets and args.targets.lower() == "all":
        model_dirs = [dir_name for dir_name, _ in all_model_dirs]
        print(f"Running on ALL model directories ({len(model_dirs)} total)")
    else:
        # Parse target model numbers if provided
        target_numbers = set()
        if args.targets:
            target_numbers = set(parse_model_numbers(args.targets))
        
        # Filter directories based on targets or use all
        if target_numbers:
            model_dirs = [dir_name for dir_name, number in all_model_dirs if number in target_numbers]
        else:
            model_dirs = [dir_name for dir_name, _ in all_model_dirs]
    
    # Sort the directories for consistent output
    model_dirs.sort()
    
    # Track success/failure
    results = []
    results_lock = Lock()
    
    # Collect all TFLite files to process
    all_tflite_files = []
    for model_dir in model_dirs:
        dir_path = os.path.join(models_dir, model_dir)
        tflite_files = glob.glob(os.path.join(dir_path, "*.tflite"))
        
        if not tflite_files:
            print(f"No .tflite files found in {model_dir}")
            continue
        
        all_tflite_files.extend(sorted(tflite_files))
    
    print(f"\nFound {len(all_tflite_files)} TFLite files to process\n")
    
    # Function to process a single file (for parallel execution)
    def process_file_wrapper(tflite_file):
        if is_exiting:
            return None
        
        model_dir = os.path.basename(os.path.dirname(tflite_file))
        print(f"\n{'=' * 80}")
        print(f"MODEL: {model_dir} - {os.path.basename(tflite_file)}")
        print(f"{'=' * 80}")
        
        try:
            success, failure_stage, max_diff = process_tflite_file(
                tflite_file, 
                dry_run=args.dry_run,
                timeout=args.timeout
            )
            return (tflite_file, success, failure_stage, max_diff)
        except Exception as e:
            print(f"ERROR: Failed to process {tflite_file}: {str(e)}")
            import traceback
            traceback.print_exc()
            return (tflite_file, False, "exception", None)
    
    # Process files in parallel
    global executor
    if args.num_workers == 1:
        # Sequential execution
        print("Running in sequential mode (1 worker)\n")
        for tflite_file in all_tflite_files:
            if is_exiting:
                print("\nExiting due to keyboard interrupt...")
                break
            result = process_file_wrapper(tflite_file)
            if result:
                results.append(result)
    else:
        # Parallel execution
        print(f"Running in parallel mode ({args.num_workers} workers)\n")
        executor = ThreadPoolExecutor(max_workers=args.num_workers)
        
        try:
            futures = {executor.submit(process_file_wrapper, tflite_file): tflite_file 
                      for tflite_file in all_tflite_files}
            
            completed = 0
            total = len(futures)
            
            for future in as_completed(futures):
                if is_exiting:
                    print("\nCancelling remaining tasks...")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                
                tflite_file = futures[future]
                try:
                    result = future.result()
                    if result:
                        with results_lock:
                            results.append(result)
                        completed += 1
                        print(f"\nProgress: {completed}/{total} files completed")
                except Exception as e:
                    print(f"ERROR: Task for {tflite_file} failed: {str(e)}")
                    with results_lock:
                        results.append((tflite_file, False, "exception", None))
        finally:
            executor.shutdown(wait=True)
            executor = None
    
    # For dry-run mode, exit early
    if args.dry_run:
        print("\n[DRY RUN] Commands displayed. Use without --dry-run to execute.")
        return
    
    # Print summary
    if results:
        if is_exiting:
            print("\nScript interrupted. Generating report for completed models...")
            
        print("\n" + "=" * 80)
        print("SUMMARY REPORT")
        print("=" * 80)
        
        # Group results by model directory
        model_groups = {}
        for model_path, success, failure_stage, max_diff in results:
            model_dir = os.path.basename(os.path.dirname(model_path))
            
            if model_dir not in model_groups:
                model_groups[model_dir] = []
            model_groups[model_dir].append((model_path, success, failure_stage, max_diff))
        
        # Calculate totals
        success_count = sum(1 for _, success, _, _ in results if success)
        failed_count = len(results) - success_count
        success_rate = (success_count / len(results) * 100) if results else 0
        
        print(f"TEST RESULTS: {success_count}/{len(results)} passed ({success_rate:.1f}%)")
        print(f"  ✅ {success_count} models processed successfully")
        print(f"  ❌ {failed_count} models failed")
        print("-" * 80)
        
        # Show breakdown by directory
        print("BREAKDOWN BY MODEL GROUP:")
        for model_dir in sorted(model_groups.keys()):
            group_results = model_groups[model_dir]
            group_success = sum(1 for _, success, _, _ in group_results if success)
            if group_success == len(group_results):
                status = "✅ All Passed"
            elif group_success == 0:
                status = "❌ All Failed"
            else:
                status = f"⚠️ {group_success}/{len(group_results)} passed"
                
            print(f"  {model_dir}: {status}")
        
        # Group failures by stage
        failures_by_stage = {}
        for model_path, success, failure_stage, max_diff in results:
            if not success and failure_stage:
                if failure_stage not in failures_by_stage:
                    failures_by_stage[failure_stage] = []
                
                model_dir_basename = os.path.basename(os.path.dirname(model_path))
                model_basename = os.path.basename(model_path)
                
                if failure_stage == "large_difference" and max_diff:
                    failures_by_stage[failure_stage].append((model_dir_basename, model_basename, max_diff))
                else:
                    failures_by_stage[failure_stage].append((model_dir_basename, model_basename, None))
        
        # Detailed failures list
        if failed_count > 0:
            print("\nFAILED MODELS BY STAGE:")
            
            for stage, failures in sorted(failures_by_stage.items()):
                display_name = get_display_stage(stage)
                print(f"  • {display_name} Stage Failures ({len(failures)}):")
                
                failures_by_dir = {}
                for failure_item in failures:
                    if len(failure_item) >= 3:
                        model_dir, model_file, max_diff = failure_item
                    else:
                        model_dir, model_file = failure_item[:2]
                        max_diff = None
                        
                    if model_dir not in failures_by_dir:
                        failures_by_dir[model_dir] = []
                    failures_by_dir[model_dir].append((model_file, max_diff))
                
                for model_dir, files in sorted(failures_by_dir.items()):
                    print(f"    - {model_dir}:")
                    for model_file, max_diff in sorted(files):
                        if stage == "large_difference" and max_diff:
                            try:
                                max_diff_formatted = f"{float(max_diff):.6f}"
                            except (ValueError, TypeError):
                                max_diff_formatted = str(max_diff)
                            print(f"      • {model_file} (Max Diff: {max_diff_formatted})")
                        else:
                            print(f"      • {model_file}")
            
            print("\nFAILED MODELS BY DIRECTORY:")
            for model_dir in sorted(model_groups.keys()):
                failed_in_group = [(path, s, fs, md) for path, s, fs, md in model_groups[model_dir] if not s]
                if failed_in_group:
                    print(f"  {model_dir}:")
                    for model_full_path, _, failure_stage, max_diff in failed_in_group:
                        display_stage = get_display_stage(failure_stage, max_diff) if failure_stage else "Unknown"
                        model_file = os.path.basename(model_full_path)
                        print(f"    - {model_file} (Failed at: {display_stage})")
        
        print("\nTest completed at:", time.strftime("%Y-%m-%d %H:%M:%S"))
        
        # Generate JSON data for report generator
        test_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        json_report_data = {
            "test_suite": {
                "name": "SynAI Models",
                "timestamp": test_timestamp,
                "base_dir": models_dir,
                "timeout": args.timeout,
                "dry_run": args.dry_run,
                "targets": args.targets
            },
            "results": []
        }
        
        # Convert results to JSON format
        for model_path, success, failure_stage, max_diff in results:
            model_dir_name = os.path.basename(os.path.dirname(model_path))
            model_name = os.path.basename(model_path)
            model_basename = os.path.splitext(model_name)[0]
            
            # Find log file
            log_file = os.path.join(os.path.dirname(model_path), "logs", f"{model_dir_name}_{model_basename}_process.log")
            if not os.path.exists(log_file):
                log_file = os.path.join(os.getcwd(), "logs", f"{model_basename}_process.log")
            
            json_report_data["results"].append({
                "model_path": model_path,
                "model_dir": model_dir_name,
                "model_name": model_name,
                "success": success,
                "failure_stage": failure_stage,
                "max_diff": str(max_diff) if max_diff is not None else None,
                "log_file": log_file if os.path.exists(log_file) else ""
            })
        
        # Save JSON report data
        # Get the directory where this test script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        reports_dir = os.path.join(script_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        json_data_path = os.path.join(reports_dir, f"synai_models_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(json_data_path, 'w') as f:
            json.dump(json_report_data, f, indent=2)
        print(f"\nTest data saved to: {json_data_path}")
        
        # Generate reports using the new report generator
        report_generator = TestReportGenerator(json_report_data)
        
        # Generate text report
        text_report_path = os.path.join(reports_dir, "synai_models_summary_report.txt")
        report_generator.generate_text_report(text_report_path)
        print(f"Summary report written to: {text_report_path}")
        
        # Generate HTML report in same folder as JSON file
        html_path = report_generator.generate_html_report(json_file_path=json_data_path)
        print(f"\nHTML report generated: {html_path}")
        print(f"View the report in a web browser: file://{html_path}")
    
    # Final cleanup
    if not args.no_kill_qemu and not is_exiting:
        killed_count = check_and_kill_qemu_processes()
        if killed_count > 0:
            print(f"\nCleaned up {killed_count} lingering qemu processes at exit")
            
    if is_exiting:
        print("\nScript execution was interrupted by user (Ctrl+C).")

if __name__ == "__main__":
    main()
