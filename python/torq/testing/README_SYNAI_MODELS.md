# SynAI Models Testing

Test TFLite models from Hugging Face through the IREE pipeline.

## Quick Start

```bash
# Test a specific model
python test_synai_models.py -t 1

# Test multiple models
python test_synai_models.py -t 1,2,3

# Test a range of models
python test_synai_models.py -t 1-10

# Test all models
python test_synai_models.py -t all
```

## Authentication

Models are downloaded from the private Hugging Face repository `Synaptics/synai_models`.

**Default authentication is automatic** using the embedded Synaptics token.

To override, use one of these methods:

```bash
# Method 1: Login via CLI (recommended)
huggingface-cli login

# Method 2: Environment variable
export HF_TOKEN='your_token_here'
python test_synai_models.py -t 1

# Method 3: Command-line flag
python test_synai_models.py --hf-token your_token_here -t 1
```

Get your token from: https://huggingface.co/settings/tokens

## Model Storage

Models are automatically downloaded to `~/.cache/synai_models/` and reused across runs.

```bash
# Use custom cache directory
python test_synai_models.py --cache-dir /custom/path -t 1

# Skip download check (use existing cache)
python test_synai_models.py --skip-download -t 1
```

## Parallel Execution

```bash
# Use all CPU cores (default)
python test_synai_models.py -t all

# Use 4 parallel workers
python test_synai_models.py -j 4 -t all

# Run sequentially (1 worker)
python test_synai_models.py -j 1 -t all
```

## Additional Options

```bash
# Dry run (show commands without executing)
python test_synai_models.py -t 1 --dry-run

# Set timeout (default: 180 seconds)
python test_synai_models.py -t 1 --timeout 300

# Skip killing qemu processes
python test_synai_models.py -t 1 --no-kill-qemu
```

## Pipeline

Each TFLite model goes through 4 stages:

1. **Import**: `iree-import-tflite` → TOSA
2. **Optimize**: `iree-opt` → MLIR
3. **Compile**: `torq-compile` → VMFB
4. **Execute**: `iree-run-module` → Compare outputs

Input and output `.npy` files are automatically downloaded from Hugging Face.

## Reports

Test reports are saved to `tests/reports/`:

- `synai_models_data_YYYYMMDD_HHMMSS.json` - JSON data
- `synai_models_summary_report.txt` - Text summary
- `synai_models_report_YYYYMMDD_HHMMSS.html` - Interactive HTML report

## Example Output

```
Using 16 parallel workers (all available CPU cores)
✅ Models already cached in: /home/user/.cache/synai_models
Found 3 TFLite files to process

================================================================================
MODEL: model001 - conv_3X3_synai.tflite
================================================================================
Processing: conv_3X3_synai.tflite
  Executing: iree-import-tflite ...
  Executing: iree-opt ...
  Executing: torq-compile ...
  Executing: iree-run-module ...
✅ SUCCESS - All outputs match!

Progress: 3/3 files completed

================================================================================
SUMMARY REPORT
================================================================================
TEST RESULTS: 3/3 passed (100.0%)
  ✅ 3 models processed successfully
  ❌ 0 models failed
```

## Troubleshooting

**No input files found**: NPY files should be automatically downloaded from Hugging Face. Check your authentication.

**Authentication errors**: Use `huggingface-cli login` or verify your token.

**Timeout errors**: Increase timeout with `--timeout 300` or higher.

**Out of memory**: Reduce parallel workers with `-j 4` or `-j 1`.
