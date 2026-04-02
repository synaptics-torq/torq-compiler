"""
DMA Throughput Measurement Test

This test measures real DMA throughput on hardware by:
1. Compiling a large reduce-sum with --torq-fake-reduce (makes compute negligible)
2. Running on hardware with runtime profiling enabled
3. Parsing NSS_WAIT duration from profiling CSV (DMA transfer dominates)
4. Computing and recording bytes/cycle throughput

Usage:
    pytest tests/test_dma_throughput.py
        --torq-runtime-hw-type=astra_machina
        --torq-addr=root@10.3.120.54
        --torq-chips=default.group
        -v --recompute-cache
        --torq-runtime-profiling-output-dir=profile-dma -s
"""

import csv
import statistics
import time
from pathlib import Path

import pytest

from torq.testing.iree import list_mlir_files

# Input sizes for DMA throughput test, selected based on chip target
INPUT_BYTES_400KB = 400000
INPUT_BYTES_200KB = 200000

# Targets with >= 400 KB LRAM that can use the larger test input
LARGE_LRAM_TARGETS = {"SL2610"}

# Clock frequency in MHz (matching debug_info.py CLOCK_FREQ_MHZ)
CLOCK_FREQ_MHZ = 800

# Number of hardware runs per test for stable measurement
NUM_ITERATIONS = 10

# Pause between iterations in seconds (lets DDR bandwidth settle)
ITER_PAUSE_SECONDS = 2

# Module-level collector for per-iteration measurements
_measurements = []


def _get_chip_display_name(chip_config_data):
    """Get a display name for the chip (e.g. 'sl2610-v1', 'default', 'SL2610')."""
    return chip_config_data.get("chip_name", chip_config_data.get("target", "unknown"))


def _get_input_bytes(chip_config_data):
    """Return input size based on chip target.

    Targets with large LRAM (>= 400 KB) use the 400 KB test; others use 200 KB.
    """
    target = chip_config_data.get("target", "")
    if target in LARGE_LRAM_TARGETS:
        return INPUT_BYTES_400KB
    return INPUT_BYTES_200KB


def _get_reducesum_mlir(chip_config_data):
    """Select the right reducesum MLIR test file based on chip target."""
    target = chip_config_data.get("target", "")
    if target in LARGE_LRAM_TARGETS:
        suffix = "_400kb"
    else:
        suffix = "_200kb"

    files = list_mlir_files("linalg_ops")
    for f in files:
        if f"reducesum-dma-throughput-test{suffix}" in str(f):
            return f
    pytest.skip(f"reducesum-dma-throughput-test{suffix}.mlir not found in testdata/linalg_ops")


@pytest.fixture(params=range(NUM_ITERATIONS), ids=[f"iter-{i}" for i in range(NUM_ITERATIONS)])
def case_config(request, runtime_hw_type, chip_config):
    if request.param > 0:
        time.sleep(ITER_PAUSE_SECONDS)
    mlir_file = _get_reducesum_mlir(chip_config.data)
    return {
        "mlir_model_file": "static_mlir_model_file",
        "static_mlir_model_file": mlir_file,
        "input_data": "tweaked_random_input_data",
        "comparison_config": "comparison_config_from_mlir",
        "skip_profile_annotation": request.param + 1,
        "iteration": request.param,
        "torq_compiler_options": [
            "--torq-fake-reduce"
        ],
    }


def _extract_nss_wait_duration_us(profiling_csv: str):
    """
    Parse a runtime profiling CSV and return the last NSS_WAIT duration in microseconds.
    The last NSS_WAIT corresponds to the DMA-in transfer of the full input tensor.
    """
    nss_wait_begin = None
    last_duration_us = None

    with open(profiling_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            event = row["event"].strip()
            timestamp_us = float(row["timestamp_us"].strip())

            if event == "NSS_WAIT_BEGIN":
                nss_wait_begin = timestamp_us
            elif event == "NSS_WAIT_END" and nss_wait_begin is not None:
                last_duration_us = timestamp_us - nss_wait_begin
                nss_wait_begin = None

    return last_duration_us


def test_dma_throughput(request, torq_results_dir):
    """
    Measure and record DMA throughput on hardware.
    """
    chip_data = request.getfixturevalue("chip_config").data
    chip_name = _get_chip_display_name(chip_data)
    input_bytes = _get_input_bytes(chip_data)

    profiling_csv = torq_results_dir.dir_path / "host_profile.csv"

    if not profiling_csv.exists():
        pytest.fail(
            "Runtime profiling CSV not found. Run with "
            "--torq-runtime-profiling-output-dir to enable profiling."
        )

    duration_us = _extract_nss_wait_duration_us(str(profiling_csv))

    if duration_us is None or duration_us <= 0:
        pytest.fail("Could not extract NSS_WAIT duration from profiling CSV")

    duration_cycles = duration_us * CLOCK_FREQ_MHZ
    measured_bytes_per_cycle = input_bytes / duration_cycles

    print(f"\n=== DMA Throughput Measurement ===")
    print(f"Chip:               {chip_name}")
    print(f"Clock:              {CLOCK_FREQ_MHZ} MHz")
    print(f"Input size:         {input_bytes} bytes ({input_bytes / 1024:.1f} KB)")
    print(f"NSS_WAIT duration:  {duration_us:.2f} us")
    print(f"Duration cycles:    {duration_cycles:.0f}")
    print(f"Measured:           {measured_bytes_per_cycle:.3f} bytes/cycle")

    _measurements.append(measured_bytes_per_cycle)
    if len(_measurements) == NUM_ITERATIONS:
        median = statistics.median(_measurements)

        print(f"\n=== DMA Throughput Summary ({NUM_ITERATIONS} iterations) ===")
        print(f"Chip:             {chip_name}")
        print(f"All measurements: {[f'{x:.3f}' for x in _measurements]}")
        print(f"Median:           {median:.3f} bytes/cycle")
