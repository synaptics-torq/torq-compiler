from pathlib import Path
from typing import Sequence, Tuple
import re

import subprocess

import pytest

from torq.testing.iree import list_mlir_files, get_input_type_options
from torq.testing.cases import get_test_cases_from_files
from torq.compile import compile_mlir_for_vm

import logging
from torq.utils.logging import configure_logging

# Shape generation defaults
DEFAULT_LRAM_SIZE = 500  # KB
DEFAULT_NUM_SAMPLES = 15

TEMPLATE_SUBDIR = "template_ops"

@pytest.fixture
def case_config(instantiated_mlir_path: Path, compiler_args_variant: tuple[str, list[str]], request):
    """Minimal case_config so mlir_io_spec can treat instantiated MLIR as the model.

    This wires the shared mlir_model_file / mlir_io_spec fixtures to our
    per-test instantiated_mlir_path without pulling in the full scenario stack.
    """
    name, extra_args = compiler_args_variant
    
    # Combine variant-specific args with any command-line options
    compiler_options = list(extra_args)
    if request.config.getoption("--extra-torq-compiler-options", default=None):
        compiler_options.extend(
            request.config.getoption("--extra-torq-compiler-options").split(" ")
        )

    return {
        "mlir_model_file": "static_mlir_model_file",
        "static_mlir_model_file": instantiated_mlir_path,
        "input_data": "tweaked_random_input_data",
        "comparison_config": "comparison_config_from_mlir",
        "torq_compiler_options": compiler_options,
        "skip_profile_annotation": True,
        "num_samples": DEFAULT_NUM_SAMPLES,
        "lram_size": DEFAULT_LRAM_SIZE,
    }

# ---------------------------------------------------------------------------
#  Data type sizes in bytes
# ---------------------------------------------------------------------------
DTYPE_SIZES = {
    'i8': 1,
    'i16': 2,
    'i32': 4,
    'i64': 8,
    'ui8': 1,
    'ui16': 2,
    'ui32': 4,
    'ui64': 8,
    'f16': 2,
    'f32': 4,
    'f64': 8,
    'bf16': 2,
}

@pytest.fixture
def lram_size(case_config):
    """Configurable maximum tensor size in KB.
    
    Retrieves value from case_config.
    """
    return case_config['lram_size']


@pytest.fixture
def num_samples(case_config):
    """Configurable number of shape samples to generate per (rank, dtype).
    
    Retrieves value from case_config.
    """
    return case_config['num_samples']

def _generate_shapes_for_rank_and_dtype(
    rank: int, 
    dtype: str, 
    lram_size: float,
    num_samples: int
) -> list[Sequence[int]]:
    """Generate shapes for a given rank and dtype that satisfy size constraints.
    
    Args:
        rank: Tensor rank (1, 2, 3, or 4)
        dtype: Data type string (e.g., 'i8', 'bf16', 'f32')
        lram_size: Maximum tensor size in KB
        num_samples: Number of shape variations to generate
        
    Returns:
        List of shapes as sequences of integers
    """
    dtype_size = DTYPE_SIZES.get(dtype, 4)  # Default to 4 bytes (f32/i32)
    max_elements = int((lram_size * 1024) / (dtype_size * 3))
    
    shapes = []
    
    # Generate size_factors with more samples at smaller sizes
    size_factors = []
    for i in range(num_samples):
        # Quadratic distribution to get more small values: 1% to 100%
        linear = i / (num_samples - 1) if num_samples > 1 else 1.0
        factor = 0.01 + 0.99 * (linear ** 2)
        size_factors.append(factor)
    
    if rank == 1:
        # 1D shapes: use size factors
        for factor in size_factors:
            size = int(max_elements * factor)
            if size > 0:
                shapes.append([size])
    
    elif rank == 2:
        # 2D shapes: [batch, features] - batch always 1
        batch = 1
        for factor in size_factors:
            features = int(max_elements * factor)
            if features > 0:
                shapes.append([batch, features])
    
    elif rank == 3:
        # 3D shapes: [batch, sequence, features] - batch always 1
        batch = 1
        # Generate sequence sizes that scale with num_samples
        sequence_sizes = [1, 2, 4, 8, 16, 32, 64]
        for i in range(len(size_factors)):
            seq_idx = min(i, len(sequence_sizes) - 1)
            seq = sequence_sizes[seq_idx]
            factor = size_factors[i] if i < len(size_factors) else size_factors[-1]
            features = int((max_elements * factor) // seq)
            if features > 0:
                shapes.append([batch, seq, features])
    
    elif rank == 4:
        # 4D shapes: [batch, channels, height, width] - batch always 1
        batch = 1
        # Generate channel configs that scale with num_samples
        channel_values = [1, 2, 4, 8, 16, 32, 64]
        for i in range(len(size_factors)):
            ch_idx = min(i, len(channel_values) - 1)
            channels = channel_values[ch_idx]
            factor = size_factors[i] if i < len(size_factors) else size_factors[-1]
            remaining = int((max_elements * factor) // channels)
            # Try to make spatial dimensions roughly square
            side = int(remaining ** 0.5)
            if side > 0:
                shapes.append([batch, channels, side, side])
    
    # Ensure we have at least some shapes
    if not shapes and max_elements > 0:
        if rank == 1:
            shapes = [[max_elements]]
        elif rank == 2:
            shapes = [[1, max_elements]]
        elif rank == 3:
            shapes = [[1, 1, max_elements]]
        elif rank == 4:
            side = int((max_elements / 1) ** 0.5)
            shapes = [[1, 1, side, side]] if side > 0 else [[1, 1, 1, max_elements]]
    
    # Make last dimension a multiple of 64 (round down, minimum 64)
    aligned_shapes = []
    for shape in shapes:
        shape_list = list(shape)
        aligned = (shape_list[-1] // 64) * 64
        shape_list[-1] = aligned if aligned > 0 else 64
        aligned_shapes.append(shape_list)
    
    # Limit to num_samples
    return aligned_shapes[:num_samples]


def _extract_shape_dtype_placeholders(mlir_text: str) -> list[Tuple[int, str]]:
    """Extract shape placeholders with dtypes from MLIR template text.
    
    Looks for patterns like:
        - {shape_1_i8}
        - {shape_2_bf16}
        - {shape_3_i32}
        - {shape_4_f32}
    
    Returns:
        List of (rank, dtype) tuples found in the template
    """
    # Pattern: {shape_<rank>_<dtype>} or {shape_<rank>} (fallback)
    pattern = r'\{shape_(\d+)(?:_([a-z0-9]+))?\}'
    matches = re.findall(pattern, mlir_text)
    
    result = []
    for rank_str, dtype in matches:
        rank = int(rank_str)
        # If no dtype specified, use generic placeholder
        dtype = dtype if dtype else None
        result.append((rank, dtype))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_result = []
    for item in result:
        if item not in seen:
            seen.add(item)
            unique_result.append(item)
    
    return unique_result


@pytest.fixture
def shapes_by_rank_dtype(lram_size, num_samples) -> dict[Tuple[int, str], list[Sequence[int]]]:
    """Generate shapes grouped by (rank, dtype) based on size constraints.
    
    This fixture generates shapes dynamically for common combinations of
    rank and dtype, ensuring each shape satisfies:
        product(shape) * dtype_size * 3 < lram_size
    
    Returns:
        Dictionary mapping (rank, dtype) -> list of shapes
    """
    shapes_dict = {}
    
    # Generate for common combinations
    common_dtypes = ['i8', 'i16', 'i32', 'bf16', 'f16', 'f32']
    ranks = [1, 2, 3, 4]
    
    for rank in ranks:
        for dtype in common_dtypes:
            key = (rank, dtype)
            shapes_dict[key] = _generate_shapes_for_rank_and_dtype(
                rank, dtype, lram_size, num_samples
            )
    
    return shapes_dict


@pytest.fixture
def all_shape_params(shapes_by_rank_dtype) -> list[Tuple[int, str, Sequence[int]]]:
    """Flattened list for parametrization: (rank, dtype, shape).
    
    This generates all combinations of rank, dtype, and corresponding shapes
    from the shapes_by_rank_dtype fixture.
    """
    params = []
    for (rank, dtype), shapes in shapes_by_rank_dtype.items():
        for shape in shapes:
            params.append((rank, dtype, shape))
    return params

# Different compiler arg variants we want to sweep.
# Adjust the names/flags to whatever you care about.
Target_SETS: list[tuple[str, list[str]]] = [
    ("nss", ["--torq-disable-css", "--torq-disable-host"]),
    ("host", ["--torq-disable-slices","--torq-disable-css"]),
    ("css", ["--torq-disable-slices", "--torq-disable-host"]),
]

@pytest.fixture(params=Target_SETS, ids=lambda p: p[0])
def compiler_args_variant(request) -> tuple[str, list[str]]:
    """(name, extra_args) for each compiler configuration."""
    return request.param

def _fmt_shape(shape: Sequence[int]) -> str:
    # For !torch.vtensor<{shape_3},bf16> this becomes: [1,1,64]
    return "[" + ",".join(str(d) for d in shape) + "]"


def _instantiate_template_text(
    mlir_text: str,
    rank: int,
    shape: Sequence[int],
    dtype: str = None,
) -> str:
    """Replace shape placeholders in the template MLIR.

    Supported syntax in the template:

      - {shape_1_i8}, {shape_2_bf16}, {shape_3_i32}, {shape_4_f32}
        -> picks shapes from the corresponding (rank, dtype) combination
      
      - {shape_1}, {shape_2}, {shape_3}, {shape_4}
        -> picks shapes from the corresponding rank list (backward compatible)

      - {shape}
        -> generic placeholder; uses all ranks (1D–4D)

    If no shape_* placeholder is present and {shape} is not present,
    the template is left unchanged but only rank-4 shapes are kept.
    """

    # First, check for dtype-specific placeholders like {shape_3_i32}
    if dtype:
        dtype_token = f"{{shape_{rank}_{dtype}}}"
        if dtype_token in mlir_text:
            return mlir_text.replace(dtype_token, _fmt_shape(shape))
    
    # Check for rank-specific placeholders like {shape_3}
    rank_token = f"{{shape_{rank}}}"
    if rank_token in mlir_text:
        return mlir_text.replace(rank_token, _fmt_shape(shape))

    # Fall back to generic {shape}
    if "{shape}" in mlir_text:
        return mlir_text.replace("{shape}", _fmt_shape(shape))

    # No shape placeholder at all: only keep 4D shapes, leave text as-is.
    if rank != 4:
        return ""
    return mlir_text


@pytest.fixture(params=get_test_cases_from_files(list_mlir_files(TEMPLATE_SUBDIR)))
def template_file(request) -> Path:
    """One parameter per template MLIR file under tests/testdata/template_ops.

    request.param is a Case; .data is the Path.
    """
    return request.param.data


def pytest_generate_tests(metafunc):
    """Dynamically parametrize shape_param based on template_file content.
    
    This hook customizes the parametrization of shape_param for each template file.
    """
    if "shape_param" in metafunc.fixturenames:
        # Generate shapes_by_rank_dtype using module-level defaults
        shapes_dict = {}
        common_dtypes = ['i8', 'i16', 'i32', 'bf16', 'f16', 'f32']
        ranks = [1, 2, 3, 4]
        
        for rank in ranks:
            for dtype in common_dtypes:
                key = (rank, dtype)
                shapes_dict[key] = _generate_shapes_for_rank_and_dtype(
                    rank, dtype, DEFAULT_LRAM_SIZE, DEFAULT_NUM_SAMPLES
                )
        
        # Flatten to all_shape_params
        all_params = []
        for (rank, dtype), shapes in shapes_dict.items():
            for shape in shapes:
                all_params.append((rank, dtype, shape))
        
        # If template_file is also parametrized, we'll get called multiple times
        # Just use all params and let the fixture filter
        ids = [f"r{p[0]}_{p[1]}_" + "x".join(map(str, p[2])) for p in all_params]
        metafunc.parametrize("shape_param", all_params, ids=ids, indirect=True)


@pytest.fixture
def shape_param(request, template_file) -> Tuple[int, str, Sequence[int]]:
    """One parameter per (rank, dtype, shape) combination applicable to the template.
    
    This fixture filters shapes based on the template's placeholders.
    
    Returns: (rank, dtype, shape) tuple
    """
    rank, dtype, shape = request.param
    
    # Check if this combination is applicable to the template
    mlir_text = template_file.read_text()
    placeholders = _extract_shape_dtype_placeholders(mlir_text)
    
    if not placeholders:
        # No specific placeholders, only accept rank 4 with bf16 (backward compat)
        if rank == 4 and dtype == 'bf16':
            return (rank, dtype, shape)
        pytest.skip("No shape placeholders in template")
    
    # Template has specific requirements
    has_match = any(
        (r == rank and (d == dtype or d is None))
        for r, d in placeholders
    )
    
    if not has_match:
        pytest.skip(f"Shape (rank={rank}, dtype={dtype}) not applicable to template")
    
    return (rank, dtype, shape)
    
    return (rank, dtype, shape)


@pytest.fixture
def instantiated_mlir_path(
    tmp_path: Path,
    template_file: Path,
    shape_param: Tuple[int, str, Sequence[int]],
) -> Path:
    """For each (template_file, rank, dtype, shape), generate a concrete MLIR file.

    Combinations that do not match the template's requested rank/dtype/placeholder
    are skipped by the shape_param fixture.
    """

    rank, dtype, shape = shape_param

    mlir_text = template_file.read_text()
    instantiated = _instantiate_template_text(mlir_text, rank, shape, dtype)

    if not instantiated:
        # This (rank, dtype, shape) is not applicable for this template.
        pytest.skip("No MLIR generated for this (template, shape, dtype) combination")

    # Name encodes which template + dtype + shapes were used
    name = (
        f"{template_file.stem}"
        f"_r{rank}_{dtype}_" + "x".join(map(str, shape)) + ".mlir"
    )
    out_path = tmp_path / name
    out_path.write_text(instantiated)
    return out_path

def test_run_templates_on_soc(torq_results, shape_param: Tuple[int, str, Sequence[int]], compiler_args_variant: tuple[str, list[str]], template_file: Path, request: pytest.FixtureRequest):
    """Run compiled VMFB on target hardware (local sim, aws_fpga, or remote SoC via SSH).

    This test leverages the torq_results fixture which handles execution and profiling.
    
    The fixture supports:
    - Remote SoC execution via SSH (runtime_hw_type='astra_machina')
    
    Profiling results are automatically collected by the reporting plugin when
    --template-profiling-enabled is specified.
    
    """
    
    # Record template name for profiling reports
    request.node.user_properties.append(("template_name", template_file.stem))
    
    assert torq_results is not None, "Results not generated"
