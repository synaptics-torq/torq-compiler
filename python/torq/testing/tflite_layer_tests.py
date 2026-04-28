
import pytest
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional
from filelock import FileLock
import tensorflow as tf
import numpy as np


from torq.testing.tensorflow import run_with_tflite

# Import the direct TFLite layer extractor (preserves quantization)
from torq.testing.tflite_layer_extractor import extract_all_layers, TFLiteTensorOutputExporter

from torq.testing.versioned_fixtures import (    
    versioned_unhashable_object_fixture, versioned_hashable_object_fixture, versioned_cached_data_fixture,
    versioned_generated_file_fixture
)


@dataclass
class TFLiteLayerCase:
    """Represents a test case for a TFLite layer or full model."""

    """ Unique name for the test case (used in test IDs) """
    name: str    

    """ The full model path is always included for reference, even for layer cases"""
    full_model_path: Optional[str] = None

    """ The layer path will only be set for layer cases, and may be None if extraction failed """
    layer_model_path:Optional[str] = None

    op_name: Optional[str] = None
    op_index: Optional[int] = None
    is_layer: bool = False
    extraction_failed: bool = False
    is_quantized: bool = False
    input_tensors: List[int] = None

    @property
    def model_path(self) -> Optional[str]:
        """Returns the path to the TFLite model to use for this test case (layer or full model)"""
        return self.layer_model_path if self.is_layer and self.layer_model_path else self.full_model_path        


# ============================================================================
# Hooks
# ============================================================================


def pytest_addoption(parser):
    parser.addoption("--tflite-layer-test-generate-full-model-only", action="store_true", default=False, help="Disable generation of TFLite layer test cases (only run full model tests)")
    parser.addoption("--tflite-layer-test-force-extract", action="store_true", default=False, help="Force re-extraction of TFLite layers even if cached versions are available")
    parser.addoption("--tflite-layer-test-max-layers", type=int, default=0, help="Maximum number of layers to generate test cases for (0 = no limit)")


# ============================================================================
# Test Case Generation
# ============================================================================

# Cache for generated test cases (avoid regenerating on each pytest_generate_tests call)
_CASES_CACHE = {}


def generate_tflite_layer_cases(model_name:str, tflite_path: Path, metafunc) -> List[TFLiteLayerCase]:
    """
    Generate test cases for each layer in a TFLite model.
    
    Uses the direct TFLite layer extractor (Option 2) which preserves 
    quantization from the original model.
    
    Args:
        tflite_path: Path to TFLite model file
        
    Returns:
        List of TFLiteLayerCase objects
    """
    import json as json_mod
    
    cases = []
    
    cache_dir = metafunc.config.cache.mkdir('tflite_layer_cache')

    layers_dir = cache_dir / f"{model_name}_layers"
    layers_dir.mkdir(parents=True, exist_ok=True)

    max_layers = metafunc.config.getoption("--tflite-layer-test-max-layers")
    force_extract = metafunc.config.getoption("--tflite-layer-test-force-extract")

    # when using pytest xdist multiple processes will be trying to run this in parallel
    # we want to make sure the first that gets here computes the stuff and the other ones
    # wait for it to complete and just use the cached data
    with FileLock(str(layers_dir / "lock")):
        
        # Check for cached cases file (fast path - no TF import needed)
        cases_cache_file = layers_dir / "_cases_cache.json"

        if cases_cache_file.exists() and not force_extract:
            try:
                with open(cases_cache_file, 'r') as f:
                    cached_cases = json_mod.load(f)
                # Reconstruct TFLiteLayerCase objects
                result = []
                for c in cached_cases:
                    if max_layers > 0 and len(result) >= max_layers + 1:  # +1 for full model
                        break
                    result.append(TFLiteLayerCase(**c))
                print(f"Loaded {len(result)} cached cases for {model_name} (use --tflite-layer-test-force-extract to regenerate)")
                return result
            except Exception as e:
                print(f"Cache load failed for {model_name}: {e}")

        # Use the new direct layer extractor that preserves quantization
        print(f"Processing {tflite_path.name} with quantization-preserving extractor...")
        
        extraction_results = extract_all_layers(
            str(tflite_path),
            str(layers_dir),
            max_layers=max_layers,
            force=force_extract
        )        

        for result in extraction_results:
            op_name = result['op_name']
            op_index = result['layer_index']
            layer_name = f"{model_name}_layer_{op_name}_{op_index}"
            
            if not result['success']:
                # Create case anyway but mark as failed
                cases.append(TFLiteLayerCase(
                    name=layer_name, 
                    full_model_path=str(tflite_path),
                    op_name=op_name,
                    op_index=op_index,
                    is_layer=True,
                    extraction_failed=True,
                    is_quantized=result.get('is_quantized', False),
                ))
                continue
            
            layer_tflite = Path(result['layer_file'])
            
            
            cases.append(TFLiteLayerCase(name=layer_name, 
                full_model_path=str(tflite_path),
                layer_model_path=str(layer_tflite),
                op_name=op_name,
                op_index=op_index,
                is_layer=True,
                is_quantized=result.get('is_quantized', False),
                input_tensors=[x['index'] for x in result.get('inputs', []) if not x['is_constant']]
            ))
        
        # Add full model case
        cases.append(TFLiteLayerCase(name=f"{model_name}_full_model", 
            full_model_path=str(tflite_path),
            is_layer=False,
        ))
        
        # Save cases cache for fast loading next time
        try:
            with open(cases_cache_file, 'w') as f:
                json_mod.dump([asdict(c) for c in cases], f)
        except Exception:
            pass  # Ignore cache write errors
        
        print(f"  Generated {len(cases)} test cases ({sum(1 for r in extraction_results if r['success'])} layers + 1 full model)")
        return cases


def generate_parametrized_tests(metafunc, model_names, model_paths, marks=lambda case_name: ()):
    """
    Generate paramerized version of the specified test function with
    one test per layer in the TFLite model, plus one for the full model.

    The test function must have a fixture parameter named "tflite_layer_model" which will
    be parametrized with TFLiteLayerCase objects containing the layer/model data.

    Parameters:

    - metafunc: the pytest metafunc object received in pytest_generate_tests
    - model_names: a list of string identifier for the model (used in test case names)
    - model_paths: a list of path to the TFLite model file
    - marks: function that returns marks to apply to a given test name

    """

    # check if the test case depends on the tflite_layer_model
    # fixture, in this case we will parametrize it, otherwise
    # return immediately

    if 'tflite_layer_model' not in metafunc.fixturenames:
        return

    if len(model_names) != len(model_paths):
        raise ValueError("model_name and model_path must have the same length")

    max_layers = metafunc.config.getoption("--tflite-layer-test-max-layers")
    full_only = metafunc.config.getoption("--tflite-layer-test-generate-full-model-only")

    all_cases = []
    for model_name, model_path in zip(model_names, model_paths):

        cache_key = f"{model_name}_max_layers={max_layers}_full={full_only}"
        
        cases = _CASES_CACHE.get(cache_key)

        # compute the test cases that were requested
        if cases is None:        
            
            if full_only:
                cases = [
                    TFLiteLayerCase(
                        name=f"{model_name}_full_model",
                        full_model_path=str(model_path),
                        is_layer=False
                    )
                ]
            else:            
                cases = generate_tflite_layer_cases(model_name, model_path, metafunc)

            _CASES_CACHE[cache_key] = cases

        all_cases.extend(cases)

    # no test case was returned, we can skip parametrization
    if not all_cases:
        return

    # create pytest.param object that carry name and marks
    params = []
    for case in all_cases:
        test_marks = []

        if case.is_layer:
            test_marks.append(pytest.mark.layer)
        else:
            test_marks.append(pytest.mark.full_model)

        test_marks.append(pytest.mark.tflite_layer_test)
        
        test_marks = test_marks + list(marks(case.name))

        params.append(pytest.param(case, marks=test_marks, id=case.name))
    
    metafunc.parametrize("tflite_layer_model", params, indirect=True)


def get_quant_params(model_path):
    """Return (in_scale, in_zp, is_int8, out_scale, out_zp) from a TFLite model."""
    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    in_detail  = interp.get_input_details()[0]
    out_detail = interp.get_output_details()[0]
    in_scale,  in_zp  = in_detail["quantization"]
    out_scale, out_zp = out_detail["quantization"]
    is_int8 = (in_detail["dtype"] == np.int8)
    return float(in_scale), int(in_zp), is_int8, float(out_scale), int(out_zp)


# ============================================================================
# Fixtures
# ============================================================================

@versioned_hashable_object_fixture
def tflite_layer_model(request):
    """Fixture that provides the TFLite model for the current test case."""

    return request.param


@versioned_generated_file_fixture("tflite")
def tflite_layer_inputs_model(request, versioned_file, tflite_layer_model: TFLiteLayerCase):    
    """
    Creates a new TFLite model file that exposes the inputs of the layer under test as subgraph outputs, so we can
    easily extract the reference inputs for the layer tests.
    """
    assert tflite_layer_model.is_layer, "This fixture should only be used for layer test cases"
    extractor = TFLiteTensorOutputExporter(tflite_layer_model.full_model_path)    
    extractor.export(versioned_file, tflite_layer_model.input_tensors)


@pytest.fixture
def full_model_input_data(request, case_config):
    """
    Fixture that provides the input data for the full model test case.

    This will be used by the tflite_layer_inputs fixture to compute the reference inputs 
    for the layer test cases, by running the full model and extracting the relevant tensors.
    For full model test cases, the value is just passed through directly.
    """
    return request.getfixturevalue(case_config['full_model_input_data'])


@versioned_cached_data_fixture
def tflite_layer_inputs(request, tflite_layer_model: TFLiteLayerCase, full_model_input_data):    
    """
    Computes the reference inputs for a TFLite layer test case by running the full model 
    and extracting the relevant tensors.
    """

    if not tflite_layer_model.is_layer:
        return full_model_input_data
    
    tflite_layer_inputs_model = request.getfixturevalue("tflite_layer_inputs_model")
    
    results = run_with_tflite(tflite_layer_inputs_model.file_path, full_model_input_data)

    # return the values for all the inputs
    return [result[1] for result in results]


@versioned_unhashable_object_fixture
def tflite_model_path(request, tflite_layer_model: TFLiteLayerCase):
    """Fixture that provides the model path for the test case"""

    record_property = request.getfixturevalue("record_property")
    record_property("compiler_input", f"tflite:{tflite_layer_model.name}")

    return tflite_layer_model.model_path


