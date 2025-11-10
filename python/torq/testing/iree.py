import ml_dtypes # support for bfloat16 dtype
import numpy as np
import os
from pathlib import Path
import re
import subprocess
import shutil
import pytest

from iree.compiler.ir import Context, Module

from .compilation_tests import check_nans, CachedTestData, CompilationTestCase, CachedTestDataFile

import abc


TOPDIR = Path(__file__).parent.parent.parent.parent

BUILD_DIR = Path(os.environ.get('IREE_BUILD_DIR', str(TOPDIR.parent / 'iree-build')))
SOC_BUILD_DIR = Path(os.environ.get('IREE_SOC_BUILD_DIR', str(TOPDIR.parent / 'iree-build-soc')))

def _find_iree_tool(env_var, tool_name):
    """Find IREE tool binary, checking environment variable first, then BUILD_DIR fallback."""

    # Check environment variable first
    env_path = os.getenv(env_var)
    if env_path:
        return env_path
    
    # Fall back to BUILD_DIR based path
    fallback_path = BUILD_DIR / 'third_party/iree/tools' / tool_name
    if fallback_path.exists():
        return str(fallback_path)

    # Check if tool is in the path
    tool_path = shutil.which(tool_name)
    if tool_path:
        return tool_path

    raise FileNotFoundError(f"Could not find {tool_name}.")


IREE_RUN_MODULE = _find_iree_tool('IREE_RUN_MODULE', 'iree-run-module')
IREE_COMPILE = _find_iree_tool('IREE_COMPILE', 'iree-compile')
IREE_OPT = _find_iree_tool('IREE_OPT', 'iree-opt')

MODELS_DIR = TOPDIR / 'tests/testdata/'

def cleanup_testcase(request, mlir_data):
    """
    Canonicalizes the input mlir data to a temporary file and returns the cleaned up mlir data
    """

    tmpdir = request.getfixturevalue("tmpdir")

    input_path = tmpdir / 'dirty.mlir'

    with open(input_path, 'w') as fp:
        fp.write(mlir_data)

    output_path = tmpdir / 'clean.mlir'

    subprocess.check_call([IREE_OPT,
                           '-canonicalize', str(input_path),
                           '-o', str(output_path)])

    with open(output_path, 'r') as fp:
        return fp.read()


def get_debug_ir_params(request):
    """
    Creates the debug parameters for invoking iree-compile based on pytest options
    """

    if request.config.getoption("--debug-ir") is not False:
        if request.config.getoption("--debug-ir") is True:
            tmpdir = request.getfixturevalue("tmpdir")
            dump_path = str(tmpdir / 'ir')
        else:
            dump_path = (Path(request.config.rootdir) / request.config.getoption("--debug-ir"))

        return ['--mlir-print-ir-after-all', f'--mlir-print-ir-tree-dir={dump_path}']

    return []


def get_extra_torq_compiler_options(request):
    """
    Finds extra torq compiler options from pytest options
    """

    cmds = []

    if request.config.getoption("--extra-torq-compiler-options"):
        cmds.extend(request.config.getoption("--extra-torq-compiler-options").split(" "))

    cmds += get_debug_ir_params(request)

    return cmds

def get_input_type_options(input_path):
    """
    Auto-detects the pipeline suitable to run the MLIR input file
    """

    with open(input_path, 'r') as mlir_file:
        mlir_content = mlir_file.read()
    if "torch.onnx" in mlir_content:
        return ['--iree-input-type=onnx-torq']
    elif "torch." in mlir_content:
        return ['--iree-input-type=torch-torq']
    return []

def compile_torq(request, input_path, output_path, ext_options=[]):
    """
    Compiles an mlir file with torq backend
    """

    ext_options = ext_options + get_input_type_options(input_path)

    cmds = [str(TOPDIR) + '/scripts/torq-compile',
            IREE_COMPILE,
            str(input_path), '-o', str(output_path),            
            *ext_options,
            *get_extra_torq_compiler_options(request)]
    
    if not request.config.getoption("--no-phases-dump"):
        cmds.append('--dump-compilation-phases-to=' + str(output_path) + '-phases')

    if request.config.getoption("--generate-hw-test-vectors"):
        cmds.append(f'--torq-dump-descriptors-dir={output_path}-cfgdesc')

    if request.config.getoption("--trace-buffers"):
        cmds.append('--torq-enable-buffer-debug-info')

    gdb_port = request.config.getoption('--debug-torq-compiler')
    if gdb_port > 0:
        cmds = ['gdbserver', 'localhost:' + str(gdb_port)] + cmds

    print("Compiling for TORQ with: " + " ".join(cmds))

    with request.getfixturevalue("scenario_log").event("torq_compile"):
        subprocess.check_call(cmds, cwd=str(Path(output_path).parent))


def create_output_args(request, output_specs, tag):
    """
    Creates the output command line args to invoke iree-run-module
    """
    output_path_root = str(request.getfixturevalue("tmpdir") / f'output_{tag}')

    output_args = []
    output_paths = []

    for idx, tensor_type in enumerate(output_specs):
        output_path = f'{output_path_root}_{idx}.bin'
        output_paths.append(output_path)
        output_args.append(f'--output=@{output_path}')

    return output_args, output_paths


def load_outputs(output_specs, output_paths):
    """
    Reads the data saved as outputs from iree-run-module
    """
    output_data = []

    for idx, tensor_type in enumerate(output_specs): 
        with open(output_paths[idx], 'rb') as f:
            data = np.frombuffer(f.read(), dtype=get_dtype(tensor_type.fmt)
                                 ).reshape(tensor_type.shape)
            output_data.append(data)
            
    return output_data


def run_torq(request, model_path, input_args, output_specs, ext_options=[], tag=''):
    """
    Runs the specified vmfb model using iree-run-module with torq backend / hal driver
    """

    output_args, output_paths = create_output_args(request, output_specs, "torq" + tag)

    cmds = [IREE_RUN_MODULE,
            '--device=torq',
            '--module=' + str(model_path),
            '--function=main',
            *output_args,
            *ext_options,
            *input_args]

    tv_dir = str(request.getfixturevalue("tmpdir") / f'output_torq{tag}') + "-tv"
    buffers_dir = str(request.getfixturevalue("tmpdir") / f'output_torq{tag}') + "-buffers"

    if request.config.getoption("--generate-hw-test-vectors"):
        cmds.append('--torq_desc_data_dir=' + str(model_path) + "-cfgdesc")
        cmds.append('--torq_dump_mem_data_dir=' + tv_dir)

    if request.config.getoption("--trace-buffers"):
        cmds.append('--torq_dump_buffers_dir=' + buffers_dir)

    if request.config.getoption("--extra-torq-runtime-options"):
        cmds.extend(request.config.getoption("--extra-torq-runtime-options").split(" "))

    print("Running for TORQ with: " + " ".join(cmds))

    with request.getfixturevalue("scenario_log").event("torq_run"):
        # FIXME: Depending on the platform and the model this will not be enough (tests with desc dumping are particularly slow).
        subprocess.check_call(cmds, timeout=60 * 15)

    if request.config.getoption("--generate-hw-test-vectors"):
        print(f"Generated test vectors in {tv_dir}")

    if request.config.getoption("--trace-buffers"):
        print("\nBuffer tracing enabled\n")
        print("Buffer trace will be available in: " + buffers_dir + "\n")
        print("To view the buffer trace run:")

        ir_path = ""
        ir_dir = str(model_path) + '-phases'
        if os.path.exists(ir_dir):            
            for irs in os.listdir(ir_dir):
                if irs.endswith('9.executable-targets.mlir'):
                    ir_path = str(Path(ir_dir) / irs)
                    break

        print(f"cd {TOPDIR} && streamlit run apps/buffer_viewer/buffer_viewer.py {buffers_dir} {ir_path}")
        print()
        
    return load_outputs(output_specs, output_paths)


def compile_llvmcpu(request, input_path, output_path, compiler_options=[]):
    """
    Compile a mlir file with llvm-cpu backend
    """

    compiler_options = compiler_options

    cmd = [IREE_COMPILE,
           '--iree-hal-target-backends=llvm-cpu',
           str(input_path),
           '-o', str(output_path),
           *compiler_options]

    print("Compiling for LLVMCPU with: " + " ".join(cmd))

    subprocess.check_call(cmd)


def run_llvmcpu(request, model_path, input_args, output_specs,
                runtime_options=[], tag=''):
    """
    Run a mlir file with local-task (CPU) HAL driver
    """

    output_args, output_paths = create_output_args(request, output_specs, "llvm" + tag)

    cmd = [IREE_RUN_MODULE,
           '--device=local-task',
           '--module=' + str(model_path),
           '--function=main',
           *output_args,
           *input_args,
           *runtime_options]

    print("Running for LLVMCPU with: " + " ".join(cmd))

    subprocess.check_call(cmd)

    return load_outputs(output_specs, output_paths)


def get_dtype(name):
    """
    Returns the numpy dtype corresponding to the given MLIR type name
    """

    dict_types = {'i1': bool,
                  'i8': np.int8,
                  'i16': np.int16,
                  'i32': np.int32,              
                  'ui8': np.uint8,
                  'f16': np.float16,
                  'f32': np.float32,
                  'si16': np.int16,
                  'si64': np.int64,
                  'si32': np.int32,
                  'bf16': ml_dtypes.bfloat16
                  }
    
    if name in dict_types:
        return dict_types[name]
        
    raise ValueError(f"Unsupported dtype {name}")


def is_float_type(dtype):
    """
    Returns true if the given dtype is a floating point type (either numpy native or bfloat16)
    """
    return np.issubdtype(dtype, np.floating) or dtype == ml_dtypes.bfloat16


class TensorType:
    """
    Representes the type of a tensor input or output of an MLIR model
    """

    def __init__(self, shape, fmt):
        self.shape = shape
        self.fmt = fmt

    def __str__(self):
        return f'tensor<{self.shape}x{self.fmt}>'
    
    def to_arg(self):
        return "x".join([str(x) for x in self.shape] + [self.fmt])

    @staticmethod
    def from_string(spec):
        *shape_str, fmt = spec.split('x')
        shape = [int(s) for s in shape_str]
        return TensorType(shape, fmt)

    def __repr__(self):
        return f'TensorType(shape={self.shape}, fmt={self.fmt})'


def get_io_specs(test_file):
    """
    Parses the input and output tensor types from the given mlir test file
    """

    with open(test_file, 'r') as mlir_file:
        mlir_content = mlir_file.read()

    module = Module.parse(mlir_content, Context())
    for op in module.body.operations:
        input_types = []
        output_types = []

        for region in op.regions:
            for block in region.blocks:
                for arg in block.arguments:
                    input_types.append(str(arg.type))

                for inner_op in block.operations:
                    if inner_op.name == "func.return":
                        for i, operand in enumerate(inner_op.operands):
                            output_types.append(str(operand.type))

    input_specs = []
    output_specs = []

    def get_torch_specs(input):
        match = re.match(r"!torch\.vtensor<\[(.*)\],(\w+)>", input)

        if match and match.group(1) == '':
            shape = []
        else:
            shape = [int(x) for x in match.group(1).split(',')] if match else None

        if match:
            dtype = match.group(2)
            return TensorType(shape, dtype)
        else:
            return None

    def get_tosa_specs(input):
        match = re.match(r"tensor<([^>]+)>", input)
        if match:
            return TensorType.from_string(match.group(1))
        else:
            return None

    for t in input_types:
        input_specs.append(get_torch_specs(t))
    for t in output_types:
        output_specs.append(get_torch_specs(t))

    if None in input_specs or None in output_specs:
        input_specs = []
        output_specs = []

        for t in input_types:
            input_specs.append(get_tosa_specs(t))
        for t in output_types:
            output_specs.append(get_tosa_specs(t))

    return input_specs, output_specs


def save_test_data(request, input_specs, input_data):
    """
    Save the received test data for inference
    """

    tmpdir = request.getfixturevalue("tmpdir")        

    input_args = []
    for i, tensor_type in enumerate(input_specs):
        file_name = tmpdir / f'in_rnd_{i}.bin'
        
        with open(file_name, 'wb') as f:
            f.write(input_data[i].tobytes())

        np.save(str(file_name) + '.npy', input_data[i])

        input_args.append(f'--input={tensor_type.to_arg()}=@{file_name}')

    return input_args


def compare_results(request, observed_outputs, expected_outputs, accept_zero_output):
    """
    Compare two tensors container in two numpy.array
    """

    tmpdir = request.getfixturevalue("tmpdir")

    observed_output_path = tmpdir / 'output_observed.npy'
    expected_output_path = tmpdir / 'output_expected.npy'

    assert len(observed_outputs) == len(expected_outputs), \
        f"Number of outputs differ: {len(observed_outputs)} vs {len(expected_outputs)}"

    for observed_output, expected_output in zip(observed_outputs, expected_outputs):

        assert observed_output.size == expected_output.size

        assert accept_zero_output or not np.all(observed_output == 0), "Output is 0 always. Sometimes changing parameters will fix it"

        actual_observed_output = observed_output
        actual_expected_output = expected_output
        observed_output, expected_output = check_nans(observed_output, expected_output)

        if (np.issubdtype(expected_output.dtype, bool)):
            # abs_diff means the number of differences when dypte is boolean
            abs_diff = differences = np.sum(expected_output != observed_output)
        else:
            abs_diff = np.abs(expected_output-observed_output).astype(np.float32)
            if (np.issubdtype(expected_output.dtype, np.integer)):
                differences = abs_diff>1
            else:
                # Compute relative difference for each element
                t_expected = expected_output
                t_observed = observed_output
                epsilon = 1e-6  # Small constant to avoid division by zero
                t_diff = np.abs((t_expected - t_observed) / (np.abs(t_expected) + np.abs(t_observed) + epsilon))
                print("Max relative difference: ", np.max(t_diff))

                # Consider error if relative error > 1%
                differences = t_diff > 1e-2

            abs_diff = differences*abs_diff

        num_diffs = np.sum(differences)
        difference_summary = f"Number of differences: {num_diffs} out of {observed_output.size} [{num_diffs / observed_output.size * 100:.2f}%]"

        print(f"Max absolute difference: {np.max(abs_diff)}")
        print(difference_summary)
        print("To display the difference between expected and observed tensor run:")
        print(f"{TOPDIR}/scripts/diff-tensor.py {observed_output_path} {expected_output_path}")
        print("or")
        print(f"cd {TOPDIR} && streamlit run apps/buffer_diff/buffer_diff.py {observed_output_path} {expected_output_path}")

        np.save(str(observed_output_path), actual_observed_output)
        np.save(str(expected_output_path), actual_expected_output)

        if (np.issubdtype(expected_output.dtype, np.integer)):
            assert (np.max(abs_diff) <= 1), difference_summary
        else:
            assert (np.max(differences) <= 1e-2), difference_summary


class WithTorqCompiler:

    @property
    def compiler_options(self):
        return []

    def compile(self) -> Path:

        tmp_dir = self.request.getfixturevalue("tmpdir")
        compiled_model_path = Path(tmp_dir / 'torq_model.vmfb')

        print("[compiling]", compiled_model_path)

        compile_torq(self.request, self.mlir_model_file, compiled_model_path, self.compiler_options)

        return compiled_model_path
        

class WithTorqRuntime:
    """
    Mixin for the CompilationTestCase to execute the compiled model with the torq runtime
    """

    @property
    def runtime_options(self):
        return []

    def execute(self, compiled_model_file, input_data):

        input_specs, output_specs = get_io_specs(self.mlir_model_file)
        input_args = save_test_data(self.request, input_specs, input_data)

        print("[executing]", compiled_model_file)

        return run_torq(self.request, compiled_model_file, input_args, output_specs, self.runtime_options)


class WithLLVMCPUReference:
    """
    Mixin for the CompilationTestCase to compute reference results with LLVMCPU backend
    """

    def generate_reference_results(self):                

        print("Generating LLVMCPU model...")

        reference_model_file = self.cache_dir / "llvmcpu_model.vmfb"
        compile_llvmcpu(self.request, self.mlir_model_file, reference_model_file)

        print("Generating LLVMCPU reference results...")

        input_specs, output_specs = get_io_specs(self.mlir_model_file)
        input_args = save_test_data(self.request, input_specs, self.input_data)

        return run_llvmcpu(self.request, reference_model_file, input_args, output_specs)        


class WithSimpleComparisonToReference(abc.ABC):
    """
    Mixin for the CompilationTestCase to compare results to reference results using a simple comparison
    that contains some heuristics to avoid common pitfalls
    """

    @abc.abstractmethod
    def generate_reference_results(self):
        pass

    reference_results = CachedTestData("reference_results.npy", "generate_reference_results")

    def check_results(self, results):
        return compare_results(self.request, results, self.reference_results, True)


class WithRandomUniformIntegerInputData:
    """
    Mixin for the CompilationTestCase to generate random uniform integer input data for the specified MLIR model
    """

    def generate_input_data(self):

        input_spec, _ = get_io_specs(self.mlir_model_file)

        result = []

        rng = np.random.default_rng(1234)

        for inp_spec in input_spec:
            dtype = get_dtype(inp_spec.fmt)

            if not np.issubdtype(dtype, np.integer):
                raise ValueError("Requested random uniform integer data for float type")

            data = rng.integers(np.iinfo(dtype).min, np.iinfo(dtype).max, size=inp_spec.shape, dtype=dtype)

            result.append(data)

        return result
    

class WithTweakedRandomDataInput:
    """
    Mixin for the CompilationTestCase to generate random input data for the specified MLIR model
    using some heuristics to avoid common pitfalls
    """

    def generate_input_data(self):

        rng = np.random.default_rng(1234)

        input_specs, _ = get_io_specs(self.mlir_model_file)
            
        input_tensors = []

        for tensor_type in input_specs:

            dtype = get_dtype(tensor_type.fmt)

            # TODO: there are many issues when we enable the full range
            # for now we limit the range to avoid issues
            if dtype == np.uint8:
                random_range = (0, 80)
            else:
                random_range = (-40, 40)

            if is_float_type(dtype):
                tensor = rng.uniform(random_range[0],
                                      random_range[1], tensor_type.shape).astype(dtype)
            elif np.issubdtype(dtype, np.integer):
                tensor = rng.integers(random_range[0], random_range[1],
                                      tensor_type.shape, dtype=dtype)
            elif dtype is bool:
                tensor = rng.integers(0, 2, tensor_type.shape, dtype=dtype)
            else:
                raise ValueError(f"Unsupported dtype {dtype}")

            input_tensors.append(tensor)

        return input_tensors


class WithCachedMlirModel(abc.ABC):
    """
    Mixin for the CompilationTestCase to provide a cached MLIR model
    """

    mlir_model_file = CachedTestDataFile("model.mlir", "generate_mlir_model")
    
    @abc.abstractmethod
    def generate_mlir_model(self, mlir_file_path):
        pass


class MlirTestCase(WithTorqCompiler, WithTorqRuntime, WithSimpleComparisonToReference, CompilationTestCase):
    """
    Subclass of CompilationTestCase for test cases that use an hardcoded MLIR model file
    """
    pass


def list_mlir_files(dir_name):
    """
    Creates pytest parameters for all mlir files in the specified testdata subdirectory    
    """
    
    test_files = []

    testdata_dir = TOPDIR / 'tests' / 'testdata' / dir_name

    for file in os.listdir(testdata_dir):
        if file.endswith('.mlir') and not file.startswith('disable-'):
            test_files.append(pytest.param(str(testdata_dir / file), id=file[:-len(".mlir")]))

    return sorted(test_files)