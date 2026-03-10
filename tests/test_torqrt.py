import pytest
import shutil
import subprocess

from pathlib import Path
from torq.testing.cases import Case
from torq.testing.iree import TOPDIR
from torq.testing.versioned_fixtures import versioned_hashable_object_fixture, VersionedUncachedData

from .test_tflite_mbv2 import download_mobilenetv2_model, mbv2_compile_options


"""
Test that exported torq_rt test vectors descriptors are correctly executed by torq_rt
"""


@pytest.fixture
def mbv2_model(request, cache):
    return VersionedUncachedData(data=download_mobilenetv2_model(cache), version="mbv2_model")
    

def get_torqrt_test_cases():

    cases = []

    # add mlir based tests cases
    base_config = {
        "mlir_model_file": "static_mlir_model_file",
        "input_data": "tweaked_random_input_data",
        "comparison_config": "comparison_config_from_mlir",
    }

    TEST_DATA_DIR = TOPDIR / "tests" / "testdata"

    test_files = [
        (TEST_DATA_DIR / "torch_ops" / "softmax-1x8x1x207xbf16.mlir", False),
        (TEST_DATA_DIR / "tosa_ops" / "dw_peeling.mlir", True)        
    ]

    for (test_file, check_results) in test_files:
        case_config = base_config.copy()
        case_config["static_mlir_model_file"] = test_file
        case_config["check_results"] = check_results
        cases.append(Case(test_file.name, case_config))


    # add e2e test cases
    cases.append(Case("mbv2_full_model", {
        "tflite_model_file": "mbv2_model",
        "mlir_model_file": "tflite_mlir_model_file",
        "input_data": "tweaked_random_input_data",
        "torq_compiler_options": mbv2_compile_options(),
        "check_results": False
    }))

    return cases


@pytest.fixture(params=get_torqrt_test_cases())
def case_config(request, runtime_hw_type):

    # run this test only if we use the cmodel
    if runtime_hw_type.data != 'sim':        
        return pytest.skip()
    
    return request.param.data


# override the fixture that enables dumping test vectors to force test vector generation 
@versioned_hashable_object_fixture
def enable_hw_test_vectors(request):
    return True


def get_outputs(ref_dir : Path):
    
    exit_mem_file = ref_dir / "tv.exit.mem.lst"

    if not exit_mem_file.exists():
        return []

    filenames = []
    with open(exit_mem_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                # Last field is the filename
                filenames.append(ref_dir / parts[-1])

    return filenames


def run_rt(torq_rt_cm, test_vector_dir, rt_output_path: Path):

    rt_output_path.mkdir(parents=True)

    shutil.copytree(test_vector_dir, rt_output_path / "tc")

    # remove all the reference bus logs
    for file in (rt_output_path / "tc").glob('*/tv.chk.*.txt'):
        file.unlink()

    # remove all the reference outputs
    for job in (rt_output_path / "ref").glob('job*'):
        for file in get_outputs(job):            
            if file.exists():
                file.unlink()                

    shutil.copy(torq_rt_cm, rt_output_path)
    
    subprocess.check_call([str(rt_output_path / "torq_rt_cm"), '-o', 'ref'], cwd=rt_output_path)


def check_rt_output(test_vector_dir, rt_output_path: Path):

    checks = []
    failed = False

    for job in (rt_output_path / "ref").glob('job*'):

        ref_dir = test_vector_dir / job.name
    
        for chk_file in get_outputs(ref_dir):            

            if not chk_file.exists():
                print(f"Output file {chk_file} does not exist, skipping check")
                continue

            if not (ref_dir / chk_file.name).exists():
                print(f"Reference file {ref_dir / chk_file.name} does not exist, skipping check")
                continue

            try:
                subprocess.check_call(['diff', '-q', ref_dir / chk_file.name, chk_file])
                matches = True
            except subprocess.CalledProcessError:
                matches = False
                failed = True

            checks.append((f"{chk_file.parent.name}/{chk_file.name}", matches))

    print()
    print("Check results:\n")

    fail_msg = '\033[91mFAIL\033[0m'
    ok_msg = '\033[92mOK  \033[0m'

    for check in sorted(checks, key=lambda x: x[0]):
        print(f"  {ok_msg if check[1] else fail_msg} {check[0]}")

    # Check output only if test cases do not use CSS
    assert not failed, f"Some outputs differ"


@pytest.mark.ci
def test_test_vectors(request, case_config, torq_test_vectors, torq_rt_cm, tmpdir):

    for dispatch_dir in torq_test_vectors.data.iterdir():
        for invocation_dir in dispatch_dir.iterdir():
            print(f"Running test vector in {invocation_dir}")

            out_dir = Path(tmpdir) / "rt"

            run_rt(torq_rt_cm.data, invocation_dir, out_dir)

            if case_config["check_results"]:
                check_rt_output(invocation_dir, out_dir)
