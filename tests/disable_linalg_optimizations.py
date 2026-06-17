import subprocess

from helpers.iree import (MODELS_DIR, IREE_OPT, get_io_specs, eval_llvmcpu, generate_test_data, compare_results, parse_tol)

def test_tanh(request, tmpdir):
    compare_rewriting(request,
                      tmpdir,
                      MODELS_DIR / "linalg_ops/tanh-bf16.mlir",
                      "BfloatTanhPattern")

def test_softmax(request, tmpdir):
    compare_rewriting(request,
                      tmpdir,
                      MODELS_DIR / "linalg_ops/softmax-bf16.mlir",
                      "BfloatSoftmaxPattern")

def test_reciprocal(request, tmpdir):
    compare_rewriting(request,
                      tmpdir,
                      MODELS_DIR / "linalg_ops/reciprocal-bf16.mlir",
                      "BfloatReciprocalPattern")

def test_div(request, tmpdir):
    compare_rewriting(request,
                      tmpdir,
                      MODELS_DIR / "linalg_ops/divf-bf16.mlir",
                      "BfloatDivfPattern")

def test_erf(request, tmpdir):
    compare_rewriting(request,
                      tmpdir,
                      MODELS_DIR / "linalg_ops/erf-bf16.mlir",
                      "BfloatErfPattern")

def test_rsqrt(request, tmpdir):
    compare_rewriting(request,
                      tmpdir,
                      MODELS_DIR / "linalg_ops/rsqrt-bf16.mlir",
                      "BfloatRsqrtPattern")

def compare_rewriting(request, tmpdir, orig_case, pass_name):

    opti_case = tmpdir / orig_case.name
    cmd = [IREE_OPT,
           f'--pass-pipeline=builtin.module(func.func(torq-linalg-to-torqhl-pre-conversion{{enable-patterns={pass_name}}}))',
           '--debug-only=dialect-conversion',
           orig_case,
           '-o',
           opti_case]

    print('Optimizing with:', ' '.join(str(c) for c in cmd))

    subprocess.check_call(cmd)

    input_specs, output_specs = get_io_specs(orig_case)
    input_file_name = generate_test_data(request, input_specs)

    output_orig = eval_llvmcpu(request, orig_case, input_file_name, output_specs)
    output_opti = eval_llvmcpu(request, opti_case, input_file_name, output_specs)

    compare_results(request, output_orig, output_opti, parse_tol(orig_case))
