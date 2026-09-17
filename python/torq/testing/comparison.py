
import numpy as np
from pathlib import Path

from torq.lab.verification.compare import DEFAULT_COMPARISON_CONFIG, compare_outputs

"""

This module provides utilities to compare test results of inference.

Comparisons use some criterias to determine if two outputs are equivalent
tweaked to account for inaccuracies due to floating point and quantization.

The numeric core lives in ``torq.lab.verification.compare.compare_outputs``; this module is
the pytest shell around it: it resolves the per-case config from fixtures, saves
the tensors under ``tmpdir`` for the diff tooling, prints the per-tensor metrics
(the gen_config accuracy pipeline parses these stdout lines), and asserts.

"""

TOPDIR = Path(__file__).parent.parent.parent.parent


def compare_test_results(request, observed_result, reference_results, case_config):

    comparison_config = dict(DEFAULT_COMPARISON_CONFIG)

    if 'comparison_config' in case_config:
        configuration_overrides = request.getfixturevalue(case_config['comparison_config'])

        configuration_overrides = getattr(configuration_overrides, 'data', configuration_overrides)

        comparison_config.update(configuration_overrides)

    compare_results(request, observed_result.data, reference_results.data, comparison_config=comparison_config)


def compare_results(request, observed_outputs, expected_outputs, comparison_config):
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

        print("To display the difference between expected and observed tensor run:")
        print(f"{TOPDIR}/scripts/diff-tensor.py {observed_output_path} {expected_output_path}")
        print("or")
        print(f"cd {TOPDIR} && streamlit run webapps/buffer_diff/buffer_diff.py {observed_output_path} {expected_output_path}")
        np.save(str(observed_output_path), observed_output)
        np.save(str(expected_output_path), expected_output)

    result = compare_outputs(observed_outputs, expected_outputs, config=comparison_config)

    for tensor in result.tensors:
        if tensor.max_rel_diff is not None:
            print(f'Max relative difference: {tensor.max_rel_diff}')
        print(f"Max absolute difference: {tensor.max_abs_diff}")
        pct = (tensor.num_diffs / tensor.size * 100) if tensor.size else 0.0
        print(f"Number of differences: {tensor.num_diffs} out of {tensor.size} [{pct:.2f}%]")

    assert result.passed, result.reason
