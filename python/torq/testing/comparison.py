
import numpy as np
from .iree import TOPDIR
from torq.testing.versioned_fixtures import VersionedData

"""

This module provides utilities to compare test results of inference.

Comparisons use some criterias to determine if two outputs are equivalent
tweaked to account for inaccuracies due to floating point and quantization.

"""

def check_nans(arr1, arr2):
    nan1 = np.isnan(arr1)
    nan2 = np.isnan(arr2)
    if not (nan1 == nan2).all():
        if nan2.any() and not nan2.all():
            print(f"Nan positions (expected): {nan2}")
        if nan1.any() and not nan1.all():
            print(f"Nan positions (observed): {nan1}")
        if not nan2.any():
            print(f"No Nan expected")

        assert False, "Nans differ."

    arr1 = arr1.copy()
    arr2 = arr2.copy()
    # Replace NaNs with 0 so that we don't break comparison
    # Skip if no NaNs to avoid errors with integer arrays
    if nan1.any():
        arr1[nan1] = 0
    if nan2.any():
        arr2[nan2] = 0
    return arr1, arr2


def compare_test_results(request, observed_result, reference_results, case_config):

    comparison_config = {"int_tol": 1,
                        "int_thld": 1,
                        "fp_avg_tol": 1e-2,
                        "fp_max_tol": 1e-2,
                        "allow_all_zero": False,
                        "epsilon": 1e-6 }

    if 'comparison_config' in case_config:
        configuration_overrides = request.getfixturevalue(case_config['comparison_config'])

        if isinstance(configuration_overrides, VersionedData):
            configuration_overrides = configuration_overrides.data

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

        actual_observed_output = observed_output
        actual_expected_output = expected_output
        print("To display the difference between expected and observed tensor run:")
        print(f"{TOPDIR}/scripts/diff-tensor.py {observed_output_path} {expected_output_path}")
        print("or")
        print(f"cd {TOPDIR} && streamlit run webapps/buffer_diff/buffer_diff.py {observed_output_path} {expected_output_path}")
        np.save(str(observed_output_path), actual_observed_output)
        np.save(str(expected_output_path), actual_expected_output)
        observed_output, expected_output = check_nans(observed_output, expected_output)

        # Guard against accidentally returning an all-zero tensor when we expect meaningful data.
        # If the reference is also all-zero (by value), allow an all-zero observed output.
        if not comparison_config["allow_all_zero"]:
            expected_is_all_zero = np.all(expected_output == 0)
            if not expected_is_all_zero:
                assert np.any(observed_output != 0), "Output is 0 always"

        if (np.issubdtype(expected_output.dtype, bool)):
            # abs_diff means the number of differences when dypte is boolean
            abs_diff = differences = np.sum(expected_output != observed_output)
        else:
            abs_diff = np.abs(expected_output.astype(np.float32)-observed_output.astype(np.float32))
            if (np.issubdtype(expected_output.dtype, np.integer)):
                differences = abs_diff > comparison_config['int_tol']
            else:
                scale = np.abs(expected_output) + np.abs(observed_output) + comparison_config['epsilon']
                rel_diff = abs_diff / scale
                # force 0/0 -> 0
                # DM: avoid TypeError: 'numpy.float32' object does not support item assignment
                # rel_diff[abs_diff == 0] = 0
                differences = rel_diff > comparison_config['fp_avg_tol']
                print(f'Max relative difference: {np.max(rel_diff)}')
            abs_diff = differences*abs_diff

        num_diffs = np.sum(differences)
        difference_summary = f"Number of differences: {num_diffs} out of {observed_output.size} [{num_diffs / observed_output.size * 100:.2f}%]"

        print(f"Max absolute difference: {np.max(abs_diff)}")
        print(difference_summary)

        if (np.issubdtype(expected_output.dtype, np.integer) or np.issubdtype(expected_output.dtype, bool)):
            assert (np.max(abs_diff) <= comparison_config['int_thld']) and not (abs_diff != 0).sum(), difference_summary
        else:
            assert np.max(rel_diff) <= comparison_config['fp_max_tol'], difference_summary
