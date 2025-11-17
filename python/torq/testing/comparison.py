
import numpy as np
from .iree import TOPDIR

"""

This module provides utilities to compare test results of inference.

Comparisons use some criterias to determine if two outputs are equivalent
tweaked to account for inaccuracies due to floating point and quantization.

"""

def check_nans(arr1, arr2):
    nan1 = np.isnan(arr1)
    nan2 = np.isnan(arr2)
    assert (nan1 == nan2).all(), "Nans differ."
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

    try:
        accept_zero_output = request.getfixturevalue("accept_zero_output")
    except pytest.FixtureLookupError:
        accept_zero_output = False

    compare_results(request, observed_result.data, reference_results.data, accept_zero_output=accept_zero_output)


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
