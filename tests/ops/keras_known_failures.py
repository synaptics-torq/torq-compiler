"""
Keras test cases that cannot run at all: running them crashes the test process.

A case that fails (wrong results, compile errors) is not skipped: it goes to
test_keras_ops-xfails.txt with an issue. Unlike a skip, an xfail runs with --try-xfails (the CI
status report of the generate-test-status-report label), which shows when it passes (XPASS).
"""

# Test cases that should be completely skipped (not run at all): {test name: reason}
SKIP_TESTS = {
    # The TFLite interpreter aborts in allocate_tensors (int16 SUB with a scalar first operand),
    # so there is no reference to compare with.
    "model154_sub_s2v_1x6x8x4_int16": "TFLite interpreter aborts in allocate_tensors",
}


def should_skip(test_name):
    """
    Determine if a test should be skipped entirely.
    
    Args:
        test_name: Name of the test case
        
    Returns:
        Tuple of (should_skip: bool, reason: str)
    """
    # Check explicit skip list first
    if test_name in SKIP_TESTS:
        return True, SKIP_TESTS[test_name]
    
    return False, None
