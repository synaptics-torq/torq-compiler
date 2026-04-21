
from ..models import Measurement, TestGroup


def get_reference_test_durations(session_ids):
    """Finds the metric "total_duration" for the tests in the 'home' group for the given sessions"""

    # Get the 'home' test group
    try:
        home_group = TestGroup.objects.get(name='home')
    except TestGroup.DoesNotExist:
        return {}

    # Get all test cases in the 'home' group
    test_cases = home_group.test_cases.all()

    # Query measurements for these test cases in the given sessions
    measurements = (Measurement.objects
        .filter(
            test_run__test_case__in=test_cases,
            test_run__test_run_batch__test_session_id__in=session_ids,
            metric__name='total_duration'
        )
        .select_related('test_run', 'test_run__test_case', 'metric', 'test_run__test_run_batch__test_session')
    )

    # Organize by session and test case
    result = {}
    for measurement in measurements:
        branch_name = measurement.test_run.test_run_batch.test_session.git_branch
        nodeid = measurement.test_run.test_case.nodeid
        duration_ms = measurement.value / 1_000_000.0

        if branch_name not in result:
            result[branch_name] = {}
        result[branch_name][nodeid] = duration_ms

    return result




