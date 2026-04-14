
from django.db.models import Count, Q
from .models import TestSession, TestRun, Measurement, TestGroup


def get_latest_sessions_stats(criterias: Q, limit=None):
    """Get the latest test session for the branches in the given pattern"""

    # Get the ID of all the latest test session for any branch matching the pattern
    ids = (TestSession.objects
            .filter(criterias)
            .order_by('git_branch', '-timestamp')
            .distinct('git_branch'))

    # Keep only the latest N sessions if a limit is provided
    if limit is not None:
        ids = ids[:limit]

    # Annotate the sessions with the counts of test runs by outcome
    sessions_with_stats = (TestSession.objects
        .filter(id__in=ids)
        .annotate(
            num_total=Count('testrunbatch__testrun'),
            num_passed=Count('testrunbatch__testrun', filter=Q(testrunbatch__testrun__outcome=TestRun.Outcome.PASS)),
            num_failed=Count('testrunbatch__testrun', filter=Q(testrunbatch__testrun__outcome=TestRun.Outcome.FAIL)),
            num_skipped=Count('testrunbatch__testrun', filter=Q(testrunbatch__testrun__outcome=TestRun.Outcome.SKIP)),
            num_error=Count('testrunbatch__testrun', filter=Q(testrunbatch__testrun__outcome=TestRun.Outcome.ERROR)),
            num_xfail=Count('testrunbatch__testrun', filter=Q(testrunbatch__testrun__outcome=TestRun.Outcome.XFAIL)))
        .order_by('-timestamp')
    )

    return sessions_with_stats


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
