from ..models import TestSession, TestRun, TestRunBatch
from django.db.models import Count, Q

from .test_session_comparison import get_duration_changes_query
from .utils import get_histogram, get_average

from django.db import connection


def get_main_branch_test_counts_over_time(limit=None):
    """Get total and xfail counts over time for sessions on the main branch."""

    sessions = (
        TestSession.objects
        .filter(git_branch='refs/heads/main')
        .annotate(
            num_total=Count(
                'testrunbatch__testrun',
                filter=~Q(testrunbatch__testrun__outcome=TestRun.Outcome.SKIP),
            ),
            num_xfail=Count(
                'testrunbatch__testrun',
                filter=Q(testrunbatch__testrun__outcome=TestRun.Outcome.XFAIL),
            ),
        )
        .order_by('-timestamp', '-id')
    )

    if limit is not None:
        sessions = sessions[:limit]

    return list(reversed(sessions))


def get_latest_sessions_stats(criterias: Q, limit=None):
    """
    Get the latest test session for the branches in the given pattern
    """

    # Get the latest test session per branch (DISTINCT ON requires ordering by branch first)
    latest_per_branch = (TestSession.objects
            .filter(criterias)
            .order_by('git_branch', '-timestamp')
            .distinct('git_branch'))

    # Re-order by recency and keep only the N most recently active branches
    ids = (TestSession.objects
            .filter(id__in=latest_per_branch)
            .order_by('-timestamp'))

    if limit is not None:
        ids = ids[:limit]

    # Annotate the sessions with the counts of test runs by outcome
    sessions_with_stats = (TestSession.objects
        .filter(id__in=ids)
        .annotate(
            num_total=Count('testrunbatch__testrun', filter=~Q(testrunbatch__testrun__outcome=TestRun.Outcome.SKIP)),
            num_passed=Count('testrunbatch__testrun', filter=Q(testrunbatch__testrun__outcome=TestRun.Outcome.PASS)),
            num_failed=Count('testrunbatch__testrun', filter=Q(testrunbatch__testrun__outcome=TestRun.Outcome.FAIL)),
            num_error=Count('testrunbatch__testrun', filter=Q(testrunbatch__testrun__outcome=TestRun.Outcome.ERROR)),
            num_xfail=Count('testrunbatch__testrun', filter=Q(testrunbatch__testrun__outcome=TestRun.Outcome.XFAIL)),
            num_nxpass=Count('testrunbatch__testrun', filter=Q(testrunbatch__testrun__outcome=TestRun.Outcome.NXPASS))
        )
        .order_by('-timestamp')
    )

    return sessions_with_stats


def get_duration_histogram(session, other_session, min_duration_ns):
        
    sql, fields = get_duration_changes_query()

    fields['current_session_id'] = session.id
    fields['baseline_session_id'] = other_session.id
    fields['min_duration'] = min_duration_ns

    return get_histogram(sql, 'change_percent', fields, num_bins=10, min_value=-10.0, max_value=10.0)


def get_top_duration_changes(session, other_session, limit, desc, min_duration_ns):

    sub_sql, fields = get_duration_changes_query()

    fields['current_session_id'] = session.id
    fields['baseline_session_id'] = other_session.id
    fields['min_duration'] = min_duration_ns

    order = "DESC" if desc else "ASC"

    sql = f" SELECT * FROM ({sub_sql}) AS subq ORDER BY subq.change_percent {order} LIMIT {limit}"

    with connection.cursor() as cursor:
        cursor.execute(sql, fields)
        columns = [col[0] for col in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]

    return rows


def get_average_duration_change_percent(session, other_session, min_duration_ns):

    sql, fields = get_duration_changes_query()

    fields['current_session_id'] = session.id
    fields['baseline_session_id'] = other_session.id
    fields['min_duration'] = min_duration_ns

    return get_average(sql, 'change_percent', fields)



def get_outcomes_statistics_comparison(session, baseline_session=None):
    """
    Get the count of each outcome for the given session and the baseline session, grouped by module, and the difference between them.
    """

    test_sessions = [session]
    
    if baseline_session:
        test_sessions.append(baseline_session)

    outcome_counts = (TestRun.objects
        .filter(test_run_batch__test_session__in=test_sessions)
        .exclude(outcome=TestRun.Outcome.SKIP) # we don't count skipped tests because each test run will skip the same
        .values('test_case__module', 'outcome', 'test_run_batch__test_session')
        .annotate(count=Count('id'))
    )

    result = {}
    
    for entry in outcome_counts:
        module = entry['test_case__module']

        outcome = TestRun.Outcome(entry['outcome']).name.lower()
        session_id = entry['test_run_batch__test_session']
        count = entry['count']

        if module not in result:
            empty_stats = {'current': 0, 'baseline': 0}
            result[module] = {outcome.name.lower(): empty_stats.copy() for outcome in TestRun.Outcome}

        if session_id == session.id:
            result[module][outcome]['current'] = count

        if baseline_session and session_id == baseline_session.id:
            result[module][outcome]['baseline'] = count            
    
    # compute the totals
    for module in result.values():
        totals = {'current': 0, 'baseline': 0}
        for outcome in module:            
            totals['current'] += module[outcome]['current']
            totals['baseline'] += module[outcome]['baseline']

        module['total'] = totals

    # compute the difference between the current session and the baseline session
    for module in result:
        for outcome in result[module]:
            current_count = result[module][outcome]['current']
            baseline_count = result[module][outcome]['baseline']
            
            difference = current_count - baseline_count

            result[module][outcome]['difference'] = difference

    # sort the modules by name
    result = dict(sorted(result.items(), key=lambda item: item[0]))

    # add totals per outcome across all modules
    totals = {}
    for module in result:
        for outcome in result[module]:            
            if outcome not in totals:
                totals[outcome] = {'current': 0, 'baseline': 0, 'difference': 0}
            totals[outcome]['current'] += result[module][outcome]['current']
            totals[outcome]['baseline'] += result[module][outcome]['baseline']
            totals[outcome]['difference'] += result[module][outcome]['difference']

    return {'totals': totals, 'per_module': result}


def get_session_summary_statistics(session, baseline_session, min_duration_ns):
    """
    Get a summary of the test session, including the count of each outcome and the histogram of duration changes.
    """

    summary = {}
    
    summary["outcomes"] = get_outcomes_statistics_comparison(session, baseline_session)

    if baseline_session:
        summary["duration_change_histogram"] = get_duration_histogram(session, baseline_session, min_duration_ns)
        summary['top_slower_tests'] = get_top_duration_changes(session, baseline_session, 5, True, min_duration_ns)
        summary['top_faster_tests'] = get_top_duration_changes(session, baseline_session, 5, False, min_duration_ns)               
        summary['average_duration_change_percent'] = get_average_duration_change_percent(session, baseline_session, min_duration_ns)
    else:
        summary["duration_change_histogram"] = None
        summary['top_slower_tests'] = []
        summary['top_faster_tests'] = []
        summary['average_duration_change_percent'] = None
    
    summary['batches'] = (TestRunBatch.objects
                            .filter(test_session=session)
                            .annotate(num_tests=Count('testrun'))
                            .order_by('created_at'))

    return summary