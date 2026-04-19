
from .models import TestSession, TestRun, Measurement, TestGroup

from django.db.models import (
    Count, F, FloatField, Max, Avg,
    ExpressionWrapper, Value, Q, TextField
)
from django.db.models.functions import Concat
from django.db import connection

import json


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


def get_default_comparison_session(session_id):

    # find the last two session on the main branch
    main_sessions = TestSession.objects.filter(git_branch='refs/heads/main').order_by('git_branch', '-timestamp')[:2]

    # there are no session on the main branch, return None
    if len(main_sessions) == 0:
        return None

    # if the current session is the latest on main, return the previous one, otherwise return the latest one
    if main_sessions[0].id == session_id:
        if len(main_sessions) > 1:
            return main_sessions[1]
        else:
            return None
    else:
        return main_sessions[0]


def get_metric_change_percent_qs(metric_name, session, other_session):
    """
    Get a queryset with the % change of the given metric for each test case between the current session and the other session.

    Only tests that pass are included in the comparison, even if the metric is available.

    The resulting query set has the following fields:

    - test_run__test_case_id: the id of the test case
    - test_run__test_case__module: the module of the test case
    - test_run__test_case__name: the name of the test case
    - test_run__test_case__parameters: the parameters of the test case
    - current_value: the value of the metric for the test case in the current session
    - other_value: the value of the metric for the test case in the other session
    - change_percent: the % change of the metric between the current session and the other session    
    
    """
    
    if other_session is None:
        return (Measurement.objects.filter(
            metric__name=metric_name,
            test_run__test_run_batch__test_session_id=session,
        ).values('test_run__test_case_id')
        .annotate(
            current_value=Max("value"),
            other_value=Value(None, output_field=FloatField()),
            change_percent=Value(None, output_field=FloatField()))
        )

    change_percent_qs = (Measurement.objects
        .filter(
            metric__name=metric_name,
            test_run__test_run_batch__test_session_id__in=[session, other_session],
        )
        .values('test_run__test_case_id')
        .annotate(
            current_value=Max(
                "value",
                filter=Q(test_run__test_run_batch__test_session_id=session),
            ),
            other_value=Max(
                "value",
                filter=Q(test_run__test_run_batch__test_session_id=other_session),
            ),
            current_outcome=Max(
                "test_run__outcome",
                filter=Q(test_run__test_run_batch__test_session_id=session),
            ),
            other_outcome=Max(
                "test_run__outcome",
                filter=Q(test_run__test_run_batch__test_session_id=other_session),
            ),
        )
        .annotate(
            change_percent=ExpressionWrapper(
                (F("current_value") - F("other_value")) * Value(100.0) / F("other_value"),
                output_field=FloatField(),
            ),
        )
    )
    
    return change_percent_qs


def get_histogram(qs, field, num_bins, min_value, max_value):
    """
    Compute a histogram of the given field for the given queryset using PostgreSQL's width_bucket function.
    """
    
    # Build histogram from the queryset SQL as a derived table:
    # SELECT width_bucket(subq.<field>, ...) ... FROM (<qs SQL>) AS subq
    inner_qs = qs.values(field)
    inner_sql, inner_params = inner_qs.query.sql_with_params()
    histogram_sql = f"""
        SELECT
            width_bucket(subq.\"{field}\"::double precision, %s, %s, %s) AS bucket,
            COUNT(*) AS count
        FROM ({inner_sql}) AS subq
        GROUP BY bucket
        ORDER BY bucket
    """

    # we use inverted bins to make sure 0 ends up in the negative bin (which is the "good bin")
    with connection.cursor() as cursor:
        cursor.execute(histogram_sql, [float(max_value), float(min_value), int(num_bins), *inner_params])
        rows = cursor.fetchall()

    count_per_bucket = {row[0]: row[1] for row in rows}

    entries = []

    for bucket in range(num_bins + 2):

        # we want to display the buckets in increasing order
        # so we need to invert the bucket number returned by
        # width_bucket
        inverted_bucket = num_bins + 1 - bucket

        if bucket == 0:
            # underflow bucket
            entries.append({
                "lower_bound": None,
                "upper_bound": min_value,
                "bucket": bucket,
                "count": count_per_bucket.get(inverted_bucket, 0),
                "label": f"≤ {min_value}%"
            })
        elif bucket == num_bins + 1:
            # overflow bucket
            entries.append({
                "lower_bound": max_value,
                "upper_bound": None,
                "bucket": bucket,
                "count": count_per_bucket.get(inverted_bucket, 0),
                "label": f">{max_value}%"
            })
        else:
            # regular bucket
            lower_bound = min_value + (max_value - min_value) * (bucket - 1) / num_bins
            upper_bound = min_value + (max_value - min_value) * bucket / num_bins
            entries.append({
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "bucket": bucket,
                "count": count_per_bucket.get(inverted_bucket, 0),
                "label": f"( {lower_bound:.1f}% , {upper_bound:.1f}% ]"
            })

    return entries

def get_average(qs, field):
    """
    Compute the average of the given field for the given queryset.
    """
    
    average_qs = qs.aggregate(average=Avg(field))
    return average_qs['average']


def get_session_summary_statistics(session, other_session):
    """
    Get a summary of the test session, including the count of each outcome and the histogram of duration changes.
    """

    summary = {"outcome_counts": {}, 
               "duration_change_histogram": None, 
               "total_tests": {
                   "current": 0,
                   "other": None,
                   "difference": None},
               "top_slower_tests": None,
               "top_faster_tests": None}
    
    summary["total_tests"]["current"] = session.test_runs.count()

    for outcome in TestRun.Outcome:
        summary["outcome_counts"][outcome] = {'current': session.test_runs.filter(outcome=outcome).count(), 
                                                    'other': None,
                                                    'difference': None}

    if other_session:
        summary["total_tests"]["other"] = other_session.test_runs.count()
        summary["total_tests"]["difference"] = summary["total_tests"]["current"] - summary["total_tests"]["other"]

        for outcome in TestRun.Outcome:
            summary["outcome_counts"][outcome]['other'] = other_session.test_runs.filter(outcome=outcome).count()
            summary["outcome_counts"][outcome]['difference'] = summary["outcome_counts"][outcome]['current'] - summary["outcome_counts"][outcome]['other']
        
        # get a query set with the % change of the total_duration metric for each test case between the current session and the other session
        # the results is stored in the "change_percent" annotation of the query set
        duration_change_qs = get_metric_change_percent_qs('total_duration', session, other_session)

        # we only consider "valid" tests for the statics
        duration_change_qs = (duration_change_qs
            .exclude(current_value__isnull=True)
            .exclude(other_value__isnull=True)
            .exclude(other_value=0.0)
            .exclude(current_value__lt=1000*1000)            
            .filter(current_outcome=TestRun.Outcome.PASS, other_outcome=TestRun.Outcome.PASS)
        )

        # return the histogram of the % change for all test cases in the session        
        summary['duration_change_histogram'] = json.dumps(get_histogram(duration_change_qs, 'change_percent', 10, -10.0, 10.0))

        duration_with_details_qs = (
            duration_change_qs
            .annotate(module=F('test_run__test_case__module'), 
                      name=F('test_run__test_case__name'), 
                      parameters=F('test_run__test_case__parameters'))
        )

        # get the top 5 slower tests        
        summary['top_slower_tests'] = duration_with_details_qs.order_by('-change_percent')[:5]

        # get the top 5 faster tests
        summary['top_faster_tests'] = duration_with_details_qs.order_by('change_percent')[:5]

        summary['average_duration_change_percent'] = get_average(duration_change_qs, 'change_percent')
        summary['new_test_failures_count'] = summary["outcome_counts"][TestRun.Outcome.FAIL]['difference'] + summary["outcome_counts"][TestRun.Outcome.ERROR]['difference']
        summary['new_xfail_failures_count'] = summary["outcome_counts"][TestRun.Outcome.XFAIL]['difference']

    return summary


def get_session_results(session, baseline_session, nodeid, status, baseline_status, sort_by):
    """
    Get the detailed results of a test session, including the status and duration of each test case, and the comparison with another session if provided.
    """

    duration_qs = get_metric_change_percent_qs('total_duration', session, baseline_session)

    duration_qs = (duration_qs
                   .annotate(
                        nodeid=Concat(F('test_run__test_case__module'), Value('::'), 
                                    F('test_run__test_case__name'), Value('['), 
                                    F('test_run__test_case__parameters'), Value(']'), output_field=TextField()),                
                        current_test_run=Max("test_run__id", filter=Q(test_run__test_run_batch__test_session_id=session)),
                        other_test_run=Max("test_run__id", filter=Q(test_run__test_run_batch__test_session_id=baseline_session)),
                    )
                    .exclude(current_outcome__isnull=True)
    )

    duration_qs = duration_qs.order_by(sort_by)

    if nodeid != '':

        for token in nodeid.split():
            duration_qs = duration_qs.filter(nodeid__icontains=token)

    if status != 'ALL':
        duration_qs = duration_qs.filter(current_outcome=TestRun.Outcome(int(status)))
    
    if baseline_status != 'ALL':
        duration_qs = duration_qs.filter(other_outcome=TestRun.Outcome(int(baseline_status)))

    return duration_qs


def get_test_run_comparison(test_run, baseline_test_run):
    """
    Get the details of a test run, including the metrics and the comparison with another test run if provided.
    """

    baseline_measurements_by_metric = {}

    if baseline_test_run:
        baseline_test_run = TestRun.objects.select_related('test_case', 'test_run_batch__test_session').prefetch_related('measurement_set__metric').get(id=baseline_test_run.id)
        baseline_measurements_by_metric = {
            measurement.metric_id: measurement
            for measurement in baseline_test_run.measurement_set.all()
        }

    comparison_rows = []
    for measurement in test_run.measurement_set.all().order_by('metric__name'):
        baseline_measurement = baseline_measurements_by_metric.get(measurement.metric_id)

        difference = None
        change_percent = None
        if baseline_measurement:
            difference = measurement.value - baseline_measurement.value
            if baseline_measurement.value != 0:
                change_percent = (difference * 100.0) / baseline_measurement.value

        comparison_rows.append(
            {
                'metric_name': measurement.metric.name,
                'metric_description': measurement.metric.description,
                'unit': measurement.metric.unit,
                'current_value': measurement.value,
                'baseline_value': baseline_measurement.value if baseline_measurement else None,
                'difference': difference,
                'change_percent': change_percent,
            }
        )

    return comparison_rows
