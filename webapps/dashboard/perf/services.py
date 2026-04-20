
from .models import TestSession, TestRun, Measurement, TestGroup, TestRunBatch

from django.db.models import (
    Count, F, FloatField, IntegerField, Max, Avg, Count,
    ExpressionWrapper, Value, Q, TextField
)
from django.db.models.functions import Concat
from django.db import connection

import json


def get_latest_sessions_stats(criterias: Q, limit=None):
    """Get the latest test session for the branches in the given pattern"""

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


def get_metric_changes_query():
    """
    Get a SQL query that returns test runs of a session with the change in the specified metric compared to corresponding
    test runs in a baseline session.

    The parameters of the query are:

    - baseline_session_id: the id of the baseline session to compare with
    - metric_name: the name of the metric to compare
    - current_session_id: the id of the current session to compare

    """

    return f"""
    
        /*
            Get the highest id test run that is passing for each test case in the baseline session
        */  

        WITH baseline_testrun AS (
            
            SELECT
                x.id,
                x.outcome,
                x.test_case_id
            FROM 
                (
                    SELECT baseline_testrun_groups.*,
                        ROW_NUMBER() OVER (
                            PARTITION BY baseline_testrun_groups.test_case_id
                            ORDER BY baseline_testrun_groups.id DESC
                        ) AS rn
                    FROM
                        perf_testrun AS baseline_testrun_groups
                    INNER JOIN
                        perf_testrunbatch AS baseline_testrunbatch ON baseline_testrun_groups.test_run_batch_id = baseline_testrunbatch.id
                    WHERE
                        baseline_testrunbatch.test_session_id = %(baseline_session_id)s AND baseline_testrun_groups.outcome = {TestRun.Outcome.PASS.value}
                ) AS x
            WHERE x.rn = 1
        ),

        /*
            Get the measurement for the specified metric for each test run.
        */

        metric_value AS (
            SELECT
                test_run_id,
                value
            FROM
                perf_measurement
            INNER JOIN
                perf_metric ON perf_measurement.metric_id = perf_metric.id
            WHERE
                perf_metric.name = %(metric_name)s
        )

        /*
            Select all test runs for the current session that have a given metric, are passing,
            have a corresponding test run in the baseline session with a value for the metric.

            For each of these pairs compute the change in the metric

        */
        
        SELECT
            c_testrun.id AS current_id,
            b_testrun.id AS baseline_id,
            CONCAT(testcase.module, '::', testcase.name, '[', testcase.parameters, ']') AS nodeid,
            c_testrun.outcome AS current_outcome,
            b_testrun.outcome AS baseline_outcome,
            c_measurement.value AS current_value,
            b_measurement.value AS baseline_value,
            (c_measurement.value - b_measurement.value) * 100.0 / NULLIF(b_measurement.value, 0) AS change_percent        
        FROM
            perf_testrun as c_testrun
        INNER JOIN
            perf_testcase as testcase ON c_testrun.test_case_id = testcase.id
        INNER JOIN
            perf_testrunbatch AS c_testrunbatch ON c_testrun.test_run_batch_id = c_testrunbatch.id
        INNER JOIN
            perf_measurement AS c_measurement ON c_measurement.test_run_id = c_testrun.id
        INNER JOIN
            perf_metric AS c_metric ON c_metric.id = c_measurement.metric_id AND c_metric.name = 'total_duration'        
        INNER JOIN
            baseline_testrun as b_testrun ON c_testrun.test_case_id = b_testrun.test_case_id
        INNER JOIN
            perf_measurement AS b_measurement ON b_measurement.test_run_id = b_testrun.id
        INNER JOIN
            perf_metric AS b_metric ON b_metric.id = b_measurement.metric_id AND b_metric.name = 'total_duration'
        WHERE 
            c_testrunbatch.test_session_id = %(current_session_id)s 
        AND
            c_testrun.outcome = {TestRun.Outcome.PASS.value}
    """, {}


def get_histogram(sql_query, field, fields, num_bins, min_value, max_value):
    """
    Compute a histogram of the given field for the given queryset using PostgreSQL's width_bucket function.
    """
    
    # Build histogram from the queryset SQL as a derived table:
    # SELECT width_bucket(subq.<field>, ...) ... FROM (<qs SQL>) AS subq
    
    histogram_sql = f"""
        SELECT
            width_bucket(subq.\"{field}\"::double precision, %(max_value)s, %(min_value)s, %(num_bins)s) AS bucket,
            COUNT(*) AS count
        FROM ({sql_query}) AS subq
        GROUP BY bucket
        ORDER BY bucket
    """

    fields['num_bins'] = num_bins
    fields['min_value'] = min_value
    fields['max_value'] = max_value

    # we use inverted bins to make sure 0 ends up in the negative bin (which is the "good bin")
    with connection.cursor() as cursor:
        cursor.execute(histogram_sql, fields)
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


def get_average(sql_query, field, fields):
    """
    Compute the average of the given field for the given queryset.
    """

    average_query = f"""
        SELECT AVG(subq.\"{field}\"::double precision) AS average
        FROM ({sql_query}) AS subq
    """

    with connection.cursor() as cursor:
        cursor.execute(average_query, fields)
        row = cursor.fetchone()

    return row[0]


def get_duration_changes_query():
    sql, fields = get_metric_changes_query()

    fields['metric_name'] = 'total_duration'
    
    return sql + ' AND b_measurement.value > %(min_duration)s', fields


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


def get_session_results(session, baseline_session, nodeid, status, baseline_status, sort_by, page_number, page_size):
    """
    Get the detailed results of a test session, including the status and duration of each test case, and the comparison with another session if provided.
    """

    # Get the test runs for the current session, excluding skipped tests, and their metric value for the specified metric (if any).
    results_query = f"""
        SELECT DISTINCT ON (x.id)
            x.id,
            x.outcome,
            x.test_case_id,
            CONCAT(testcase.module, '::', testcase.name, '[', testcase.parameters, ']') AS nodeid,
            measurement.value AS value
        FROM
            perf_testrun AS x
        INNER JOIN
            perf_testrunbatch AS testrunbatch ON x.test_run_batch_id = testrunbatch.id            
        INNER JOIN
            perf_testcase as testcase ON x.test_case_id = testcase.id
        LEFT JOIN
            perf_measurement AS measurement ON measurement.test_run_id = x.id AND measurement.metric_id = (SELECT id FROM perf_metric WHERE name = 'total_duration')        
        WHERE
            testrunbatch.test_session_id = %s AND x.outcome != {TestRun.Outcome.SKIP.value}
    """

    if baseline_session:

        # Get one test run per test case from the baseline session, preferring passing runs if multiple runs exist 
        # for the same test case. Exclude tests that where skipped as they don't bring any useful information.
        # Add the metric value for the specified metric (if any) for these test runs.       

        baseline_query = f"""     
            SELECT DISTINCT ON (x.id)
                x.id,
                x.outcome,
                x.test_case_id,
                measurement.value AS value
            FROM 
                (
                    SELECT baseline_testrun_groups.*,
                        ROW_NUMBER() OVER (
                            PARTITION BY baseline_testrun_groups.test_case_id
                            ORDER BY baseline_testrun_groups.outcome = {TestRun.Outcome.PASS.value} DESC, baseline_testrun_groups.id DESC
                        ) AS rn
                    FROM
                        perf_testrun AS baseline_testrun_groups
                    INNER JOIN
                        perf_testrunbatch AS baseline_testrunbatch ON baseline_testrun_groups.test_run_batch_id = baseline_testrunbatch.id
                    WHERE
                        baseline_testrunbatch.test_session_id = %s and baseline_testrun_groups.outcome != {TestRun.Outcome.SKIP.value}
                ) AS x
            LEFT JOIN
                perf_measurement AS measurement ON measurement.test_run_id = x.id AND measurement.metric_id = (SELECT id FROM perf_metric WHERE name = 'total_duration')            
            WHERE x.rn = 1        
        """        

        # Join the current session test runs with the baseline test runs based on the test case, and compute the change in duration between them (if available for both runs).    

        joined_results = f"""
            WITH baseline_testrun AS ({baseline_query}), testrun AS ({results_query})
                    
            SELECT
                c_testrun.id AS current_id,
                b_testrun.id AS baseline_id,
                c_testrun.nodeid AS nodeid,
                c_testrun.outcome AS current_outcome,
                b_testrun.outcome AS baseline_outcome,
                c_testrun.value AS current_duration,
                b_testrun.value AS baseline_duration,
                (c_testrun.value - b_testrun.value) * 100.0 / NULLIF(b_testrun.value, 0) AS change_percent
            FROM
                testrun as c_testrun
            LEFT JOIN
                baseline_testrun as b_testrun ON c_testrun.test_case_id = b_testrun.test_case_id
        """
        query_params = [baseline_session.id, session.id]
    else:
        joined_results = f"""
            WITH testrun AS ({results_query})
                    
            SELECT
                c_testrun.id AS current_id,
                NULL AS baseline_id,
                c_testrun.nodeid AS nodeid,
                c_testrun.outcome AS current_outcome,
                NULL AS baseline_outcome,
                c_testrun.value AS current_duration,
                NULL AS baseline_duration,
                NULL AS change_percent
            FROM
                testrun as c_testrun            
        """
        query_params = [session.id]

    # filter and order the results based on the parameters    
    query = f"SELECT * FROM ({joined_results}) AS subquery WHERE TRUE"

    if nodeid != '':
        for token in nodeid.split():
            query += " AND nodeid ILIKE %s"
            query_params.append(f"%{token}%")   

    if status != 'ALL':
        query += " AND current_outcome = %s"
        query_params.append(int(status))

    if baseline_status != 'ALL':
        query += " AND baseline_outcome = %s"
        query_params.append(int(baseline_status))

    if sort_by:
        query += f" ORDER BY {sort_by}"
    else:
        query += " ORDER BY nodeid"

    # Get the total number of results without the pagination
    with connection.cursor() as cursor:
        cursor.execute(f"SELECT COUNT(*) FROM ({query})", query_params)
        total = cursor.fetchone()[0]

    # Apply pagination to the query
    if page_number is not None and page_size is not None:
        offset = (page_number - 1) * page_size
        query += f" LIMIT {page_size} OFFSET {offset}"

    # Get the paginated results
    with connection.cursor() as cursor:
        cursor.execute(query, query_params)
        columns = [col[0] for col in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]

    return rows, total


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
