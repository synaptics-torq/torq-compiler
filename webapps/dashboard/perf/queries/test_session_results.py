from ..models import TestRun, TestSession

from django.db import connection

outcome_rank_sql = f"""
    CASE
        WHEN {{alias}} = {TestRun.Outcome.PASS.value} THEN 1
        WHEN {{alias}} = {TestRun.Outcome.NXPASS.value} THEN 1
        WHEN {{alias}} = {TestRun.Outcome.XFAIL.value} THEN 2
        WHEN {{alias}} = {TestRun.Outcome.FAIL.value} THEN 3
        WHEN {{alias}} = {TestRun.Outcome.ERROR.value} THEN 4
        ELSE 5
    END
"""

def execute_paginated_results_query(query, query_params, sort_by, page_number, page_size):
    if sort_by:
        query += f" ORDER BY {sort_by}"
    else:
        query += " ORDER BY nodeid"

    # Get the total number of results without the pagination
    with connection.cursor() as cursor:
        cursor.execute(f"SELECT COUNT(*) FROM ({query}) AS count_query", query_params)
        total = cursor.fetchone()[0]

     # Apply pagination to the query
    if page_number is not None and page_size is not None:
        offset = (page_number - 1) * page_size
        query += f" LIMIT %s OFFSET %s"
        query_params = [*query_params, page_size, offset]

    # Get the paginated results
    with connection.cursor() as cursor:
        cursor.execute(query, query_params)
        columns = [col[0] for col in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]

    return rows, total

def apply_test_result_filters(
        query_params,
        nodeid,
        status,
        comparison_transition,
        joined_results,
        baseline_session = None
    ):

    # filter and order the results based on the parameters    
    query = f"SELECT * FROM ({joined_results}) AS subquery WHERE TRUE"

    if nodeid != '':
        for token in nodeid.split():
            query += " AND nodeid ILIKE %s"
            query_params.append(f"%{token}%")   

    if status and status != 'ALL':
        query += " AND current_outcome = %s"
        query_params.append(int(status))

    if baseline_session and comparison_transition == 'FAIL_TO_PASS':
        query += " AND baseline_outcome = %s AND current_outcome = %s"
        query_params.extend([TestRun.Outcome.FAIL.value, TestRun.Outcome.PASS.value])

    if baseline_session and comparison_transition == 'PASS_TO_FAIL':
        query += " AND baseline_outcome = %s AND current_outcome = %s"
        query_params.extend([TestRun.Outcome.PASS.value, TestRun.Outcome.FAIL.value])

    if baseline_session and comparison_transition == 'ERROR_TO_PASS':
        query += " AND baseline_outcome = %s AND current_outcome = %s"
        query_params.extend([TestRun.Outcome.ERROR.value, TestRun.Outcome.PASS.value])

    if baseline_session and comparison_transition == 'PASS_TO_XFAIL':
        query += " AND baseline_outcome = %s AND current_outcome = %s"
        query_params.extend([TestRun.Outcome.PASS.value, TestRun.Outcome.XFAIL.value])

    if baseline_session and comparison_transition == 'ANY_REGRESSION':
        query += f" AND ({outcome_rank_sql.format(alias='current_outcome')}) > ({outcome_rank_sql.format(alias='baseline_outcome')})"

    if baseline_session and comparison_transition == 'ANY_IMPROVEMENT':
        query += f" AND ({outcome_rank_sql.format(alias='current_outcome')}) < ({outcome_rank_sql.format(alias='baseline_outcome')})"

    return query, query_params

def get_test_runs(include_metadata=False):
    results_query = f"""
        SELECT DISTINCT ON (x.id)
            x.id,
            x.outcome,
            x.test_case_id,
            {"metadata.compiler_input," if include_metadata else ""}
            CONCAT(testcase.module, '::', testcase.name, '[', testcase.parameters, ']') AS nodeid,
            measurement.value AS value
        FROM
            perf_testrun AS x
        INNER JOIN
            perf_testrunbatch AS testrunbatch
                ON x.test_run_batch_id = testrunbatch.id
        INNER JOIN
            perf_testcase AS testcase
                ON x.test_case_id = testcase.id
        {"LEFT JOIN perf_testmetadata AS metadata ON x.test_metadata_id = metadata.id" if include_metadata else ""}
        LEFT JOIN
            perf_measurement AS measurement
                ON measurement.test_run_id = x.id
                AND measurement.metric_id = (
                    SELECT id FROM perf_metric WHERE name = 'total_duration'
                )
        WHERE
            testrunbatch.test_session_id = %s
            AND x.outcome != {TestRun.Outcome.SKIP.value}
    """

    return results_query

def join_current_and_baseline_results(baseline_query, results_query, join_key = ""):
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
            testrun AS c_testrun
        LEFT JOIN
            baseline_testrun AS b_testrun ON c_testrun.{join_key} = b_testrun.{join_key}
    """
    return joined_results

def get_session_results(session, baseline_session, nodeid, status, comparison_transition, sort_by, page_number, page_size):
    """
    Get the detailed results of a test session, including the status and duration of each test case, and the comparison with another session if provided.
    """

    results_query = get_test_runs()


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

        joined_results = join_current_and_baseline_results(baseline_query, results_query, join_key="test_case_id")
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

    query, query_params = apply_test_result_filters(
        query_params,
        nodeid,
        status,
        comparison_transition,
        joined_results,
        baseline_session
    )

    return execute_paginated_results_query(
        query, query_params, sort_by, page_number, page_size
    )


def get_latest_alternative_engine_session():
    return (
        TestSession.objects
        .filter(git_branch="main")
        .filter(test_plan="alt")
        .order_by("-timestamp")
        .first()
    )

def get_compiler_targets_for_session(session_id: int):
    if not session_id:
        return []
    
    return (
        TestRun.objects
        .filter(test_run_batch__test_session_id=session_id)
        .filter(test_metadata__compiler="external_engine")
        .exclude(test_metadata__compiler_target__isnull=True)
        .exclude(test_metadata__compiler_target="")
        .values_list("test_metadata__compiler_target", flat=True)
        .distinct()
        .order_by("test_metadata__compiler_target")
    )


def get_session_results_compared_to_external_engine(
    session,
    alt_session_id,
    engine_name,
    nodeid,
    status,
    comparison_transition,
    sort_by,
    page_number,
    page_size,
):
    """
    Get session results compared against a selected external engine.

    The baseline is not matched by test_case_id. It is matched by compiler_input,
    because the current/default test and external-engine test may be different
    test cases for the same model/layer.
    """

    results_query = get_test_runs(include_metadata=True)

    baseline_query = f"""
        SELECT
            metadata.compiler_input,
            MIN(measurement.value) AS value,
            MIN(x.id) AS id,
            MIN(x.outcome) AS outcome
        FROM
            perf_testrun AS x
        INNER JOIN
            perf_testrunbatch AS testrunbatch
                ON x.test_run_batch_id = testrunbatch.id
        INNER JOIN
            perf_testmetadata AS metadata
                ON x.test_metadata_id = metadata.id
        LEFT JOIN
            perf_measurement AS measurement
                ON measurement.test_run_id = x.id
                AND measurement.metric_id = (
                    SELECT id FROM perf_metric WHERE name = 'total_duration'
                )
        WHERE
            testrunbatch.test_session_id = %s
            AND x.outcome != {TestRun.Outcome.SKIP.value}
            AND metadata.compiler = 'external_engine'
            AND metadata.compiler_target = %s
            AND metadata.compiler_input IS NOT NULL
            AND metadata.compiler_input != ''
        GROUP BY
            metadata.compiler_input
    """

    joined_results = join_current_and_baseline_results(baseline_query, results_query, join_key="compiler_input")

    query_params = [
        alt_session_id,
        engine_name,
        session.id,
    ]

    query, query_params = apply_test_result_filters(
        query_params,
        nodeid,
        status,
        comparison_transition,
        joined_results
    )

    return execute_paginated_results_query(
        query, query_params, sort_by, page_number, page_size
    )