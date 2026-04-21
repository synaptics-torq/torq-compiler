from ..models import TestRun

from django.db import connection

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


