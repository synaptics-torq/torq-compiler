from ..models import TestRun, TestSession


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


def get_duration_changes_query():
    """
    Get a SQL query that returns test runs of a session with the change in total_duration compared to corresponding
    test runs in a baseline session, but only for test runs where the total_duration is above a certain threshold 
    in the baseline session. This is useful to focus on the most significant changes.
    """

    sql, fields = get_metric_changes_query()

    fields['metric_name'] = 'total_duration'
    
    return sql + ' AND b_measurement.value > %(min_duration)s', fields


def get_default_comparison_session(session):
    """
    Get the default session to compare against for a given session. The logic is as follows:
    """

    # find the last two session on the main branch
    main_sessions = TestSession.objects.filter(git_branch=session.git_branch, test_plan=session.test_plan).order_by('git_branch', '-timestamp')[:2]

    # there are no session on the main branch, return None
    if len(main_sessions) == 0:
        return None

    # if the current session is the latest on main, return the previous one, otherwise return the latest one
    if main_sessions[0].id == session.id:
        if len(main_sessions) > 1:
            return main_sessions[1]
        else:
            return None
    else:
        return main_sessions[0]
