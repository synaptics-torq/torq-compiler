from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse, FileResponse, Http404
from django.db.models import Prefetch, Q
from .models import TestSession, TestRun, TestRunBatch, Measurement
import os
from pathlib import Path
from perf.perfetto_report_generator import generate_html as generate_perfetto_html
from django.db.models import Max

from . import services

def health(request):
    """Health check endpoint for Docker."""
    return JsonResponse({'status': 'healthy'}, status=200)


def home(request):

    # Get the last session all the version branches
    version_branch_sessions = services.get_latest_sessions_stats(Q(git_branch__startswith='refs/heads/v') | Q(git_branch='refs/heads/main'))
        
    # Get the latest session for each of the 10 most recent PR branches    
    pr_branch_sessions = services.get_latest_sessions_stats(Q(git_branch__startswith='refs/pull'), limit=10)

    # Get the performance for each test case in the "home" group    
    test_durations = services.get_reference_test_durations([s.id for s in version_branch_sessions])
         
    return render(request, 'perf/home.html', {
        'version_branch_sessions': version_branch_sessions,
        'pr_branch_sessions': pr_branch_sessions,
        'test_durations': test_durations,
        'dashboard_git_commit': os.environ.get('DASHBOARD_GIT_COMMIT', 'unknown')
    })

def get_alternative_engines_results(layer):
    # Fetch results of alternative engines for the current layer
    latest_ids = (
        Measurement.objects.filter(
            test_run__test_case__name="test_alternative_engine",
            test_run__test_case__parameters__icontains=layer + "-",
            metric__name="total_duration",
            metric__unit="ns",
        )
        .values(
            "test_run__test_case__module",
            "test_run__test_case__name",
            "test_run__test_case__parameters",
        )
        .annotate(latest_id=Max("id"))
        .values_list("latest_id", flat=True)
    )

    results = (
        Measurement.objects.filter(id__in=latest_ids)
        .select_related("test_run__test_case", "metric")
        .order_by("-id")
    )

    return results

def build_test_key(test_case):
    return f"{test_case.module}_{test_case.name}_{test_case.parameters}"

def build_test_display_name(test_case):
    return f"{test_case.module}::{test_case.name}[{test_case.parameters}]"

def format_measurement_value(measurement):
    if measurement.metric.unit == '%':
        return f"{measurement.value:.2f}"

    ns = measurement.value
    if ns >= 1_000_000_000:
        return f"{ns / 1_000_000_000:.3f}s"
    elif ns >= 1_000_000:
        return f"{ns / 1_000_000:.3f}ms"
    elif ns >= 1_000:
        return f"{ns / 1_000:.3f}µs"
    else:
        return f"{ns:.3f}ns"

def build_summary_for_test_run(test_run):
    summary = {}
    for measurement in test_run.measurement_set.all():
        metric_key = measurement.metric.name
        summary[metric_key] = format_measurement_value(measurement)
    return summary

def test_session(request, session_id):
    # Prefetch all related data including measurements to avoid N+1 queries
    # Use Prefetch objects for more control and only fetch needed fields
    session = TestSession.objects.prefetch_related(
        Prefetch(
            'testrunbatch_set',
            queryset=TestRunBatch.objects.select_related().only('id', 'name', 'test_session_id')
        ),
        Prefetch(
            'testrunbatch_set__testrun_set',
            queryset=TestRun.objects.select_related('test_case').only(
                'id', 'outcome', 'profiling_data', 'failure_log', 'failure_type', 'test_run_batch_id', 'test_case_id',
                'test_case__name', 'test_case__parameters', 'test_case__module'
            )
        ),
        Prefetch(
            'testrunbatch_set__testrun_set__measurement_set',
            queryset=Measurement.objects.filter(metric__unit__in=['ns', '%']).select_related('metric')
        )
    ).get(id=session_id)
    
    # Collect all .pb files from all batches in this session
    pb_files = []
    test_run_by_pb = {}  # Map pb file path to test_run for querying measurements
    test_names_by_pb = {}  # Map pb file path to test case name for display
    test_run_ids_by_pb = {}  # Map pb file path to test_run.id for download URLs
    test_statuses_by_pb = {}  # Map pb file path to test status (Pass/Fail/Skip)
    non_profiled_tests = []  # Tests without .pb profiling data
    
    # Organize all test runs by outcome
    all_test_runs = []
    passed_tests = []
    failed_tests = []
    error_tests = []
    skipped_tests = []

    reference_keys = []
    reference_results_by_layer = {}
    reference_test_runs_by_key = {}
    xfail_tests = []
    nxpass_tests = []
    
    # Cache outcome display strings to avoid repeated method calls (performance optimization)
    outcome_display_map = {choice[0]: choice[1] for choice in TestRun.Outcome.choices}

    for batch in session.testrunbatch_set.all():
        batch_name = batch.name  # Cache batch name
        for test_run in batch.testrun_set.all():
            # Use cached outcome display instead of calling get_outcome_display() each time
            outcome_display = outcome_display_map.get(test_run.outcome, 'Unknown')

            # Getting the layer name assuming that the convention is [layer_name-engine]
            layer_name = test_run.test_case.parameters.split('-')[0]

            # Check if reference results are already in reference_results_by_layer to avoid re-fetching and register reference-results once per layer
            if layer_name not in reference_results_by_layer:
                # Query the database for the last test session reference-results for the given layer
                reference_results_by_layer[layer_name] = get_alternative_engines_results(layer_name)

                for reference in reference_results_by_layer[layer_name]:
                    reference_key = build_test_key(reference.test_run.test_case)
                    if reference_key not in reference_test_runs_by_key:

                        reference_test_runs_by_key[reference_key] = reference.test_run
                        reference_keys.append(Path(reference_key))
                        test_names_by_pb[reference_key] = build_test_display_name(reference.test_run.test_case)

            # Create test run info for display
            test_info = {
                'id': test_run.id,
                'name': test_run.test_case.name,
                'parameters': test_run.test_case.parameters,
                'module': test_run.test_case.module,
                'outcome': outcome_display,
                'outcome_class': test_run.outcome,
                'has_profiling': bool(test_run.profiling_data),
                'batch_name': batch_name,
                'has_failure_log': bool(test_run.failure_log) if test_run.outcome in (TestRun.Outcome.FAIL, TestRun.Outcome.ERROR) else False,
                'failure_log_url': f'/download-failure-log/{test_run.id}/' if test_run.outcome in (TestRun.Outcome.FAIL, TestRun.Outcome.ERROR) and test_run.failure_log else None,
                'failure_type': test_run.failure_type if test_run.outcome in (TestRun.Outcome.FAIL, TestRun.Outcome.ERROR) else None,
            }
            all_test_runs.append(test_info)
            
            # Categorize by outcome
            if test_run.outcome == TestRun.Outcome.PASS:
                passed_tests.append(test_info)
            elif test_run.outcome == TestRun.Outcome.FAIL:
                failed_tests.append(test_info)
            elif test_run.outcome == TestRun.Outcome.ERROR:
                error_tests.append(test_info)
            elif test_run.outcome == TestRun.Outcome.SKIP:
                skipped_tests.append(test_info)
            elif test_run.outcome == TestRun.Outcome.XFAIL:
                xfail_tests.append(test_info)
            elif test_run.outcome == TestRun.Outcome.NXPASS:
                nxpass_tests.append(test_info)
            
            # Handle profiling data for Perfetto viewer
            if test_run.profiling_data:
                pb_path = build_test_key(test_run.test_case) + '.pb'

                pb_files.append(Path(pb_path))
                # Use str(Path) as key to match what's checked in generate_html()
                test_run_by_pb[pb_path] = test_run
                # Store the test case name for display (module::name[params] matches session-metrics key)
                test_names_by_pb[pb_path] = build_test_display_name(test_run.test_case)
                # Store test_run.id for download URLs
                test_run_ids_by_pb[pb_path] = test_run.id
                # Store test status (using cached outcome display)
                test_statuses_by_pb[pb_path] = {
                    'outcome': outcome_display,
                    'outcome_value': test_run.outcome,
                    'failure_log_url': f'/download-failure-log/{test_run.id}/' if test_run.outcome in (TestRun.Outcome.FAIL, TestRun.Outcome.ERROR) and test_run.failure_log else None,
                    'failure_type': test_run.failure_type if test_run.outcome in (TestRun.Outcome.FAIL, TestRun.Outcome.ERROR) else None,
                }
            else:
                if test_info['name'] != 'test_alternative_engine':
                    non_profiled_tests.append(test_info)

    
    # Fetch metrics from database for all test runs in this session
    db_summaries = {
        pb_path: build_summary_for_test_run(test_run)
        for pb_path, test_run in test_run_by_pb.items()
    }

    reference_summaries = {
        key: build_summary_for_test_run(test_run)
        for key, test_run in reference_test_runs_by_key.items()
    }

    # Get list of recent sessions for comparison (limit to 100 most recent)
    # Only fetch needed fields for performance with large session tables
    recent_sessions = TestSession.objects.only('id', 'timestamp', 'git_branch').order_by('-id')[:100]
    available_sessions = [{'id': s.id, 'timestamp': s.timestamp.strftime('%Y-%m-%d %H:%M'), 'branch': s.git_branch or 'N/A'} for s in recent_sessions]
    
    # Find the most recent session from main branch (excluding current session) for default comparison
    default_comparison_session_id = None
    for recent_session in recent_sessions:
        if recent_session.id != session_id and recent_session.git_branch:
            branch_lower = recent_session.git_branch.lower()
            if branch_lower == 'main' or branch_lower == 'refs/heads/main':
                default_comparison_session_id = recent_session.id
                break
    
    # Get base URL from request for absolute URLs (when HTML viewed outside server)
    base_url = f"{request.scheme}://{request.get_host()}"
    
    # Generate one combined HTML for all traces in the session
    perfetto_viewer_html = None
    if pb_files or non_profiled_tests:
        try:
            # Pass database summaries, test names, test_run_ids, test statuses, base_url, and session info for comparison
            perfetto_viewer_html = generate_perfetto_html(
                pb_files, 
                db_summaries=db_summaries,
                reference_keys=reference_keys,
                reference_summaries=reference_summaries,
                test_names=test_names_by_pb, 
                test_run_ids=test_run_ids_by_pb,
                test_statuses=test_statuses_by_pb,
                base_url=base_url,
                current_session_id=session_id,
                available_sessions=available_sessions,
                default_comparison_session_id=default_comparison_session_id,
                non_profiled_tests=non_profiled_tests
            )
        except Exception as e:
            print(f'ERROR: Could not generate Perfetto viewer HTML: {e}', flush=True)
            import traceback
            traceback.print_exc()

    return render(request, 'perf/test_session.html', {
        'session': session, 
        'perfetto_viewer_html': perfetto_viewer_html,
        'has_perfetto_viewer': perfetto_viewer_html is not None,
        'all_test_runs': all_test_runs,
        'passed_tests': passed_tests,
        'failed_tests': failed_tests,
        'error_tests': error_tests,
        'skipped_tests': skipped_tests,
        'xfail_tests': xfail_tests,
        'nxpass_tests': nxpass_tests,
        'num_total': len(all_test_runs),
        'num_passed': len(passed_tests),
        'num_failed': len(failed_tests),
        'num_error': len(error_tests),
        'num_skipped': len(skipped_tests),
        'num_xfail': len(xfail_tests),
        'num_nxpass': len(nxpass_tests),
    })


def download_trace(request, test_run_id):
    """Download a perfetto trace file (.pb) for a specific test run."""
    # Use select_related to fetch test_case in the same query
    test_run = get_object_or_404(TestRun.objects.select_related('test_case'), id=test_run_id)
    
    if not test_run.profiling_data:
        raise Http404("No profiling data available for this test run")
        
    # Generate a friendly filename
    test_case = test_run.test_case
    filename = f"{test_case.name}_{test_case.parameters}.pb".replace('[', '_').replace(']', '_').replace(' ', '_')
    
    response = FileResponse(test_run.profiling_data.open('rb'), content_type='application/octet-stream')
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response


def download_failure_log(request, test_run_id):
    """Download a failure log file for a specific test run."""
    test_run = get_object_or_404(TestRun.objects.select_related('test_case'), id=test_run_id)

    if not test_run.failure_log:
        raise Http404("No failure log available for this test run")

    test_case = test_run.test_case
    filename = f"{test_case.name}_{test_case.parameters}.log".replace('[', '_').replace(']', '_').replace(' ', '_')

    response = FileResponse(test_run.failure_log.open('rb'), content_type='text/plain')
    response['Content-Disposition'] = f'inline; filename="{filename}"'
    return response


def get_session_metrics(request, session_id):
    """Get all metrics for a test session - used for comparison feature."""
    try:
        # Use prefetch_related to fetch all related data in just 3 queries instead of N+1 queries
        # This is critical for performance with 3000 test cases
        session = TestSession.objects.prefetch_related(
            'testrunbatch_set__testrun_set__test_case',
            Prefetch(
                'testrunbatch_set__testrun_set__measurement_set',
                queryset=Measurement.objects.filter(metric__unit__in=['ns', '%']).select_related('metric')
            )
        ).get(id=session_id)
        
        # Collect all test runs and their metrics
        metrics_data = {}
        
        for batch in session.testrunbatch_set.all():
            for test_run in batch.testrun_set.all():
                test_key = f"{test_run.test_case.module}::{test_run.test_case.name}[{test_run.test_case.parameters}]"
                
                # Access prefetched measurements (no additional database query)
                measurements = test_run.measurement_set.all()
                
                test_metrics = {}
                for measurement in measurements:
                    # Handle both time (ns) and percentage (%) metrics
                    # For time metrics, store as-is in nanoseconds for comparison calculations
                    # For percentage metrics, store the value directly (not used in comparison but kept for consistency)
                    test_metrics[measurement.metric.name] = float(measurement.value)
                
                if test_metrics:
                    metrics_data[test_key] = {
                        'metrics': test_metrics,
                        'test_run_id': test_run.id,
                        'has_trace': bool(test_run.profiling_data),
                        'outcome': test_run.get_outcome_display(),
                        'outcome_code': test_run.outcome
                    }
                else:
                    metrics_data[test_key] = {
                        'metrics': {},
                        'test_run_id': test_run.id,
                        'has_trace': bool(test_run.profiling_data),
                        'outcome': test_run.get_outcome_display(),
                        'outcome_code': test_run.outcome
                    }
        
        return JsonResponse({
            'session_id': session_id,
            'timestamp': session.timestamp.isoformat(),
            'branch': session.git_branch,
            'metrics': metrics_data
        })
        
    except TestSession.DoesNotExist:
        return JsonResponse({'error': 'Session not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
