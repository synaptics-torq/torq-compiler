from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse, FileResponse, Http404
from django.db.models import Max, Count, Q, Case, When, IntegerField, Prefetch
from .models import TestSession, TestRun, TestRunBatch, Measurement
import os
from pathlib import Path
from perf.perfetto_report_generator import generate_html as generate_perfetto_html


def health(request):
    """Health check endpoint for Docker."""
    return JsonResponse({'status': 'healthy'}, status=200)


def space(request):
    # check if we are running inside a huggingface space
    if os.environ.get("SPACE_HOST") is None:
        return JsonResponse({'error': 'Not running inside a HuggingFace Space'}, status=400)
    
        
    token = request.GET.get('__sign', None)

    if token is None:                
        return JsonResponse({'error': 'Missing authentication parameter'}, status=400) 
    
    next = request.GET.get('next', '/')
    
    if not next.startswith('/'):
        return JsonResponse({'error': 'Invalid next parameter'}, status=400)
    
    redirected_url = f"{next}?__sign={token}"
    
    return render(request, 'perf/space.html', {'redirect_url': redirected_url})


def home(request):

    main_branch = 'refs/heads/main'

    # Get the latest session from the main branch (annotate counts to avoid N+1 queries)
    main_branch_session = TestSession.objects.filter(
        git_branch=main_branch
    ).prefetch_related(
        'testrunbatch_set__testrun_set__test_case'
    ).annotate(
        num_total=Count('testrunbatch__testrun'),
        num_passed=Count('testrunbatch__testrun', filter=Q(testrunbatch__testrun__outcome=TestRun.Outcome.PASS)),
        num_failed=Count('testrunbatch__testrun', filter=Q(testrunbatch__testrun__outcome=TestRun.Outcome.FAIL)),
        num_skipped=Count('testrunbatch__testrun', filter=Q(testrunbatch__testrun__outcome=TestRun.Outcome.SKIP))
    ).only('id', 'timestamp', 'git_branch', 'git_commit').order_by('-timestamp').first()
    
    # Organize test runs by outcome for main branch
    main_branch_test_data = None
    if main_branch_session:
        passed_tests = []
        failed_tests = []
        skipped_tests = []
        
        for batch in main_branch_session.testrunbatch_set.all():
            for test_run in batch.testrun_set.all():
                test_info = {
                    'name': test_run.test_case.name,
                    'parameters': test_run.test_case.parameters,
                    'module': test_run.test_case.module,
                    'id': test_run.id
                }
                if test_run.outcome == TestRun.Outcome.PASS:
                    passed_tests.append(test_info)
                elif test_run.outcome == TestRun.Outcome.FAIL:
                    failed_tests.append(test_info)
                elif test_run.outcome == TestRun.Outcome.SKIP:
                    skipped_tests.append(test_info)
        
        main_branch_test_data = {
            'passed': passed_tests,
            'failed': failed_tests,
            'skipped': skipped_tests
        }
    
    # Get the latest session for each of the 10 most recent non-main branches
    # First, get all non-main branches ordered by their latest session timestamp
    other_branches = TestSession.objects.exclude(
        git_branch=main_branch
    ).exclude(
        git_branch__isnull=True
    ).exclude(
        git_branch=''
    ).values('git_branch').annotate(
        latest_timestamp=Max('timestamp')
    ).order_by('-latest_timestamp')[:10]
    
    # Get the actual latest session for each of these branches (annotate counts and prefetch test data)
    other_branch_sessions = []
    other_branch_test_data = {}
    
    for branch_info in other_branches:
        session = TestSession.objects.filter(
            git_branch=branch_info['git_branch']
        ).prefetch_related(
            'testrunbatch_set__testrun_set__test_case'
        ).annotate(
            num_total=Count('testrunbatch__testrun'),
            num_passed=Count('testrunbatch__testrun', filter=Q(testrunbatch__testrun__outcome=TestRun.Outcome.PASS)),
            num_failed=Count('testrunbatch__testrun', filter=Q(testrunbatch__testrun__outcome=TestRun.Outcome.FAIL)),
            num_skipped=Count('testrunbatch__testrun', filter=Q(testrunbatch__testrun__outcome=TestRun.Outcome.SKIP))
        ).only('id', 'timestamp', 'git_branch', 'git_commit').order_by('-timestamp').first()
        
        if session:
            other_branch_sessions.append(session)
            
            # Organize test runs by outcome for this branch
            passed_tests = []
            failed_tests = []
            skipped_tests = []
            
            for batch in session.testrunbatch_set.all():
                for test_run in batch.testrun_set.all():
                    test_info = {
                        'name': test_run.test_case.name,
                        'parameters': test_run.test_case.parameters,
                        'module': test_run.test_case.module,
                        'id': test_run.id
                    }
                    if test_run.outcome == TestRun.Outcome.PASS:
                        passed_tests.append(test_info)
                    elif test_run.outcome == TestRun.Outcome.FAIL:
                        failed_tests.append(test_info)
                    elif test_run.outcome == TestRun.Outcome.SKIP:
                        skipped_tests.append(test_info)
            
            other_branch_test_data[session.id] = {
                'passed': passed_tests,
                'failed': failed_tests,
                'skipped': skipped_tests
            }

    return render(request, 'perf/home.html', {
        'main_branch_session': main_branch_session,
        'main_branch_test_data': main_branch_test_data,
        'other_branch_sessions': other_branch_sessions,
        'other_branch_test_data': other_branch_test_data
    })


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
                'id', 'outcome', 'profiling_data', 'test_run_batch_id', 'test_case_id',
                'test_case__name', 'test_case__parameters', 'test_case__module'
            )
        ),
        Prefetch(
            'testrunbatch_set__testrun_set__measurement_set',
            queryset=Measurement.objects.filter(metric__unit='ns').select_related('metric')
        )
    ).get(id=session_id)
    
    # Collect all .pb files from all batches in this session
    pb_files = []
    test_run_by_pb = {}  # Map pb file path to test_run for querying measurements
    test_names_by_pb = {}  # Map pb file path to test case name for display
    test_run_ids_by_pb = {}  # Map pb file path to test_run.id for download URLs
    test_statuses_by_pb = {}  # Map pb file path to test status (Pass/Fail/Skip)
    
    # Organize all test runs by outcome
    all_test_runs = []
    passed_tests = []
    failed_tests = []
    skipped_tests = []
    
    # Cache outcome display strings to avoid repeated method calls (performance optimization)
    outcome_display_map = {choice[0]: choice[1] for choice in TestRun.Outcome.choices}
    
    for batch in session.testrunbatch_set.all():
        batch_name = batch.name  # Cache batch name
        for test_run in batch.testrun_set.all():
            # Use cached outcome display instead of calling get_outcome_display() each time
            outcome_display = outcome_display_map.get(test_run.outcome, 'Unknown')
            
            # Create test run info for display
            test_info = {
                'id': test_run.id,
                'name': test_run.test_case.name,
                'parameters': test_run.test_case.parameters,
                'module': test_run.test_case.module,
                'outcome': outcome_display,
                'outcome_class': test_run.outcome,
                'has_profiling': bool(test_run.profiling_data),
                'batch_name': batch_name
            }
            all_test_runs.append(test_info)
            
            # Categorize by outcome
            if test_run.outcome == TestRun.Outcome.PASS:
                passed_tests.append(test_info)
            elif test_run.outcome == TestRun.Outcome.FAIL:
                failed_tests.append(test_info)
            elif test_run.outcome == TestRun.Outcome.SKIP:
                skipped_tests.append(test_info)
            
            # Handle profiling data for Perfetto viewer
            if test_run.profiling_data:
                pb_path = test_run.profiling_data.path
                if os.path.exists(pb_path) and pb_path.endswith('.pb'):
                    pb_path_obj = Path(pb_path)
                    pb_files.append(pb_path_obj)
                    # Use str(Path) as key to match what's checked in generate_html()
                    test_run_by_pb[str(pb_path_obj)] = test_run
                    # Store the test case name for display
                    test_names_by_pb[str(pb_path_obj)] = f"{test_run.test_case.name}[{test_run.test_case.parameters}]"
                    # Store test_run.id for download URLs
                    test_run_ids_by_pb[str(pb_path_obj)] = test_run.id
                    # Store test status (using cached outcome display)
                    test_statuses_by_pb[str(pb_path_obj)] = {
                        'outcome': outcome_display,
                        'outcome_value': test_run.outcome
                    }
    
    # Fetch metrics from database for all test runs in this session
    db_summaries = {}
    if test_run_by_pb:
        for pb_path, test_run in test_run_by_pb.items():
            # Access prefetched measurements (no additional database query)
            measurements = test_run.measurement_set.all()
            
            # Build summary dictionary matching extract_perfetto_summary() format
            summary = {}
            for measurement in measurements:
                metric_key = measurement.metric.name
                # All measurements are already filtered to ns unit at database level
                
                # Convert nanoseconds to human-readable format
                ns = measurement.value
                if ns >= 1_000_000:
                    formatted = f"{ns / 1_000_000:.3f}ms"
                elif ns >= 1_000:
                    formatted = f"{ns / 1_000:.3f}µs"
                else:
                    formatted = f"{ns:.3f}ns"
                summary[metric_key] = formatted
            
            # Only add to db_summaries if we got metrics
            if summary:
                db_summaries[pb_path] = summary
    
    # Get list of recent sessions for comparison (limit to 100 most recent)
    # Only fetch needed fields for performance with large session tables
    recent_sessions = TestSession.objects.only('id', 'timestamp', 'git_branch').order_by('-id')[:100]
    available_sessions = [{'id': s.id, 'timestamp': s.timestamp.strftime('%Y-%m-%d %H:%M'), 'branch': s.git_branch or 'N/A'} for s in recent_sessions]
    
    # Find the most recent session from main branch (excluding current session) for default comparison
    default_comparison_session_id = None
    for session in recent_sessions:
        if session.id != session_id and session.git_branch:
            branch_lower = session.git_branch.lower()
            if branch_lower == 'main' or branch_lower == 'refs/heads/main':
                default_comparison_session_id = session.id
                break
    
    # Get base URL from request for absolute URLs (when HTML viewed outside server)
    base_url = f"{request.scheme}://{request.get_host()}"
    
    # Generate one combined HTML for all traces in the session
    perfetto_viewer_html = None
    if pb_files:
        try:
            # Pass database summaries, test names, test_run_ids, test statuses, base_url, and session info for comparison
            perfetto_viewer_html = generate_perfetto_html(
                pb_files, 
                db_summaries=db_summaries, 
                test_names=test_names_by_pb, 
                test_run_ids=test_run_ids_by_pb,
                test_statuses=test_statuses_by_pb,
                base_url=base_url,
                current_session_id=session_id,
                available_sessions=available_sessions,
                default_comparison_session_id=default_comparison_session_id
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
        'skipped_tests': skipped_tests,
        'num_total': len(all_test_runs),
        'num_passed': len(passed_tests),
        'num_failed': len(failed_tests),
        'num_skipped': len(skipped_tests)
    })


def download_trace(request, test_run_id):
    """Download a perfetto trace file (.pb) for a specific test run."""
    # Use select_related to fetch test_case in the same query
    test_run = get_object_or_404(TestRun.objects.select_related('test_case'), id=test_run_id)
    
    if not test_run.profiling_data:
        raise Http404("No profiling data available for this test run")
    
    file_path = test_run.profiling_data.path
    if not os.path.exists(file_path):
        raise Http404("Profiling data file not found")
    
    # Generate a friendly filename
    test_case = test_run.test_case
    filename = f"{test_case.name}_{test_case.parameters}.pb".replace('[', '_').replace(']', '_').replace(' ', '_')
    
    response = FileResponse(open(file_path, 'rb'), content_type='application/octet-stream')
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
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
                queryset=Measurement.objects.filter(metric__unit='ns').select_related('metric')
            )
        ).get(id=session_id)
        
        # Collect all test runs and their metrics
        metrics_data = {}
        
        for batch in session.testrunbatch_set.all():
            for test_run in batch.testrun_set.all():
                test_key = f"{test_run.test_case.name}[{test_run.test_case.parameters}]"
                
                # Access prefetched measurements (no additional database query)
                measurements = test_run.measurement_set.all()
                
                test_metrics = {}
                for measurement in measurements:
                    # All measurements are already filtered to ns unit at database level
                    test_metrics[measurement.metric.name] = float(measurement.value)
                
                if test_metrics:
                    metrics_data[test_key] = {
                        'metrics': test_metrics,
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
