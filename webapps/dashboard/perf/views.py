from django.shortcuts import render
from django.http import JsonResponse
from django.db.models import Max
from .models import TestSession, TestRun, Measurement
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

    # Get the latest session from the main branch
    main_branch_session = TestSession.objects.filter(
        git_branch=main_branch
    ).order_by('-timestamp').first()
    
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
    
    # Get the actual latest session for each of these branches
    other_branch_sessions = []
    for branch_info in other_branches:
        session = TestSession.objects.filter(
            git_branch=branch_info['git_branch']
        ).order_by('-timestamp').first()
        if session:
            other_branch_sessions.append(session)

    return render(request, 'perf/home.html', {
        'main_branch_session': main_branch_session,
        'other_branch_sessions': other_branch_sessions
    })


def test_session(request, session_id):
    session = TestSession.objects.prefetch_related('testrunbatch_set__testrun_set__test_case').get(id=session_id)
    
    # Collect all .pb files from all batches in this session
    pb_files = []
    test_run_by_pb = {}  # Map pb file path to test_run for querying measurements
    test_names_by_pb = {}  # Map pb file path to test case name for display
    
    for batch in session.testrunbatch_set.all():
        for test_run in batch.testrun_set.all():
            if test_run.profiling_data:
                pb_path = test_run.profiling_data.path
                if os.path.exists(pb_path) and pb_path.endswith('.pb'):
                    pb_path_obj = Path(pb_path)
                    pb_files.append(pb_path_obj)
                    # Use str(Path) as key to match what's checked in generate_html()
                    test_run_by_pb[str(pb_path_obj)] = test_run
                    # Store the test case name for display
                    test_names_by_pb[str(pb_path_obj)] = f"{test_run.test_case.name}[{test_run.test_case.parameters}]"
    
    # Fetch metrics from database for all test runs in this session
    db_summaries = {}
    if test_run_by_pb:
        for pb_path, test_run in test_run_by_pb.items():
            # Query all measurements for this test run
            measurements = Measurement.objects.filter(test_run=test_run).select_related('metric')
            
            # Build summary dictionary matching extract_perfetto_summary() format
            summary = {}
            for measurement in measurements:
                metric_key = measurement.metric.name
                metric_unit = measurement.metric.unit
                
                # Format value based on unit
                if metric_unit == 'ns':
                    # Convert nanoseconds to human-readable format
                    ns = measurement.value
                    if ns >= 1_000_000:
                        formatted = f"{ns / 1_000_000:.3f}ms"
                    elif ns >= 1_000:
                        formatted = f"{ns / 1_000:.3f}µs"
                    else:
                        formatted = f"{ns:.3f}ns"
                    summary[metric_key] = formatted
                elif metric_unit == '%':
                    # Format percentage (stored as numeric, display with 2 decimals)
                    summary[metric_key] = f"{measurement.value:.2f}"
                else:
                    # Keep as-is
                    summary[metric_key] = str(measurement.value)
            
            # Only add to db_summaries if we got metrics
            if summary:
                db_summaries[pb_path] = summary
    
    # Generate one combined HTML for all traces in the session
    perfetto_viewer_html = None
    if pb_files:
        try:
            # Pass database summaries and test names to avoid re-parsing .pb files
            perfetto_viewer_html = generate_perfetto_html(pb_files, db_summaries=db_summaries, test_names=test_names_by_pb)
        except Exception as e:
            print(f'ERROR: Could not generate Perfetto viewer HTML: {e}', flush=True)
            import traceback
            traceback.print_exc()

    return render(request, 'perf/test_session.html', {
        'session': session, 
        'perfetto_viewer_html': perfetto_viewer_html,
        'has_perfetto_viewer': perfetto_viewer_html is not None
    })
