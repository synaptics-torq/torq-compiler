from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse, FileResponse, Http404
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
    test_run_ids_by_pb = {}  # Map pb file path to test_run.id for download URLs
    
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
                    # Store test_run.id for download URLs
                    test_run_ids_by_pb[str(pb_path_obj)] = test_run.id
    
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
    
    # Get list of all sessions for comparison
    recent_sessions = TestSession.objects.order_by('-id')
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
            # Pass database summaries, test names, test_run_ids, base_url, and session info for comparison
            perfetto_viewer_html = generate_perfetto_html(
                pb_files, 
                db_summaries=db_summaries, 
                test_names=test_names_by_pb, 
                test_run_ids=test_run_ids_by_pb,
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
        'has_perfetto_viewer': perfetto_viewer_html is not None
    })


def download_trace(request, test_run_id):
    """Download a perfetto trace file (.pb) for a specific test run."""
    test_run = get_object_or_404(TestRun, id=test_run_id)
    
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
        session = TestSession.objects.get(id=session_id)
        
        # Collect all test runs and their metrics
        metrics_data = {}
        
        for batch in session.testrunbatch_set.all():
            for test_run in batch.testrun_set.all():
                test_key = f"{test_run.test_case.name}[{test_run.test_case.parameters}]"
                
                # Get all measurements for this test run
                measurements = Measurement.objects.filter(test_run=test_run).select_related('metric')
                
                test_metrics = {}
                for measurement in measurements:
                    # Store only time-based metrics (unit='ns'), not percentages
                    if measurement.metric.unit == 'ns':
                        test_metrics[measurement.metric.name] = float(measurement.value)
                
                if test_metrics:
                    metrics_data[test_key] = {
                        'metrics': test_metrics,
                        'test_run_id': test_run.id,
                        'has_trace': bool(test_run.profiling_data)
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
