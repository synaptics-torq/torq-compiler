from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, FileResponse, Http404
from django.db.models import Q
from django.core.paginator import Paginator

from .models import TestSession, TestRun

from . import forms
from . import queries

import os


def health(request):
    """Health check endpoint for Docker."""

    return JsonResponse({'status': 'healthy'}, status=200)


def home(request):
    """Home page showing recent performance sessions and test cases."""

    # Get the last session all the version branches
    head_or_versions = Q(git_branch__startswith='refs/heads/v') | Q(git_branch='refs/heads/main')
    version_branch_sessions = queries.test_session_statistics.get_latest_sessions_stats(head_or_versions)
        
    # Get the latest session for each of the 10 most recent PR branches    
    pull_or_local = Q(git_branch__startswith='refs/pull') | Q(git_branch__isnull=True)
    pr_branch_sessions = queries.test_session_statistics.get_latest_sessions_stats(pull_or_local, limit=10)

    # Get the performance for each test case in the "home" group    
    test_durations = queries.test_case_summary.get_reference_test_durations([s.id for s in version_branch_sessions])
         
    return render(request, 'perf/home.html', {
        'version_branch_sessions': version_branch_sessions,
        'pr_branch_sessions': pr_branch_sessions,
        'test_durations': test_durations,
        'dashboard_git_commit': os.environ.get('DASHBOARD_GIT_COMMIT', 'unknown')
    })


def main_branch_test_trends(request):
    """Top-level view showing total and xfail counts over time for main sessions."""

    sessions = queries.test_session_statistics.get_main_branch_test_counts_over_time()
    latest_session = sessions[-1] if sessions else None
    chart_data = {
        'kind': 'multi_series',
        'datasets': [
            {
                'name': 'num_total',
                'label': 'Total tests',
                'unit': 'unitless',
                'borderColor': 'rgb(13, 110, 253)',
                'backgroundColor': 'rgba(13, 110, 253, 0.12)',
            },
            {
                'name': 'num_xfail',
                'label': 'XFail tests',
                'unit': 'unitless',
                'borderColor': 'rgb(255, 193, 7)',
                'backgroundColor': 'rgba(255, 193, 7, 0.16)',
            },
        ],
        'points': [
            {
                'session_id': session.id,
                'timestamp': session.timestamp.isoformat(),
                'git_branch': session.git_branch or '',
                'git_commit': session.git_commit or '',
                'workflow_url': session.workflow_url or '',
                'values': {
                    'num_total': session.num_total,
                    'num_xfail': session.num_xfail,
                },
            }
            for session in sessions
        ],
        'x_axis_label': 'Session timestamp',
        'y_axis_label': 'Number of tests',
    }

    return render(request, 'perf/main_branch_test_trends.html', {
        'sessions': sessions,
        'latest_session': latest_session,
        'chart_data': chart_data,
    })


def test_session_summary(request, session_id):
    """Summary page for a test session showing overall statistics."""

    session = get_object_or_404(TestSession, id=session_id)

    options = {'baseline_session': None, 'min_duration_ns': 1000 * 1000}

    if request.GET:
        form = forms.TestSessionSummaryOptions(data=request.GET)

        if not form.is_valid():                
            return redirect('test_session_summary', session_id=session_id)

        baseline_session = form.cleaned_data['baseline_session']
        
        if min_duration_ns := form.cleaned_data['min_duration_ns']:
            options['min_duration_ns'] = min_duration_ns

    else:
        baseline_session = queries.test_session_comparison.get_default_comparison_session(session_id)
        form = forms.TestSessionSummaryOptions(initial={'baseline_session': baseline_session, 'min_duration_ns': options['min_duration_ns']})
    
    statistics = queries.test_session_statistics.get_session_summary_statistics(session, baseline_session, options['min_duration_ns'])

    return render(request, 'perf/test_session/summary.html', {'session': session, 'baseline_session': baseline_session, 'statistics': statistics, 'form': form})


def test_session_results(request, session_id):
    """Detailed results page for a test session showing all test case runs and their metrics with filtering and sorting."""

    session = get_object_or_404(TestSession, id=session_id)

    options = {'baseline_session': None, 'nodeid': '', 'status': 'ALL', 'comparison_transition': 'ALL', 'sort_by': 'nodeid', 'page': 1}

    if request.GET:
        form = forms.TestSessionResultsOptions(data=request.GET)

        if not form.is_valid():                
            return redirect('test_session_results', session_id=session_id)

        baseline_session = form.cleaned_data['baseline_session']
        options.update(form.cleaned_data)
    else:
        baseline_session = queries.test_session_comparison.get_default_comparison_session(session_id)
        form = forms.TestSessionResultsOptions(initial={'baseline_session': baseline_session})
    

    results_page, total = queries.test_session_results.get_session_results(
        session,
        baseline_session,
        options['nodeid'],
        options['status'],
        options['comparison_transition'],
        options['sort_by'],
        options['page'],
        50,
    )

    # paginate the results
    paginator = Paginator(range(total), 50) # since this view uses raw SQL we don't directly pass a queryset to the paginator
    page_obj = paginator.get_page(options['page'])
    page_obj.object_list = results_page # Attach the actual results to the page object for rendering

    return render(request, 'perf/test_session/results.html',
        {
            'session': session,
            'baseline_session': baseline_session,
            'results': page_obj,
            'form': form
        })


def test_run(request, test_run_id):
    """Detailed page for a specific test run showing its metrics and comparison with a baseline test run."""

    test_run = get_object_or_404(
        TestRun.objects.select_related('test_case', 'test_run_batch__test_session').prefetch_related('measurement_set__metric'),
        id=test_run_id,
    )

    if request.GET:
        form = forms.TestRunOptions(data=request.GET)

        if not form.is_valid():
            return redirect('test_run', test_run_id=test_run_id)

        baseline_session = form.cleaned_data['baseline_session']
        baseline_test_run = form.cleaned_data['baseline_test_run']
        history_options = {
            'metric': form.cleaned_data['metric'] or None,
            'start_date': form.cleaned_data['start_date'],
            'end_date': form.cleaned_data['end_date'],
        }

    else:
        form = forms.TestRunOptions()
        baseline_session = None
        baseline_test_run = None
        history_options = {}

    if baseline_session is not None and baseline_test_run is None:
        baseline_test_run = queries.test_run_comparison.get_corresponding_test_run(test_run, baseline_session)

    metrics_with_comparison = queries.test_run_comparison.get_test_run_comparison(test_run, baseline_test_run)
    grouped_metrics_with_comparison = queries.test_run_comparison.group_metrics_with_comparison(metrics_with_comparison)
    test_run_history = queries.test_run_comparison.get_test_run_history(
        test_run,
        baseline_test_run,
        history_options=history_options,
    )

    total_duration_row = next(
        (row for row in metrics_with_comparison if row['metric_name'] == 'total_duration'),
        None,
    )

    data = {
        'comparison_rows': metrics_with_comparison,
        'total_duration': total_duration_row,
    }

    return render(request, 'perf/test_run.html',
        {
            'test_run': test_run,
            'baseline_test_run': baseline_test_run,
            'baseline_session': baseline_session,
            'form': form,
            'data': data,
            'metrics_with_comparison': metrics_with_comparison,
            'grouped_metrics_with_comparison': grouped_metrics_with_comparison,
            'test_run_history': test_run_history,
            'total_duration_row': total_duration_row,
        },
    )


def test_session(request, session_id):
    return redirect('test_session_summary', session_id=session_id)


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

