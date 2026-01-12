from django.shortcuts import render
from django.http import JsonResponse
from django.db.models import Max
from .models import TestSession
import os


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
    
    redirected_url = f"/?__sign={token}"
    
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
    session = TestSession.objects.prefetch_related('testrun_set__test_case').get(id=session_id)

    return render(request, 'perf/test_session.html', {'session': session})
