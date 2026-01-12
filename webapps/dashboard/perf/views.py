from django.shortcuts import render
from django.http import JsonResponse
from .models import TestSession


def health(request):
    """Health check endpoint for Docker."""
    return JsonResponse({'status': 'healthy'}, status=200)


def home(request):
    latest_session = TestSession.objects.order_by('-timestamp').first()

    return render(request, 'perf/home.html', {'latest_session': latest_session})


def test_session(request, session_id):
    session = TestSession.objects.prefetch_related('testrun_set__test_case').get(id=session_id)

    return render(request, 'perf/test_session.html', {'session': session})
