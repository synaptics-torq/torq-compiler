from django import forms
from django.contrib import admin, messages
from django.shortcuts import redirect, render
from django.utils import timezone

from datetime import timedelta

from .models import TestSession
from .processing import trim_database
from .trim_status import TrimStatus, get_trim_status, save_trim_status, timestamp


class TrimDatabaseForm(forms.Form):
    retain_days = forms.IntegerField(
        min_value=0,
        initial=14,
        label='Keep PR branches active in the last N days',
        help_text='PR branches are identified by refs/pull/* and kept if their latest session is within this window.',
    )


def trim_database_view(request):
    if request.method == 'POST':
        form = TrimDatabaseForm(request.POST)

        if form.is_valid():
            retain_days = form.cleaned_data['retain_days']
            cutoff = timezone.now() - timedelta(days=retain_days)

            sessions_to_trim = TestSession.objects.filter(
                git_branch__startswith='refs/pull/',
                timestamp__lt=cutoff,
            )

            trim_count = sessions_to_trim.count()
            if trim_count == 0:
                messages.info(request, 'Nothing to trim. No PR sessions are older than the selected age.')
                return redirect('admin:index')

            # Hand off the deletion to the background spooler so the request doesn't block on it
            save_trim_status(TrimStatus(
                in_progress=True,
                retain_days=retain_days,
                sessions_total=trim_count,
                started_at=timestamp(),
            ))
            trim_database.spool(retain_days=str(retain_days).encode('utf-8'))

            messages.warning(
                request,
                f'Trim queued. Deleting {trim_count} PR sessions older than {retain_days} days in the background.',
            )
            return redirect('admin:index')
    else:
        form = TrimDatabaseForm()

    context = {
        **admin.site.each_context(request),
        'form': form,
        'title': 'Trim Database',
        'trim_status': get_trim_status(),
    }
    return render(request, 'admin/perf/trim_database.html', context)