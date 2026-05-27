from django import forms
from django.contrib import admin, messages
from django.db import transaction
from django.shortcuts import redirect, render
from django.utils import timezone

from datetime import timedelta

from .models import TestSession


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

            try:
                with transaction.atomic():
                    deleted_total, _ = sessions_to_trim.delete()
            except Exception as exc:
                messages.error(request, f'Trim failed: {exc}')
            else:
                messages.warning(
                    request,
                    f'Trim completed. Deleted {trim_count} PR sessions older than {retain_days} days '
                    f'({deleted_total} total rows including related data).',
                )
                return redirect('admin:index')
    else:
        form = TrimDatabaseForm()

    context = {
        **admin.site.each_context(request),
        'form': form,
        'title': 'Trim Database',
    }
    return render(request, 'admin/perf/trim_database.html', context)