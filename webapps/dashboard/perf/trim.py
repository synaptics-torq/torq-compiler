from django import forms
from django.contrib import admin, messages
from django.shortcuts import redirect, render

from .processing import start_trim
from .trim_status import get_trim_status


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
            result = start_trim(retain_days)

            if result is None:
                messages.info(request, 'Nothing to trim. No PR sessions are older than the selected age.')
                return redirect('admin:index')

            trim_count, pr_count = result
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