from django.contrib import admin
from django.contrib.admin.widgets import AutocompleteSelectMultiple
from django import forms

from .models import TestCase, TestGroup, TestSession, TestRun, TestRunBatch, Metric, Measurement


@admin.register(TestCase)
class TestCaseAdmin(admin.ModelAdmin):
    list_display = ('id', 'module', 'name', 'parameters')
    search_fields = ('module', 'name', 'parameters')


@admin.register(TestSession)
class TestSessionAdmin(admin.ModelAdmin):
    list_display = ('id', 'timestamp', 'owner', 'git_commit', 'git_branch', 'workflow_url', 'num_passed', 'num_skipped', 'num_failed', 'num_total')
    readonly_fields = ('timestamp', 'num_passed', 'num_skipped', 'num_failed', 'num_total')
    list_filter = ('git_branch', 'owner')
    search_fields = ('git_commit', 'git_branch', 'owner')


@admin.register(TestRunBatch)
class TestRunBatchAdmin(admin.ModelAdmin):
    list_display = ('id', 'test_session', 'name', 'created_at', 'processed')
    list_filter = ('processed', 'test_session')
    search_fields = ('name', 'test_session__git_commit')
    readonly_fields = ('created_at',)


@admin.register(TestRun)
class TestRunAdmin(admin.ModelAdmin):
    list_display = ('id', 'test_case', 'test_run_batch', 'outcome', 'failure_type')
    list_filter = ('outcome', 'failure_type')
    search_fields = ('test_case__module', 'test_case__name', 'test_case__parameters')


@admin.register(Metric)
class MetricAdmin(admin.ModelAdmin):
    list_display = ('name', 'unit', 'description')
    search_fields = ('name', 'description')


@admin.register(Measurement)
class MeasurementAdmin(admin.ModelAdmin):
    list_display = ('id', 'metric', 'test_run', 'value')
    list_filter = ('metric',)
    search_fields = ('metric__name', 'test_run__test_case__name')


class TestGroupAdminForm(forms.ModelForm):
    class Meta:
        model = TestGroup
        fields = '__all__'

        # we want an autocomplete widget that is wider because test names are very long
        widgets = {
            "test_cases": AutocompleteSelectMultiple (
                TestGroup.test_cases.field,
                admin.site,
                attrs={"style": "width: 800px;"}
            ),
        }


@admin.register(TestGroup)
class TestGroupAdmin(admin.ModelAdmin):
    form = TestGroupAdminForm
    list_display = ('id', 'name')
    search_fields = ('name',)
    autocomplete_fields = ('test_cases',)
