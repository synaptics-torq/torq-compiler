from django.contrib import admin

from .models import TestCase, TestSession, TestRun, TestRunBatch, Metric, Measurement

@admin.register(TestCase)
class TestCaseAdmin(admin.ModelAdmin):
    list_display = ('id', 'module', 'name', 'parameters')
    search_fields = ('module', 'name')


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
    list_display = ('id', 'test_case', 'test_run_batch', 'outcome')
    list_filter = ('outcome',)
    search_fields = ('test_case__name',)


@admin.register(Metric)
class MetricAdmin(admin.ModelAdmin):
    list_display = ('name', 'unit', 'description')
    search_fields = ('name', 'description')


@admin.register(Measurement)
class MeasurementAdmin(admin.ModelAdmin):
    list_display = ('id', 'metric', 'test_run', 'value')
    list_filter = ('metric',)
    search_fields = ('metric__name', 'test_run__test_case__name')

