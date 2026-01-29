from rest_framework import serializers
from .models import TestCase, TestSession, TestRun


class TestCaseSerializer(serializers.ModelSerializer):
    class Meta:
        model = TestCase
        fields = ['id', 'module', 'name', 'parameters']


class TestSessionSerializer(serializers.ModelSerializer):
    num_passed = serializers.SerializerMethodField()
    num_failed = serializers.SerializerMethodField()
    num_skipped = serializers.SerializerMethodField()
    num_total = serializers.SerializerMethodField()

    class Meta:
        model = TestSession
        fields = [
            'id', 'timestamp', 'owner', 'git_commit', 'git_branch', 'workflow_url',
            'num_passed', 'num_failed', 'num_skipped', 'num_total'
        ]
        read_only_fields = ['timestamp', 'owner', 'num_passed', 'num_failed', 'num_skipped', 'num_total']

    def get_num_passed(self, obj):
        return getattr(obj, 'num_passed', 0)

    def get_num_failed(self, obj):
        return getattr(obj, 'num_failed', 0)

    def get_num_skipped(self, obj):
        return getattr(obj, 'num_skipped', 0)

    def get_num_total(self, obj):
        return getattr(obj, 'num_total', 0)


class TestRunSerializer(serializers.ModelSerializer):
    class Meta:
        model = TestRun
        fields = ['id', 'test_run_batch', 'test_case', 'profiling_data', 'outcome']

