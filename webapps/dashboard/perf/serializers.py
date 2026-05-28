from rest_framework import serializers
from .models import TestCase, TestSession, TestRun, TestGroup


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
        fields = ['id', 'test_run_batch', 'test_case', 'profiling_data', 'outcome', 'failure_type', 'failure_log']


class TestGroupListSerializer(serializers.ModelSerializer):    
    details = serializers.HyperlinkedIdentityField(view_name='testgroup-detail', lookup_field='name')

    class Meta:
        model = TestGroup
        fields = ['name', 'details']


class TestGroupDetailSerializer(serializers.ModelSerializer):
    test_cases = TestCaseSerializer(many=True, read_only=True)

    class Meta:
        model = TestGroup
        fields = ['name', 'test_cases']


class TestGroupMetricsQuerySerializer(serializers.Serializer):
    session_id = serializers.IntegerField(required=False)
    git_commit = serializers.CharField(required=False, allow_blank=True)
    git_branch = serializers.CharField(required=False, allow_blank=True)
    runtime_target = serializers.CharField(required=False, allow_blank=True)
    runtime_hw_type = serializers.CharField(required=False, allow_blank=True)
    compiler_input = serializers.CharField(required=False, allow_blank=True)

    def validate(self, attrs):
        # Normalize blank query values so downstream code can treat them as unset.
        for key in ['git_commit', 'git_branch', 'runtime_target', 'runtime_hw_type', 'compiler_input']:
            if attrs.get(key) == '':
                attrs[key] = None

        if not any([attrs.get('session_id'), attrs.get('git_commit'), attrs.get('git_branch')]):
            raise serializers.ValidationError('One of session_id, git_commit, or git_branch is required.')

        return attrs
