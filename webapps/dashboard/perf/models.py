from django.db import models
from django.utils.functional import cached_property


class TestCase(models.Model):
    module = models.TextField()
    name = models.TextField()
    parameters = models.TextField()

    @property
    def nodeid(self):
        if self.parameters:
            return f"{self.module}::{self.name}[{self.parameters}]"
        else:
            return f"{self.module}::{self.name}"

    @property
    def model_type(self):
        parts = self.name.split('_', 3)

        if len(parts) < 3:
            return 'other'
        
        return parts[1]

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=['module', 'name', 'parameters'],
                name='unique_testcase_module_name_parameters'
            )
        ]

    def __str__(self):
        return f"{self.module}::{self.name}[{self.parameters}]"


class TestGroup(models.Model):
    name = models.TextField(unique=True)
    test_cases = models.ManyToManyField(TestCase)

    def __str__(self):
        return f"TestGroup: {self.name}"


class TestSession(models.Model):    
    timestamp = models.DateTimeField(auto_now_add=True)
    owner = models.TextField(blank=True, null=True)
    git_commit = models.TextField(blank=True, null=True)
    git_branch = models.TextField(blank=True, null=True)
    workflow_url = models.TextField(blank=True, null=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=['workflow_url'],
                condition=models.Q(workflow_url__isnull=False),
                name='unique_workflow_url'
            )
        ]
        indexes = [
            models.Index(fields=['git_branch', '-timestamp'], name='perf_session_branch_ts_idx'),
        ]

    @property
    def test_runs(self):
        return TestRun.objects.filter(test_run_batch__test_session=self)

    @cached_property
    def num_passed(self):        
        return TestRun.objects.filter(test_run_batch__test_session=self, outcome=TestRun.Outcome.PASS).count()

    @cached_property
    def num_skipped(self):        
        return TestRun.objects.filter(test_run_batch__test_session=self, outcome=TestRun.Outcome.SKIP).count()

    @cached_property
    def num_failed(self):
        return TestRun.objects.filter(test_run_batch__test_session=self, outcome=TestRun.Outcome.FAIL).count()

    @cached_property
    def num_error(self):
        return TestRun.objects.filter(test_run_batch__test_session=self, outcome=TestRun.Outcome.ERROR).count()

    @cached_property
    def num_total(self):
        return TestRun.objects.filter(test_run_batch__test_session=self).count()

    @cached_property
    def num_xfail(self):
        return TestRun.objects.filter(test_run_batch__test_session=self, outcome=TestRun.Outcome.XFAIL).count()

    @property
    def git_commit_url(self):
        """Generate GitHub commit URL from workflow_url if available, otherwise use default repo."""
        if self.git_commit:
            # Try to extract repo base URL from workflow_url
            # Format: https://github.com/owner/repo/actions/runs/...
            if self.workflow_url and '/actions/runs/' in self.workflow_url:
                # Extract everything before /actions/runs/
                repo_base = self.workflow_url.split('/actions/runs/')[0]
                return f"{repo_base}/commit/{self.git_commit}"
            # Fallback to default repo
            return f"https://github.com/synaptics-torq/torq-compiler-dev/commit/{self.git_commit}"
        return None

    def __str__(self):
        return f"Session #{self.id} (commit: {self.git_commit}, branch: {self.git_branch})"


class TestRunBatch(models.Model):
    test_session = models.ForeignKey(TestSession, on_delete=models.CASCADE, db_index=True)
    name = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)

    class Meta:
        verbose_name = "Test run batch"
        verbose_name_plural = "Test run batches"

    def __str__(self):
        return f"TestRunBatch #{self.id}: {self.name} for Session #{self.test_session.id}"


class Metric(models.Model):
    name = models.TextField()
    description = models.TextField(blank=True, null=True)
    unit = models.TextField()

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=['name'],
                name='unique_metric_name'
            )
        ]

    @property
    def short_description(self):
        return self.name.replace('_', ' ').capitalize()

    def __str__(self):
        return f"Metric: {self.name} [{self.unit}]"


class TestRun(models.Model):

    class Outcome(models.IntegerChoices):
        PASS = 1, 'Pass'
        FAIL = 2, 'Fail'
        SKIP = 3, 'Skip'
        ERROR = 4, 'Error'
        XFAIL = 5, 'XFail'
        NXPASS = 6, 'NXPass'

    test_run_batch = models.ForeignKey(TestRunBatch, on_delete=models.CASCADE)
    test_case = models.ForeignKey(TestCase, on_delete=models.CASCADE)
    profiling_data = models.FileField(blank=True, null=True)
    outcome = models.IntegerField(choices=Outcome.choices)
    failure_log = models.FileField(blank=True, null=True)
    failure_type = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=['test_run_batch', 'test_case'],
                name='unique_test_run_per_batch_and_case'
            )
        ]

    def __str__(self):
        return f"TestRun: {self.test_case} in Session #{self.test_run_batch.test_session.id} - {self.get_outcome_display()}"


class Measurement(models.Model):
    test_run = models.ForeignKey(TestRun, on_delete=models.CASCADE)
    metric = models.ForeignKey(Metric, on_delete=models.CASCADE)
    value = models.FloatField()

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['test_run', 'metric'], 
                                    name='unique_measurement_per_test_run_and_metric')
        ]

    def __str__(self):
        return f"Measurement: {self.metric.name} = {self.value} for {self.test_run}"
