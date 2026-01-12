import os
from django.db import models
from django.utils.text import slugify


class TestCase(models.Model):
    module = models.TextField()
    name = models.TextField()
    parameters = models.TextField()

    def __str__(self):
        return f"{self.module}::{self.name}[{self.parameters}]"


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

    @property
    def num_passed(self):
        return self.testrun_set.filter(outcome=TestRun.Outcome.PASS).count()

    @property
    def num_skipped(self):
        return self.testrun_set.filter(outcome=TestRun.Outcome.SKIP).count()

    @property
    def num_failed(self):
        return self.testrun_set.filter(outcome=TestRun.Outcome.FAIL).count()

    @property
    def num_total(self):
        return self.testrun_set.count()

    def __str__(self):
        return f"Session #{self.id} (commit: {self.git_commit}, branch: {self.git_branch})"


def profiling_data_upload_path(instance, filename):
    """Generate a custom path for profiling data files."""
    # Extract file extension
    ext = os.path.splitext(filename)[1]
    
    # Create a naming convention: session_id/testcase_module_name_params.ext
    test_case = instance.test_case
    session_id = instance.test_session.id
    
    # Sanitize the components
    module = slugify(test_case.module)
    name = slugify(test_case.name)
    params = slugify(test_case.parameters)
    
    # Build the filename
    new_filename = f"{module}_{name}_{params}{ext}"
    
    return f"profiling_data/session_{session_id}/{new_filename}"


class Metric(models.Model):
    name = models.TextField()
    description = models.TextField(blank=True, null=True)
    unit = models.TextField()

    def __str__(self):
        return f"Metric: {self.name} [{self.unit}]"


class TestRun(models.Model):

    class Outcome(models.IntegerChoices):
        PASS = 1, 'Pass',
        FAIL = 2, 'Fail'
        SKIP = 3, 'Skip'

    test_session = models.ForeignKey(TestSession, on_delete=models.CASCADE)
    test_case = models.ForeignKey(TestCase, on_delete=models.CASCADE)
    profiling_data = models.FileField(blank=True, null=True, upload_to=profiling_data_upload_path)
    outcome = models.IntegerField(choices=Outcome.choices)

    def __str__(self):
        return f"TestRun: {self.test_case} in Session #{self.test_session.id} - {self.get_outcome_display()}"


class Measurement(models.Model):
    test_run = models.ForeignKey(TestRun, on_delete=models.CASCADE)
    metric = models.ForeignKey(Metric, on_delete=models.CASCADE)
    value = models.FloatField()

    def __str__(self):
        return f"Measurement: {self.metric.name} = {self.value} for {self.test_run}"
