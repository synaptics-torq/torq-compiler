from django.db import models
from django.utils.functional import cached_property


class TestCase(models.Model):
    """
    Represents a given test case that can be run during a TestSession.

    This is maps to a pytest test case.

    """

    """ Name of the module containing the test case, typically corresponding to the pytest file name."""
    module = models.TextField()

    """ Name of the test case, typically corresponding to the pytest test function name."""
    name = models.TextField()
    
    """ Paramters string of the test case as defined by pytest"""
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


class TestMetadata(models.Model):
    """
    Tracks the information about what was tested in a given test case, 
    this may evolve over time so we keep it separate from the TestCase model.

    For instance the compiler options may change over time as the compiler
    changes, but we want to be able to track the evolution of performance
    for the same test case (same input model, same target hardware, etc.) over time.
    """
    
    """ Identifier of the compiler used for this test case (e.g. torq or an alternative engine)"""
    compiler = models.TextField(blank=True, null=True)

    """ Identifier of the input program (e.g. tflite model) used for this test case. """
    compiler_input = models.TextField(blank=True, null=True)

    """ Target hardware for which the input is compiler for (e.g. sl2610 or some other hardware)"""
    compiler_target = models.TextField(blank=True, null=True)

    """ Additional compiler options used for this test case (e.g. optimization level, tiling parameters, etc.)"""
    compiler_options = models.TextField(blank=True, null=True)

    """ Identifier of the runtime used for this test case (e.g. torq or an alternative engine)"""
    runtime = models.TextField(blank=True, null=True)

    """ Target hardware for which the runtime is executed (e.g. sl2610 or some other hardware)"""
    runtime_target = models.TextField(blank=True, null=True)

    """ Type of hardware used for execution (e.g. simulation, FPGA, eval board, etc.)"""
    runtime_hw_type = models.TextField(blank=True, null=True)

    """ Additional runtime options used for this test case (e.g. runtime performance flags, etc.)"""
    runtime_options = models.TextField(blank=True, null=True)

    """ Identifier of the runtime input used for this test case (e.g. input data files or data generation function)"""
    runtime_input = models.TextField(blank=True, null=True)

    class Metadata:
        constraints = [
            models.UniqueConstraint(
                fields=['compiler', 'compiler_input', 'compiler_target', 'compiler_options', 
                        'runtime', 'runtime_target', 'runtime_hw_type', 'runtime_options', 'runtime_input'],
                name='unique_test_metadata'
            )
        ]


class TestGroup(models.Model):
    """
    Represents a group of test cases.

    This is used to show tests grouped by some common characteristic (e.g. important models, etc.) in the dashboard.
    """
    name = models.TextField(unique=True)
    test_cases = models.ManyToManyField(TestCase)

    def __str__(self):
        return f"TestGroup: {self.name}"


class TestSession(models.Model):
    """
    Represents a group of pytest invocations that were executed on
    a given code base (identified by git commit and branch) 
    and possibly associated with a CI workflow run (identified by workflow_url).

    It is possible to have no git_commit/branch/workflow_url if the 
    test session was created manually (e.g. for local runs).

    For tests with a workflow_url, we enforce uniqueness to 
    prevent duplicate sessions objects for the same CI run.

    There is roughly a 1:1 relationship between TestSession and 
    CI workflow runs (the main exception are local runs which).

    """

    """Timestamp when the test session was created."""
    timestamp = models.DateTimeField(auto_now_add=True)

    """Owner of the test session (github user name)."""
    owner = models.TextField(blank=True, null=True)

    """Git commit of the torq codebase associated with the test session."""
    git_commit = models.TextField(blank=True, null=True)

    """Git branch of the torq codebase associated with the test session."""
    git_branch = models.TextField(blank=True, null=True)

    """URL of the CI workflow associated with the test session."""
    workflow_url = models.TextField(blank=True, null=True)

    """ Flag to indicate whether the test session has completed and all its batches have been uploaded."""
    completed = models.BooleanField(default=False)

    """ Optional field to specify the test plan associated with the session (e.g. "torq", "alternative_engines", ...)."""
    test_plan = models.TextField(blank=True, null=True)

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
    """
    Represents a single run of pytest in a TestSession. A TestSession can have multiple 
    TestRunBatch entries if pytest was run multiple times (e.g. for different test
    subsets). Runs are executed in batches because different batches may require
    different hardware resources (e.g. FPGA instances, physical boards, etc.).

    There is roughly a 1:1 relationship between TestRunBatch and CI workflow jobs (in 
    case of matrix jobs there is a test run batch for each parameter combination).
    """

    test_session = models.ForeignKey(TestSession, on_delete=models.CASCADE, db_index=True)

    """ Optional name for the test run batch (e.g. "cmodel-chip1", "fpga-chip2", etc.)"""
    name = models.TextField(blank=True, null=True)

    """ Timestamp when the test run batch was created."""
    created_at = models.DateTimeField(auto_now_add=True)

    """ Flag to indicate whether the test run batch has been processed and its results are available for querying."""
    processed = models.BooleanField(default=False)

    class Meta:
        verbose_name = "Test run batch"
        verbose_name_plural = "Test run batches"

    def __str__(self):
        return f"TestRunBatch #{self.id}: {self.name} for Session #{self.test_session.id}"


class Metric(models.Model):
    """
    Represent a performance metric that can be measured for a test run,
    such as execution time, memory usage, etc.

    Metrics are stored as separate objects to allow for flexible addition of new metrics
    without having to change the database schema.
    """

    """Identifier of the metric (e.g. "execution_time", "memory_usage", etc.)"""
    name = models.TextField(unique=True)

    """Human readable description of the metric (e.g. "Execution time in seconds")"""
    description = models.TextField(blank=True, null=True)

    """ Unit of the metric (e.g. "seconds", "MB", etc.)"""
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
    """
    Represents a single run of a test case within a TestRunBatch. 
    
    It has an outcome (pass/fail/skip) and associated profiling data 
    (performance traces and metrics).
    """

    class Outcome(models.IntegerChoices):
        PASS = 1, 'Pass'
        FAIL = 2, 'Fail'
        SKIP = 3, 'Skip'
        ERROR = 4, 'Error'
        XFAIL = 5, 'XFail'
        NXPASS = 6, 'NXPass'
    
    test_run_batch = models.ForeignKey(TestRunBatch, on_delete=models.CASCADE)
    test_case = models.ForeignKey(TestCase, on_delete=models.CASCADE)
    test_metadata = models.ForeignKey(TestMetadata, on_delete=models.CASCADE, blank=True, null=True)

    """ File field to store detailed profiling data associated with the test run."""
    profiling_data = models.FileField(blank=True, null=True)

    """ Outcome of the test run (pass/fail/skip)."""
    outcome = models.IntegerField(choices=Outcome.choices)

    """ Optional file field to store failure logs or error messages for failed test runs."""
    failure_log = models.FileField(blank=True, null=True)

    """ Optional field with a summary of the failure error (e.g. compilation error, runtime error, correctness failure, etc.)"""
    failure_type = models.CharField(max_length=50, blank=True, null=True)
    linked_issue = models.TextField(blank=True, null=True)

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
    """
    Represents a single measurement of a performance metric for a given TestRun.
    """

    test_run = models.ForeignKey(TestRun, on_delete=models.CASCADE)
    metric = models.ForeignKey(Metric, on_delete=models.CASCADE)

    """ Value of the measurement for the given metric and test run."""
    value = models.FloatField()

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['test_run', 'metric'], 
                                    name='unique_measurement_per_test_run_and_metric')
        ]

    def __str__(self):
        return f"Measurement: {self.metric.name} = {self.value} for {self.test_run}"


class RuntimeTarget(models.Model):
    """
    Represents a runtime target (e.g. specific hardware platform or simulator) that can be used for executing test runs.

    This is used to track the medata about different hardware targets that are being tested in the dashboard.
    """

    """ Identifier of the runtime target (e.g. "sl2610", etc.)"""
    name = models.TextField(unique=True)

    """ Type of hardware used for execution (e.g. simulation, FPGA, eval board, etc.)"""
    hw_type = models.TextField(blank=True, null=True)

    """ Human readable description of the runtime target."""
    description = models.TextField(blank=True, null=True)

    """ Inference Frequency """
    inference_frequency = models.IntegerField(blank=True, null=True)

    """ Cache size in MB """
    cache_size = models.IntegerField(blank=True, null=True)

    def __str__(self):
        return f"RuntimeTarget: {self.name} ({self.hw_type})"