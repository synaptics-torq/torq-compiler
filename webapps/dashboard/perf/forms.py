import django.forms as forms
from .models import TestSession, TestRun, TestGroup, Metric


class TestGroupChoiceField(forms.ModelChoiceField):

    def __init__(self, **kwargs):
        super().__init__(queryset=TestGroup.objects.order_by('name').all(), required=False, empty_label="All", **kwargs)
        
    def label_from_instance(self, obj):
        return obj.name


class TestSessionChoiceField(forms.ModelChoiceField):
    
    def __init__(self, **kwargs):
        super().__init__(queryset=TestSession.objects.order_by('-timestamp').all(), empty_label="None", **kwargs)
        
    def label_from_instance(self, obj):
        return f"#{obj.id} {obj.git_branch} [test plan: {obj.test_plan}, timestamp: {obj.timestamp.strftime('%Y-%m-%d %H:%M')}]"


class MetricChoiceField(forms.ModelChoiceField):

    def __init__(self, **kwargs):
        queryset = kwargs.pop('queryset', Metric.objects.order_by('priority', 'name').all())
        super().__init__(
            queryset=queryset,
            to_field_name='name',
            **kwargs,
        )

    def label_from_instance(self, obj):
        return f"{obj.short_description} ({obj.unit})"


class BaseBootstrapForm(forms.Form):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for name, field in self.fields.items():
            widget = field.widget
            classes = widget.attrs.get("class", "").split()

            if isinstance(widget, forms.CheckboxInput):
                classes.append("form-check-input")
            elif isinstance(widget, (forms.Select, forms.SelectMultiple)):
                classes.append("form-select")
            elif not isinstance(widget, forms.RadioSelect):
                classes.append("form-control")

            if self.errors.get(name):
                classes.append("is-invalid")

            widget.attrs["class"] = " ".join(sorted(set(classes)))


STATUS_FILTER_CHOICES = [("ALL", "all")] + [(str(x), y) for x, y in TestRun.Outcome.choices]
COMPARISON_TRANSITION_CHOICES = [
    ("ALL", "all"),
    ("FAIL_TO_PASS", "fail to pass"),
    ("PASS_TO_FAIL", "pass to fail"),
    ("ERROR_TO_PASS", "error to pass"),
    ("PASS_TO_XFAIL", "pass to xfail"),
    ("ANY_REGRESSION", "any regression"),
    ("ANY_IMPROVEMENT", "any improvement"),
]
HISTORY_DATETIME_INPUT_FORMATS = [
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
]

class TestSessionSummaryOptions(BaseBootstrapForm):
    baseline_session = TestSessionChoiceField(required=False, label="Compare with session")


class TestSessionMetricDetailsOptions(BaseBootstrapForm):
    baseline_session = TestSessionChoiceField(required=True, label="Compare with session")
    metric = MetricChoiceField(required=True, label="Select metric")
    cutoff_threshold = forms.FloatField(required=False, label="Cutoff threshold for highlighting significant changes (leave blank for default cutoff)", widget=forms.NumberInput(attrs={"step": "any"}))
    number_of_top = forms.IntegerField(required=False, label="Number of top regressions/improvements to display", min_value=1)


class TestSessionResultsOptions(BaseBootstrapForm):
    baseline_session = TestSessionChoiceField(required=False, label="Compare with session")
    metric = MetricChoiceField(required=True, label="Metric", queryset=Metric.objects.filter(unit="ns").order_by("priority", "name"))
    nodeid = forms.CharField(required=False, label="Filter by Test Name")    
    status = forms.ChoiceField(choices=STATUS_FILTER_CHOICES, required=False, label="Filter by status in current session")
    comparison_transition = forms.ChoiceField(choices=COMPARISON_TRANSITION_CHOICES, required=False, label="Filter by comparison transition")
    sort_by = forms.ChoiceField(choices=[("nodeid", "Test Name"), ("-nodeid", "Test Name ↓"), ("current_value", "Metric ↑"), ("-current_value", "Metric ↓"), ("baseline_value", "Baseline Metric ↑"), ("-baseline_value", "Baseline Metric ↓"), ("change_percent", "Change Percent ↑"), ("-change_percent", "Change Percent ↓")], required=False, label="Sort by", widget=forms.HiddenInput)
    page = forms.IntegerField(required=False, min_value=1, initial=1, widget=forms.HiddenInput)
    external_engine = forms.CharField(required=False)


class TestRunOptions(BaseBootstrapForm):
    baseline_session = TestSessionChoiceField(required=False, label="Compare with session", widget=forms.HiddenInput)
    baseline_test_run = forms.ModelChoiceField(queryset=TestRun.objects.all(), required=False, empty_label="None", widget=forms.HiddenInput)
    metric = forms.CharField(required=False, widget=forms.HiddenInput)
    start_date = forms.DateTimeField(required=False, widget=forms.HiddenInput, input_formats=HISTORY_DATETIME_INPUT_FORMATS)
    end_date = forms.DateTimeField(required=False, widget=forms.HiddenInput, input_formats=HISTORY_DATETIME_INPUT_FORMATS)
