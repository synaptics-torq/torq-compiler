import django.forms as forms
from .models import TestSession, TestRun, TestGroup


class TestGroupChoiceField(forms.ModelChoiceField):

    def __init__(self, **kwargs):
        super().__init__(queryset=TestGroup.objects.order_by('name').all(), required=False, empty_label="All", **kwargs)
        
    def label_from_instance(self, obj):
        return obj.name


class TestSessionChoiceField(forms.ModelChoiceField):
    
    def __init__(self, **kwargs):
        super().__init__(queryset=TestSession.objects.order_by('-timestamp').all(), empty_label="None", **kwargs)
        
    def label_from_instance(self, obj):
        return f"#{obj.id} {obj.git_branch} ({obj.timestamp.strftime('%Y-%m-%d %H:%M')})"


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


class TestSessionSummaryOptions(BaseBootstrapForm):
    baseline_session = TestSessionChoiceField(required=False, label="Compare with session")


class TestSessionResultsOptions(BaseBootstrapForm):
    baseline_session = TestSessionChoiceField(required=False, label="Compare with session")
    nodeid = forms.CharField(required=False, label="Filter by Test Name")    
    status = forms.ChoiceField(choices=STATUS_FILTER_CHOICES, required=False, label="Filter by status in current session")
    baseline_status = forms.ChoiceField(choices=STATUS_FILTER_CHOICES, required=False, label="Filter by status in comparison session")
    sort_by = forms.ChoiceField(choices=[("nodeid", "Test Name"), ("current_value", "Duration ↑"), ("-current_value", "Duration ↓"), ("change_percent", "Change Percent ↑"), ("-change_percent", "Change Percent ↓")], required=False, label="Sort by")
    page = forms.IntegerField(required=False, min_value=1, initial=1, widget=forms.HiddenInput)


class TestRunOptions(BaseBootstrapForm):
    baseline_test_run = forms.ModelChoiceField(queryset=TestRun.objects.all(), required=False, empty_label="None", label="Compare with test run", widget=forms.HiddenInput)