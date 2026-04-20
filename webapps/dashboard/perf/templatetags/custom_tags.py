from django import template
from django.utils.html import format_html
from perf.models import TestRun
import json

register = template.Library()


def _format_value(value, unit):

    if value is None or value == "":
        return "N/A"
    if unit == 'ns':
        return f"{value / 1_000_000:.2f} ms"
    elif unit == '%':
        return f"{value:.2f} %"
    elif unit == 'unitless':
        if isinstance(value, int):
            return str(value)
        else:
            return f"{value:.2f}"
    else:
        return f"{value:.2f} {unit}"


@register.filter
def format_measurement(value, unit):
    return _format_value(value, unit)

def _format_difference(measurement, unit, is_regression=False):

    if measurement is None or measurement == "":
        return "N/A"
    
    sign = ""
    if measurement > 0:
        sign = "+"
        symbol = "▲"
    elif measurement < 0:
        symbol = "▼"
    else:
        symbol = "•"
    
    if (is_regression and measurement > 0) or (not is_regression and measurement < 0):                
        format_class = "text-bg-danger"
    elif (is_regression and measurement < 0) or (not is_regression and measurement > 0):        
        format_class = "text-bg-success"
    else:        
        format_class = "text-bg-secondary"

    return format_html(
        '<span class="badge rounded-pill {}">{} {}{}</span>',
        format_class,
        symbol,
        sign,
        _format_value(measurement, unit),
    )


@register.filter
def format_regression(measurement, unit="unitless"):
    return _format_difference(measurement, unit, is_regression=True)


@register.filter
def format_improvement(measurement, unit="unitless"):
    return _format_difference(measurement, unit, is_regression=False)


def _outcome_class(outcome):
    if outcome == TestRun.Outcome.PASS:
        return "text-bg-success"
    elif outcome == TestRun.Outcome.XFAIL:
        return "text-bg-warning"
    elif (outcome == TestRun.Outcome.FAIL or 
          outcome == TestRun.Outcome.ERROR):
        return "text-bg-danger"
    else:
        return "text-bg-secondary"


@register.filter
def outcomes():
    return TestRun.Outcome.choices


@register.simple_tag
def outcome_badge(outcome, value=None):

    if isinstance(outcome, str) and outcome != "":            
        outcome = TestRun.Outcome[outcome.upper()]

    if value is None and outcome is not None and outcome != "":            
        value = TestRun.Outcome(outcome).label
        
    if not value:
        value = "N/A"

    return format_html('<span class="badge {}">{}</span>', _outcome_class(outcome), value)


@register.simple_tag
def chart(data, class_name, width, height):
    return format_html('<canvas class="{}" data-histogram=\'{}\' width="{}" height="{}"></canvas>', class_name, json.dumps(data), width, height)