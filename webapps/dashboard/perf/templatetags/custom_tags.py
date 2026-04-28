from django import template
from django.utils.html import format_html
from perf.models import TestRun
import json

register = template.Library()


def _format_value(value, unit):

    if value is None or value == "":
        return "N/A"
    if unit == 'ns':
        absolute_value = abs(value)
        if absolute_value >= 1_000_000:
            return f"{value / 1_000_000:.2f} ms"
        if absolute_value >= 1_000:
            return f"{value / 1_000:.2f} us"
        if float(value).is_integer():
            return f"{int(value)} ns"
        return f"{value:.2f} ns"
    elif unit == 'Hz':
        if value >= 1_000_000_000:
            return f"{value / 1_000_000_000:.2f} GHz"
        if value >= 1_000_000:
            return f"{value / 1_000_000:.2f} MHz"
        if value >= 1_000:
            return f"{value / 1_000:.2f} kHz"
        if float(value).is_integer():
            return f"{int(value)} Hz"
        return f"{value:.2f} Hz"
    elif unit == '%':
        return f"{value:.2f} %"
    elif unit == 'unitless':
        if isinstance(value, int):
            return str(value)
        else:
            return f"{value:.2f}"
    else:
        return f"{value:.2f} {unit}"


def _difference_parts(measurement, unit, is_regression=False):

    if measurement is None or measurement == "":
        return None

    sign = ""
    if measurement > 0:
        sign = "+"
        symbol = "▲"
    elif measurement < 0:
        symbol = "▼"
    else:
        symbol = "•"

    if (is_regression and measurement > 0) or (not is_regression and measurement < 0):
        badge_class = "text-bg-danger"
        text_class = "text-danger"
    elif (is_regression and measurement < 0) or (not is_regression and measurement > 0):
        badge_class = "text-bg-success"
        text_class = "text-success"
    else:
        badge_class = "text-bg-secondary"
        text_class = "text-body-secondary"

    return {
        "badge_class": badge_class,
        "text_class": text_class,
        "symbol": symbol,
        "sign": sign,
        "formatted_value": _format_value(measurement, unit),
    }


@register.filter
def format_measurement(value, unit):
    return _format_value(value, unit)

def _format_difference(measurement, unit, is_regression=False):

    parts = _difference_parts(measurement, unit, is_regression=is_regression)

    if parts is None:
        return "N/A"

    return format_html(
        '<span class="badge rounded-pill {}">{} {}{}</span>',
        parts["badge_class"],
        parts["symbol"],
        parts["sign"],
        parts["formatted_value"],
    )


def _format_difference_inline(measurement, unit, is_regression=False):
    parts = _difference_parts(measurement, unit, is_regression=is_regression)

    if parts is None:
        return format_html('<span class="text-body-secondary">{}</span>', 'N/A')

    return format_html(
        '<span class="fw-semibold {}">{} {}{}</span>',
        parts["text_class"],
        parts["symbol"],
        parts["sign"],
        parts["formatted_value"],
    )


@register.filter
def format_regression(measurement, unit="unitless"):
    return _format_difference(measurement, unit, is_regression=True)


@register.filter
def format_regression_inline(measurement, unit="unitless"):
    return _format_difference_inline(measurement, unit, is_regression=True)


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
        
    if value is None or value == "":
        value = "N/A"

    return format_html('<span class="badge {}">{}</span>', _outcome_class(outcome), value)


@register.simple_tag
def chart(data, class_name, width, height):
    return format_html('<canvas class="{}" data-chart=\'{}\' width="{}" height="{}"></canvas>', class_name, json.dumps(data), width, height)


@register.inclusion_tag('perf/partials/timeseries_chart.html')
def timeseries_chart(timeseries, chart_id, title, header_note='', meta_text='', form=None):
    return {
        'chart_id': chart_id,
        'canvas_id': f'{chart_id}-canvas',
        'form': form,
        'header_note': header_note,
        'json_script_id': f'{chart_id}-data',
        'meta_text': meta_text,
        'timeseries': timeseries,
        'title': title,
    }


@register.simple_tag
def issue_link(issue_name):
    if not issue_name:
        return ""
    
    repo, id = issue_name.split("#")

    return format_html('<a href="https://github.com/{}/issues/{}" target="_blank" rel="noopener noreferrer">{}</a>', repo, id, issue_name)
