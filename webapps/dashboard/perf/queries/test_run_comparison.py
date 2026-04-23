from datetime import timedelta

from django.db.models import Case, IntegerField, Q, Value, When
from django.utils import timezone

from ..models import Measurement, TestRun


HISTORY_RANGE_DAYS = {
    '7': 7,
    '30': 30,
    '90': 90,
    '180': 180,
    '365': 365,
}
HISTORY_RANGE_LABELS = {
    '7': 'the last week',
    '30': 'the last month',
    '90': 'the last 3 months',
    '180': 'the last 6 months',
    '365': 'the last year',
    'all': 'all available sessions',
}
DEFAULT_HISTORY_RANGE_KEY = '30'


def group_metrics_with_comparison(metrics_with_comparison):
    grouped_rows = []
    grouped_rows_by_name = {}

    for row in metrics_with_comparison:
        metric_name = row['metric_name']

        if metric_name.endswith('_time'):
            group_name = metric_name[:-5]
            slot = 'time'
        elif metric_name.endswith('_percent'):
            group_name = metric_name[:-8]
            slot = 'percent'
        else:
            grouped_rows.append(
                {
                    'label': metric_name,
                    'single': row,
                    'time': None,
                    'percent': None,
                }
            )
            continue

        grouped_row = grouped_rows_by_name.get(group_name)
        if grouped_row is None:
            grouped_row = {
                'label': group_name,
                'single': None,
                'time': None,
                'percent': None,
            }
            grouped_rows_by_name[group_name] = grouped_row
            grouped_rows.append(grouped_row)

        grouped_row[slot] = row

    return grouped_rows


def get_test_run_comparison(test_run, baseline_test_run):
    """
    Get the details of a test run, including the metrics and the comparison with another test run if provided.
    """

    current_measurements_by_name = {
        measurement.metric.name: measurement
        for measurement in test_run.measurement_set.all()
    }

    baseline_measurements_by_name = {}

    if baseline_test_run:
        baseline_test_run = TestRun.objects.select_related('test_case', 'test_run_batch__test_session').prefetch_related('measurement_set__metric').get(id=baseline_test_run.id)
        baseline_measurements_by_name = {
            measurement.metric.name: measurement
            for measurement in baseline_test_run.measurement_set.all()
        }

    baseline_total_duration = baseline_measurements_by_name.get('total_duration')

    comparison_rows = []
    current_time_measurements = sorted(
        (
            measurement
            for measurement in current_measurements_by_name.values()
            if measurement.metric.unit == 'ns'
        ),
        key=lambda measurement: (measurement.metric.name != 'total_duration', measurement.metric.name),
    )

    for measurement in current_time_measurements:
        metric_name = measurement.metric.name
        baseline_measurement = baseline_measurements_by_name.get(metric_name)

        percent_metric_name = None
        if metric_name.endswith('_time'):
            percent_metric_name = f"{metric_name[:-5]}_percent"

        current_percent_measurement = current_measurements_by_name.get(percent_metric_name) if percent_metric_name else None
        baseline_percent_measurement = baseline_measurements_by_name.get(percent_metric_name) if percent_metric_name else None

        difference = None
        metric_change_percent = None
        if baseline_measurement:
            difference = measurement.value - baseline_measurement.value
            if baseline_measurement.value != 0:
                metric_change_percent = (difference * 100.0) / baseline_measurement.value

        potential_impact_percent = None
        if difference is not None and baseline_total_duration and baseline_total_duration.value != 0:
            potential_impact_percent = (difference * 100.0) / baseline_total_duration.value

        comparison_rows.append(
            {
                'metric_name': metric_name,
                'metric_label': metric_name[:-5] if metric_name.endswith('_time') else metric_name,
                'metric_description': measurement.metric.description,
                'unit': measurement.metric.unit,
                'current_value': measurement.value,
                'current_percent_value': current_percent_measurement.value if current_percent_measurement else None,
                'current_percent_available': current_percent_measurement is not None,
                'baseline_value': baseline_measurement.value if baseline_measurement else None,
                'baseline_percent_value': baseline_percent_measurement.value if baseline_percent_measurement else None,
                'baseline_percent_available': baseline_percent_measurement is not None,
                'difference': difference,
                'metric_change_percent': metric_change_percent,
                'potential_impact_percent': potential_impact_percent,
                'potential_impact_available': potential_impact_percent is not None,
            }
        )

    return comparison_rows


def get_corresponding_test_run(test_run, baseline_session):
    """
    Resolve the most relevant baseline test run for a given test case in another session.

    Preference order matches the session results page: passing runs first, then the latest run.
    """

    if not baseline_session:
        return None

    return (
        TestRun.objects.select_related('test_case', 'test_run_batch__test_session')
        .prefetch_related('measurement_set__metric')
        .filter(
            test_run_batch__test_session=baseline_session,
            test_case_id=test_run.test_case_id,
        )
        .exclude(outcome=TestRun.Outcome.SKIP)
        .annotate(
            pass_priority=Case(
                When(outcome=TestRun.Outcome.PASS, then=Value(1)),
                default=Value(0),
                output_field=IntegerField(),
            )
        )
        .order_by('-pass_priority', '-id')
        .first()
    )


def _default_history_metric_name(history_metrics):
    if any(metric['name'] == 'total_duration' for metric in history_metrics):
        return 'total_duration'

    return history_metrics[0]['name'] if history_metrics else None


def _coerce_history_datetime(value):
    if value is None or not timezone.is_naive(value):
        return value

    return timezone.make_aware(value, timezone.get_current_timezone())


def _quick_range_start_session_id(session_options, range_key):
    if not session_options:
        return None

    if range_key == 'all':
        return session_options[0]['session_id']

    days = HISTORY_RANGE_DAYS.get(range_key)
    if days is None:
        return None

    latest_timestamp = session_options[-1]['timestamp_value']
    cutoff_time = latest_timestamp - timedelta(days=days)

    for session_option in session_options:
        if session_option['timestamp_value'] >= cutoff_time:
            return session_option['session_id']

    return session_options[0]['session_id']


def _resolve_history_window(session_options, requested_start_date, requested_end_date):
    if not session_options:
        return {
            'active_range': None,
            'range_label': 'Showing selected session window',
            'selected_end_session_id': None,
            'selected_session_ids': set(),
            'selected_start_session_id': None,
        }

    start_date = _coerce_history_datetime(requested_start_date)
    end_date = _coerce_history_datetime(requested_end_date)

    if start_date and end_date and start_date > end_date:
        start_date, end_date = end_date, start_date

    earliest_timestamp = session_options[0]['timestamp_value']
    latest_timestamp = session_options[-1]['timestamp_value']

    if end_date is None:
        end_date = latest_timestamp
    elif end_date < earliest_timestamp:
        end_date = earliest_timestamp
    elif end_date > latest_timestamp:
        end_date = latest_timestamp

    used_default_range = False
    if start_date is None:
        if requested_end_date is None:
            start_date = latest_timestamp - timedelta(days=HISTORY_RANGE_DAYS[DEFAULT_HISTORY_RANGE_KEY])
            used_default_range = True
        else:
            start_date = earliest_timestamp
    elif start_date < earliest_timestamp:
        start_date = earliest_timestamp
    elif start_date > latest_timestamp:
        start_date = latest_timestamp

    if start_date > end_date:
        start_date = end_date

    selected_session_ids = [
        session_option['session_id']
        for session_option in session_options
        if start_date <= session_option['timestamp_value'] <= end_date
    ]

    if not selected_session_ids:
        selected_session_ids = [session_options[-1]['session_id']]

    selected_start_session_id = selected_session_ids[0]
    selected_end_session_id = selected_session_ids[-1]
    active_range = None

    if selected_end_session_id == session_options[-1]['session_id']:
        if used_default_range:
            active_range = DEFAULT_HISTORY_RANGE_KEY
        else:
            for range_key in (*HISTORY_RANGE_DAYS.keys(), 'all'):
                if selected_start_session_id == _quick_range_start_session_id(session_options, range_key):
                    active_range = range_key
                    break

    base_range_label = HISTORY_RANGE_LABELS.get(active_range, 'selected session window')
    if selected_start_session_id == selected_end_session_id:
        range_label = f"Showing {base_range_label}, session #{selected_start_session_id}"
    else:
        range_label = (
            f"Showing {base_range_label}, sessions "
            f"#{selected_start_session_id} to #{selected_end_session_id}"
        )

    return {
        'active_range': active_range,
        'range_label': range_label,
        'selected_end_session_id': selected_end_session_id,
        'selected_session_ids': set(selected_session_ids),
        'selected_start_session_id': selected_start_session_id,
    }


def get_test_run_history(test_run, baseline_test_run=None, history_options=None):
    """
    Build a historical time series for the same test case up to the current session.

    For each session, select the most relevant run with the same preference used elsewhere
    in the dashboard: explicit current/baseline runs first, then a passing run, then the
    latest run. History is limited to the current session plus historical sessions from the
    main branch.
    """

    history_options = history_options or {}

    current_session = test_run.test_run_batch.test_session
    forced_test_run_ids = [test_run.id]

    if baseline_test_run:
        forced_test_run_ids.append(baseline_test_run.id)

    current_run_measurements = test_run.measurement_set.select_related('metric').all()
    history_metric_definitions = sorted(
        (
            {
                'id': measurement.metric_id,
                'name': measurement.metric.name,
                'label': measurement.metric.short_description,
                'description': measurement.metric.description,
                'unit': measurement.metric.unit,
            }
            for measurement in current_run_measurements
            if measurement.metric.unit == 'ns'
        ),
        key=lambda metric: (
            metric['name'] != 'total_duration',
            metric['label'],
        ),
    )
    history_metrics = [
        {
            'name': metric['name'],
            'label': metric['label'],
            'description': metric['description'],
            'unit': metric['unit'],
        }
        for metric in history_metric_definitions
    ]
    history_metrics_by_name = {
        metric['name']: metric
        for metric in history_metric_definitions
    }
    default_metric_name = _default_history_metric_name(history_metrics)
    selected_metric_name = history_options.get('metric')
    if selected_metric_name not in history_metrics_by_name:
        selected_metric_name = default_metric_name

    selected_metric_definition = history_metrics_by_name.get(selected_metric_name)
    selected_metric = None
    if selected_metric_definition is not None:
        selected_metric = {
            'name': selected_metric_definition['name'],
            'label': selected_metric_definition['label'],
            'description': selected_metric_definition['description'],
            'unit': selected_metric_definition['unit'],
        }

    history_branch_filters = (
        Q(test_run_batch__test_session__git_branch='refs/heads/main')
        | Q(test_run_batch__test_session_id=current_session.id)
    )

    selected_candidates = list(
        TestRun.objects.select_related('test_run_batch__test_session')
        .filter(
            history_branch_filters,
            test_case_id=test_run.test_case_id,
            test_run_batch__test_session__timestamp__lte=current_session.timestamp,
        )
        .exclude(outcome=TestRun.Outcome.SKIP)
        .annotate(
            forced_priority=Case(
                When(id__in=forced_test_run_ids, then=Value(1)),
                default=Value(0),
                output_field=IntegerField(),
            ),
            pass_priority=Case(
                When(outcome=TestRun.Outcome.PASS, then=Value(1)),
                default=Value(0),
                output_field=IntegerField(),
            ),
        )
        .order_by(
            'test_run_batch__test_session_id',
            '-forced_priority',
            '-pass_priority',
            '-id',
        )
        .distinct('test_run_batch__test_session_id')
    )

    selected_candidates.sort(
        key=lambda candidate: (
            candidate.test_run_batch.test_session.timestamp,
            candidate.test_run_batch.test_session_id,
            candidate.id,
        )
    )

    session_options = []
    for candidate in selected_candidates:
        session = candidate.test_run_batch.test_session
        session_options.append(
            {
                'session_id': session.id,
                'timestamp': session.timestamp.isoformat(),
                'timestamp_label': session.timestamp.strftime('%b %d, %Y'),
                'timestamp_value': session.timestamp,
                'is_current': candidate.id == test_run.id,
                'is_baseline': baseline_test_run is not None and candidate.id == baseline_test_run.id,
            }
        )

    history_window = _resolve_history_window(
        session_options,
        history_options.get('start_date'),
        history_options.get('end_date'),
    )
    filtered_candidates = [
        candidate
        for candidate in selected_candidates
        if candidate.test_run_batch.test_session_id in history_window['selected_session_ids']
    ]

    selected_run_ids = [candidate.id for candidate in filtered_candidates]
    if selected_metric_definition is not None and selected_run_ids:
        selected_run_measurements = Measurement.objects.filter(
            test_run_id__in=selected_run_ids,
            metric_id=selected_metric_definition['id'],
        )
    else:
        selected_run_measurements = Measurement.objects.none()

    measurements_by_run_id = {
        measurement.test_run_id: measurement.value
        for measurement in selected_run_measurements
    }

    points = []
    for candidate in filtered_candidates:
        session = candidate.test_run_batch.test_session
        points.append(
            {
                'session_id': session.id,
                'test_run_id': candidate.id,
                'timestamp': session.timestamp.isoformat(),
                'value': measurements_by_run_id.get(candidate.id),
                'outcome': candidate.get_outcome_display(),
                'outcome_value': candidate.outcome,
                'git_branch': session.git_branch or '',
                'git_commit': session.git_commit or '',
                'workflow_url': session.workflow_url or '',
                'is_current': candidate.id == test_run.id,
                'is_baseline': baseline_test_run is not None and candidate.id == baseline_test_run.id,
            }
        )

    return {
        'active_range': history_window['active_range'],
        'kind': 'single_metric',
        'meta_text': (
            'Best available run per session for this test case using the current session '
            f'plus main-branch history up to session #{current_session.id}.'
        ),
        'metrics': history_metrics,
        'default_metric': default_metric_name,
        'points': points,
        'range_label': history_window['range_label'],
        'selected_end_session_id': history_window['selected_end_session_id'],
        'selected_metric': selected_metric,
        'selected_metric_name': selected_metric_name,
        'selected_start_session_id': history_window['selected_start_session_id'],
        'session_options': [
            {
                'session_id': session_option['session_id'],
                'timestamp': session_option['timestamp'],
                'timestamp_label': session_option['timestamp_label'],
                'is_current': session_option['is_current'],
                'is_baseline': session_option['is_baseline'],
            }
            for session_option in session_options
        ],
    }
