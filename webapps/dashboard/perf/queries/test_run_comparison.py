from ..models import TestRun


def get_test_run_comparison(test_run, baseline_test_run):
    """
    Get the details of a test run, including the metrics and the comparison with another test run if provided.
    """

    baseline_measurements_by_metric = {}

    if baseline_test_run:
        baseline_test_run = TestRun.objects.select_related('test_case', 'test_run_batch__test_session').prefetch_related('measurement_set__metric').get(id=baseline_test_run.id)
        baseline_measurements_by_metric = {
            measurement.metric_id: measurement
            for measurement in baseline_test_run.measurement_set.all()
        }

    comparison_rows = []
    for measurement in test_run.measurement_set.all().order_by('metric__name'):
        baseline_measurement = baseline_measurements_by_metric.get(measurement.metric_id)

        difference = None
        change_percent = None
        if baseline_measurement:
            difference = measurement.value - baseline_measurement.value
            if baseline_measurement.value != 0:
                change_percent = (difference * 100.0) / baseline_measurement.value

        comparison_rows.append(
            {
                'metric_name': measurement.metric.name,
                'metric_description': measurement.metric.description,
                'unit': measurement.metric.unit,
                'current_value': measurement.value,
                'baseline_value': baseline_measurement.value if baseline_measurement else None,
                'difference': difference,
                'change_percent': change_percent,
            }
        )

    total_duration_row = next(
        (row for row in comparison_rows if row['metric_name'] == 'total_duration'),
        None,
    )

    return {'comparison_rows': comparison_rows,
            'total_duration': total_duration_row}
