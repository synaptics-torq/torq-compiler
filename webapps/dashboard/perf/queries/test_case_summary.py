
from ..models import Measurement, RuntimeTarget, TestGroup


def query_test_durations(
    session_ids,
    group_name,
    runtime_target=None,
    runtime_hw_type=None,
    compiler_input=None,
):
    """Query total_duration measurements from the database for tests in the selected group and sessions.

    Returns a flat list of dicts, one per measurement.
    """

    try:
        test_group = TestGroup.objects.get(name=group_name)
    except TestGroup.DoesNotExist:
        return []

    test_cases = test_group.test_cases.all()

    measurements = Measurement.objects.filter(
        test_run__test_case__in=test_cases,
        test_run__test_run_batch__test_session_id__in=session_ids,
        metric__name='total_duration',
    )

    if runtime_target:
        measurements = measurements.filter(test_run__test_metadata__runtime_target=runtime_target)

    if runtime_hw_type:
        measurements = measurements.filter(test_run__test_metadata__runtime_hw_type=runtime_hw_type)

    if compiler_input:
        measurements = measurements.filter(test_run__test_metadata__compiler_input=compiler_input)

    measurements = measurements.select_related(
        'test_run',
        'test_run__test_case',
        'test_run__test_metadata',
        'metric',
        'test_run__test_run_batch__test_session',
    )

    runtime_target_names = {
        m.test_run.test_metadata.runtime_target
        for m in measurements
        if m.test_run.test_metadata and m.test_run.test_metadata.runtime_target
    }

    runtime_targets = RuntimeTarget.objects.filter(name__in=runtime_target_names)
    runtime_target_by_key = {
        (target.name, target.hw_type): target
        for target in runtime_targets
    }

    rows = []
    for measurement in measurements:
        metadata = measurement.test_run.test_metadata

        if metadata is None:
            continue

        rt_name = metadata.runtime_target
        hw_type = metadata.runtime_hw_type
        runtime_target_model = runtime_target_by_key.get((rt_name, hw_type))

        total_duration = measurement.value

        if runtime_target_model and runtime_target_model.inference_frequency:
            scaling_factor = runtime_target_model.inference_frequency / 1_000_000_000.0
            normalized_duration = total_duration * scaling_factor
        else:
            normalized_duration = None

        rows.append({
            'compiler_input': metadata.compiler_input,
            'node_id': measurement.test_run.test_case.nodeid,
            'total_duration': total_duration,
            'normalized_duration': normalized_duration,
            'test_run_id': measurement.test_run.id,
            'runtime': metadata.runtime,
            'runtime_target': rt_name,
            'runtime_hw_type': hw_type,
            'git_branch': measurement.test_run.test_run_batch.test_session.git_branch,
            'git_commit': measurement.test_run.test_run_batch.test_session.git_commit,
            'inference_frequency': runtime_target_model.inference_frequency if runtime_target_model else None,
            'memory_bandwidth': runtime_target_model.memory_bandwidth if runtime_target_model else None,
            'cache_size': runtime_target_model.cache_size if runtime_target_model else None,
        })

    return rows


def get_reference_test_durations(session_ids):
    """Find total_duration for tests in the selected group and sessions, grouped by compiler_input."""

    rows = query_test_durations(session_ids, group_name="home")

    result = {}
    for row in rows:
        key = row['compiler_input']
        if key not in result:
            result[key] = []
        result[key].append(row)

    for key in result:
        result[key].sort(key=lambda x: (x['runtime'] or '', x['node_id']))

    return dict(sorted(result.items(), key=lambda item: item[0] or ''))




