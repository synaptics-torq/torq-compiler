
from ..models import Measurement, RuntimeTarget, TestGroup


def get_reference_test_durations(session_ids):
    """Finds the metric "total_duration" for the tests in the 'home' group for the given sessions"""

    # Get the 'home' test group
    try:
        home_group = TestGroup.objects.get(name='home')
    except TestGroup.DoesNotExist:
        return {}

    # Get all test cases in the 'home' group
    test_cases = home_group.test_cases.all()

    # Query measurements for these test cases in the given sessions
    measurements = (Measurement.objects
        .filter(
            test_run__test_case__in=test_cases,
            test_run__test_run_batch__test_session_id__in=session_ids,
            metric__name='total_duration'
        )
        .select_related('test_run', 'test_run__test_case', 'test_run__test_metadata', 'metric', 'test_run__test_run_batch__test_session')
    )

    runtime_target_names = {
        measurement.test_run.test_metadata.runtime_target
        for measurement in measurements
        if measurement.test_run.test_metadata and measurement.test_run.test_metadata.runtime_target
    }

    runtime_targets = RuntimeTarget.objects.filter(name__in=runtime_target_names)
    runtime_target_by_key = {
        (target.name, target.hw_type): target
        for target in runtime_targets
    }

    runtime_target_by_name = {target.name: target for target in runtime_targets}

    result = {}

    for measurement in measurements:
        metadata = measurement.test_run.test_metadata
        compiler_input = metadata.compiler_input if metadata else None
        runtime = metadata.runtime if metadata else None
        runtime_target = metadata.runtime_target if metadata else None
        runtime_hw_type = metadata.runtime_hw_type if metadata else None
        runtime_target_model = runtime_target_by_key.get((runtime_target, runtime_hw_type)) if runtime_target else None
        if runtime_target and runtime_target_model is None:
            runtime_target_model = runtime_target_by_name.get(runtime_target)
        nodeid = measurement.test_run.test_case.nodeid
        total_duration = measurement.value
        run_id = measurement.test_run.id
        git_branch = measurement.test_run.test_run_batch.test_session.git_branch

        if runtime_target_model and runtime_target_model.inference_frequency:            
            scaling_factor = runtime_target_model.inference_frequency / 1_000_000_000.0
            normalized_duration = total_duration * scaling_factor
        else:
            normalized_duration = None

        if compiler_input not in result:
            result[compiler_input] = []

        result[compiler_input].append({'node_id': nodeid, 'total_duration': total_duration, 
                                       'test_run_id': run_id, 'runtime': runtime,
                                       'runtime_target': runtime_target, 'git_branch': git_branch,
                                       'runtime_hw_type': runtime_hw_type,                                    
                                       'inference_frequency': runtime_target_model.inference_frequency if runtime_target_model else None,
                                       'normalized_duration': normalized_duration,
                                       'cache_size': runtime_target_model.cache_size if runtime_target_model else None
                                       })

    return result




