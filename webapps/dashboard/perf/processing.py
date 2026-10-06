import os
import zipfile
import json
from datetime import timedelta
from tempfile import TemporaryDirectory
from django.core.files.base import File
from django.core.files.storage import default_storage
from django.db import transaction
from django.utils import timezone

# conditionally import uwsgi spool decorator to make sure this module can be imported also in manage.py
try:
    import uwsgi
    from uwsgidecorators import spool
except ImportError:
    def spool(func):
        return func
    
from perf.models import TestCase, TestRun, TestRunBatch, TestMetadata, Metric, Measurement, TestSession
from perf.trim_status import TrimStatus, get_trim_status, save_trim_status, timestamp


def _classify_failure(failed_phase, failure_log):
    """Classify a test failure based on the phase and error content.

    Returns a short failure type string:
      - 'Compilation Error'  - compiler crashed or returned non-zero during setup
      - 'Compilation Timeout' - compiler timed out
      - 'Runtime Error'      - runtime binary crashed
      - 'Runtime Timeout'    - runtime timed out
      - 'Output Mismatch'    - numeric comparison failure
      - 'Assertion Error'    - other assertion failures in the test body
      - 'Error'              - anything else
    """
    if not failure_log:
        if failed_phase == 'setup':
            return 'Compilation Error'
        return 'Error'

    log = failure_log

    # Timeout detection (independent of phase)
    if 'TimeoutExpired' in log or 'timed out' in log.lower():
        if failed_phase == 'setup':
            return 'Compilation Timeout'
        return 'Runtime Timeout'

    # Setup phase failures are compilation errors
    if failed_phase == 'setup':
        return 'Compilation Error'

    # Call phase: classify from error content
    # Output mismatch patterns from comparison.py
    if 'Number of differences:' in log or 'Nans differ' in log or 'Output is 0 always' in log or 'Number of outputs differ:' in log:
        return 'Output Mismatch'

    # Runtime crash (subprocess failure in the call phase, not an assertion)
    if 'CalledProcessError' in log and 'AssertionError' not in log:
        return 'Runtime Error'

    # Generic assertion
    if 'AssertionError' in log:
        return 'Assertion Error'

    return 'Error'


def create_metrics(metric_names):
    """Create Metric objects for the given list of metric names, if they don't already exist."""
    metrics = {}

    for metric in metric_names:                
        metric, _ = Metric.objects.get_or_create(name=metric['name'], defaults={'unit': metric['unit'], 'description': metric['description']})                
        metrics[metric.name] = metric

    return metrics


@spool
def process_uploaded_zip(args):
    """
    Spooled function to process test runs from uploaded zip files.

    This is processed in background thanks to the @spool decorator to avoid having a timeout on upload.
    
    Args:
        args: Dictionary containing:
            - zip_path: Path to the uploaded zip file
            - test_run_batch_id: ID of the pre-created test run batch
    """

    zip_path = args['zip_path']
    test_run_batch_id = int(args['test_run_batch_id'])
    
    metrics = {}
    
    try:
        # Get the test run batch
        test_run_batch = TestRunBatch.objects.get(id=test_run_batch_id)
        
        with TemporaryDirectory() as temp_dir:
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            # Find JSON manifest named test_session.json
            manifest_path = os.path.join(temp_dir, 'test_session.json')
            
            if not os.path.exists(manifest_path):
                raise ValueError('Required manifest test_session.json not found in archive.')

            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)

            metrics = create_metrics(manifest.get('metrics', []))

            outcome_map = {
                'passed': TestRun.Outcome.PASS,
                'failed': TestRun.Outcome.FAIL,
                'skipped': TestRun.Outcome.SKIP,
                'error': TestRun.Outcome.ERROR,
                'xfail': TestRun.Outcome.XFAIL,
                'nxpass': TestRun.Outcome.NXPASS,
            }

            test_runs = manifest.get('test_runs', [])
            
            for item in test_runs:
                try:
                    module = item.get('module', '')
                    name = item.get('name', '')
                    parameters = item.get('parameters', '')
                    outcome = item.get('outcome', '')
                    linked_issue = item.get('linked_issue', None)  # optional linked issue
                    profiling_rel = item.get('profiling_file')  # relative path inside zip (optional)
                    
                    if not module or not name or outcome not in outcome_map:
                        # Log error but continue processing other test runs
                        print(f'Invalid run entry in manifest: module={module}, name={name}, outcome={outcome}')
                        continue
                    
                    test_case, _ = TestCase.objects.get_or_create(
                        module=str(module), name=str(name), parameters=str(parameters)
                    )

                    test_metadata, _ = TestMetadata.objects.get_or_create(
                        compiler=item.get('metadata', {}).get('compiler'),
                        compiler_input=item.get('metadata', {}).get('compiler_input'),
                        compiler_target=item.get('metadata', {}).get('compiler_target'),
                        compiler_options=item.get('metadata', {}).get('compiler_options'),
                        runtime=item.get('metadata', {}).get('runtime'),
                        runtime_target=item.get('metadata', {}).get('runtime_target'),
                        runtime_hw_type=item.get('metadata', {}).get('runtime_hw_type'),
                        runtime_options=item.get('metadata', {}).get('runtime_options'),
                        runtime_input=item.get('metadata', {}).get('runtime_input'),
                    )

                    test_run = TestRun.objects.create(
                        test_run_batch=test_run_batch,
                        test_case=test_case,
                        test_metadata=test_metadata,
                        outcome=outcome_map[outcome],
                        linked_issue=linked_issue
                    )

                    # Handle failure log file and server-side classification for failed tests
                    failure_log_rel = item.get('failure_log_file')  # relative path inside zip (optional)
                    failed_phase = item.get('failed_phase', 'call')
                    if outcome in ('failed', 'error') and failure_log_rel:
                        failure_log_path = os.path.join(os.path.dirname(manifest_path), failure_log_rel)
                        if os.path.exists(failure_log_path):
                            # Read log content for classification
                            try:
                                with open(failure_log_path, 'r', encoding='utf-8') as fl:
                                    log_content = fl.read()
                            except Exception as e:
                                print(f'Warning: Could not read failure log {failure_log_rel}: {e}')
                                log_content = None

                            # Save the failure log file to storage (same pattern as profiling_data)
                            try:
                                with open(failure_log_path, 'rb') as fl:
                                    test_run.failure_log.save("failure.log", File(fl), save=False)
                            except Exception as e:
                                print(f'Warning: Could not upload failure log {failure_log_rel}: {e}')

                            # Classify failure type server-side
                            if log_content:
                                test_run.failure_type = _classify_failure(failed_phase, log_content)
                            else:
                                test_run.failure_type = _classify_failure(failed_phase, None)
                        else:
                            test_run.failure_type = _classify_failure(failed_phase, None)
                    elif outcome in ('failed', 'error'):
                        test_run.failure_type = _classify_failure(failed_phase, None)

                    # Determine final outcome: Output Mismatch stays as FAIL, everything else becomes ERROR
                    if test_run.failure_type and test_run.failure_type != 'Output Mismatch':
                        test_run.outcome = TestRun.Outcome.ERROR

                    test_run.save()

                    measurements = item.get('measurements', {})

                    if measurements is None:
                        measurements = {}
                    
                    for metric_name, value in measurements.items():

                        if metric_name not in metrics:
                            print(f'Warning: Metric {metric_name} not found for {module}::{name}')
                            continue
                        
                        Measurement.objects.create(test_run=test_run, metric=metrics[metric_name], value=value)
                        
                    if profiling_rel:
                        profiling_path = os.path.join(os.path.dirname(manifest_path), profiling_rel)

                        if os.path.exists(profiling_path):
                            # Save the profiling file
                            try:
                                with open(profiling_path, 'rb') as pf:
                                    test_run.profiling_data.save("trace.pb", File(pf), save=False)
                            except Exception as e:
                                print(f'Warning: Could not upload profiling file {profiling_rel}: {e}')
                                                                
                            test_run.save()
                        else:
                            print(f'Warning: Profiling file {profiling_rel} not found in archive.')

                except Exception as e:
                    # print stack trace 
                    import traceback
                    traceback.print_exc()

                    print(f'Error processing test run: {e}')

            # Mark the batch as processed after all test runs have been created
            test_run_batch.processed = True
            test_run_batch.save()

    except TestRunBatch.DoesNotExist:
        print(f'Error: Test run batch {test_run_batch_id} not found')
    except Exception as e:
        print(f'Error processing uploaded zip: {e}')
    finally:
        # Clean up the uploaded zip file
        if os.path.exists(zip_path):
            os.remove(zip_path)


def _delete_test_session(session_id, chunk_size=500):
    """Delete a TestSession and all its rows bottom-up in bounded chunks.

    A single TestSession can cascade to many thousands of TestRun/Measurement
    rows; letting QuerySet.delete() cascade through all of them at once pulls
    the whole set into memory. Deleting TestRun (and its Measurement cascade)
    in small chunks first keeps each delete's cascade bounded, then
    TestRunBatch and the TestSession itself are cheap to delete since no
    children remain.
    """    

    # we want to use django for the delete here so that the attached files gets wiped too
    while True:
        run_ids = []
        file_names = []
        for pk, profiling_data, failure_log in TestRun.objects.filter(
            test_run_batch__test_session_id=session_id
        ).values_list('pk', 'profiling_data', 'failure_log')[:chunk_size]:
            run_ids.append(pk)
            if profiling_data:
                file_names.append(profiling_data)
            if failure_log:
                file_names.append(failure_log)

        if not run_ids:
            break

        for file_name in file_names:
            try:
                default_storage.delete(file_name)
            except Exception as e:
                print(f'Warning: Could not delete attached file {file_name}: {e}')

        with transaction.atomic():
            TestRun.objects.filter(pk__in=run_ids).delete()        

    with transaction.atomic():
        TestSession.objects.filter(pk=session_id).delete()


def start_trim(retain_days):
    """
    Queue a background job to delete PR test sessions older than retain_days.

    Returns a tuple (sessions_total, prs_total) if a job was queued, or None if there was
    nothing to trim.
    """
    cutoff = timezone.now() - timedelta(days=retain_days)
    sessions_to_trim = TestSession.objects.filter(
        git_branch__startswith='refs/pull/',
        timestamp__lt=cutoff,
    )

    sessions_total = sessions_to_trim.count()
    if sessions_total == 0:
        return None

    prs_total = sessions_to_trim.values_list('git_branch', flat=True).distinct().count()

    save_trim_status(TrimStatus(
        in_progress=True,
        retain_days=retain_days,
        sessions_total=sessions_total,
        prs_total=prs_total,
        prs_done=0,
        sessions_deleted=0,
        started_at=timestamp(),
    ))
    trim_database.spool(retain_days=str(retain_days).encode('utf-8'))

    return sessions_total, prs_total


@spool
def trim_database(args):
    """
    Spooled function to delete old PR test sessions in the background.

    This is processed in background thanks to the @spool decorator to avoid having a timeout on the admin request.

    Args:
        args: Dictionary containing:
            - retain_days: number of days of PR branch history to keep
    """
    retain_days = int(args['retain_days'])
    cutoff = timezone.now() - timedelta(days=retain_days)

    status = get_trim_status()

    # we split the work to avoid doing OOM on small servers
    sessions_deleted = 0
    prs_done = 0
    try:
        while True:
            branches = list(
                TestSession.objects.filter(
                    git_branch__startswith='refs/pull/',
                    timestamp__lt=cutoff,
                ).order_by('git_branch').values_list('git_branch', flat=True)[:1]
            )

            if not branches:
                break
            branch = branches[0]

            while True:
                session_ids = list(
                    TestSession.objects.filter(
                        git_branch=branch,
                        timestamp__lt=cutoff,
                    ).order_by('pk').values_list('id', flat=True)[:1]
                )

                if not session_ids:
                    break

                with transaction.atomic():                    
                    _delete_test_session(session_ids[0], chunk_size=500)

                sessions_deleted += 1

                status.sessions_deleted = sessions_deleted
                save_trim_status(status)

            prs_done += 1
            status.prs_done = prs_done
            save_trim_status(status)

        print(f'Trim completed. Deleted {sessions_deleted} PR sessions for {prs_done} PR branches older than {retain_days} days.')
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f'Error trimming database: {e}')
        status.error = str(e)
    finally:
        status.in_progress = False
        status.finished_at = timestamp()
        save_trim_status(status)
