import os
import zipfile
import json
from tempfile import TemporaryDirectory
from django.core.files.base import File

# conditionally import uwsgi spool decorator to make sure this module can be imported also in manage.py
try:
    import uwsgi
    from uwsgidecorators import spool
except ImportError:
    def spool(pass_arguments):
        return lambda x: x
    
from perf.models import TestCase, TestSession, TestRun, TestRunBatch


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

            outcome_map = {
                'passed': TestRun.Outcome.PASS,
                'failed': TestRun.Outcome.FAIL,
                'skipped': TestRun.Outcome.SKIP,
            }

            test_runs = manifest.get('test_runs', [])
            
            for item in test_runs:                
                module = item.get('module', '')
                name = item.get('name', '')
                parameters = item.get('parameters', '')
                outcome = item.get('outcome', '')
                profiling_rel = item.get('profiling_file')  # relative path inside zip (optional)

                if not module or not name or outcome not in outcome_map:
                    # Log error but continue processing other test runs
                    print(f'Invalid run entry in manifest: module={module}, name={name}, outcome={outcome}')
                    continue
                
                test_case, _ = TestCase.objects.get_or_create(
                    module=str(module), name=str(name), parameters=str(parameters)
                )

                test_run = TestRun.objects.create(
                    test_run_batch=test_run_batch,
                    test_case=test_case,
                    outcome=outcome_map[outcome],
                )

                if profiling_rel:
                    profiling_path = os.path.join(os.path.dirname(manifest_path), profiling_rel)

                    if os.path.exists(profiling_path):
                        with open(profiling_path, 'rb') as pf:
                            test_run.profiling_data.save(os.path.basename(profiling_path), File(pf), save=True)
                    else:
                        print(f'Warning: Profiling file {profiling_rel} not found in archive.')

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
