import os
import zipfile
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from django.core.files.base import File

# conditionally import uwsgi spool decorator to make sure this module can be imported also in manage.py
try:
    import uwsgi
    from uwsgidecorators import spool
except ImportError:
    def spool(pass_arguments):
        return lambda x: x
    
from perf.models import TestCase, TestSession, TestRun, TestRunBatch, Metric, Measurement
from perf.perfetto_report_generator import generate_html as generate_perfetto_html
from perf.perfetto_report_generator import extract_perfetto_summary

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
    
    # Metric descriptions dictionary (moved outside loop for reuse)
    metric_descriptions = {
        'total_duration': 'Total execution duration of the workload',
        'dma_time': 'Combined time for DMA and CDMA operations (union)',
        'dma_percent': 'Percentage of dma_time over total duration',
        'dma_only_time': 'Time spent exclusively on DMA/CDMA without compute overlap',
        'dma_only_percent': 'Percentage of dma_only_time over total duration',
        'dma_total_time': 'Total DMA operation time (doesnt include CDMA)',
        'dma_total_percent': 'Percentage of dma_total_time over total duration',
        'cdma_time': 'Total CDMA (Constant DMA) operation time',
        'cdma_percent': 'Percentage of cdma_time over total duration',
        'dma_in_time': 'Time spent on DMA input transfers',
        'dma_in_percent': 'Percentage of dma_in_time over total duration',
        'dma_out_time': 'Time spent on DMA output transfers',
        'dma_out_percent': 'Percentage of dma_out_time over total duration',
        'compute_time': 'Combined compute time for SLICE and CSS operations (union)',
        'compute_percent': 'Percentage of compute_time over total duration',
        'compute_only_time': 'Time spent exclusively on compute without DMA overlap',
        'compute_only_percent': 'Percentage of compute_only_time over total duration',
        'slice_time': 'Total time spent on SLICE 0 and 1 compute operations (union)',
        'slice_percent': 'Percentage of slice_time over total duration',
        'slice_0_time': 'Time spent on SLICE 0 compute operations (exclusive)',
        'slice_0_percent': 'Percentage of slice_0_time over total duration',
        'slice_1_time': 'Time spent on SLICE 1 compute operations (exclusive)',
        'slice_1_percent': 'Percentage of slice_1_time over total duration',
        'css_time': 'Total time spent on CSS (Compute Subsystem) operations',
        'css_percent': 'Percentage of css_time over total duration',
        'overlap_time': 'Time where DMA/CDMA and compute operations overlap',
        'overlap_percent': 'Percentage of overlap_time over total duration',
        'idle_time': 'Time when hardware is idle (no DMA or compute, it could be host operation as well)',
        'idle_percent': 'Percentage of idle_time over total duration',
    }
    
    # Cache for metric objects to avoid repeated database queries
    metrics_cache = {}
    
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
                        # Save the profiling file
                        with open(profiling_path, 'rb') as pf:
                            test_run.profiling_data.save(os.path.basename(profiling_path), File(pf), save=False)
                        
                        # Extract and save metrics from the .pb file
                        try:
                            summary = extract_perfetto_summary(profiling_path)
                            if summary and summary.get('available'):
                                for key, value in summary.items():
                                    print(f'DEBUG: Processing metric: {key} = {value} (type: {type(value).__name__})', flush=True)
                                    if key == 'available' or not value:
                                        print(f'DEBUG: Skipping {key} (available or empty)', flush=True)
                                        continue
                                    
                                    # Parse and normalize value
                                    parsed_value = None
                                    unit = ""
                                    
                                    if isinstance(value, str):
                                        # Check if this is a percentage metric by name
                                        if key.endswith('_percent'):
                                            try:
                                                parsed_value = float(value.replace('%', '').strip())
                                                unit = '%'
                                            except ValueError as e:
                                                continue
                                        # Parse time values to nanoseconds - check specific patterns first
                                        elif 'ms' in value:
                                            try:
                                                num = float(value.replace('ms', '').strip())
                                                parsed_value = num * 1_000_000  # ms to ns
                                                unit = 'ns'
                                            except ValueError as e:
                                                continue
                                        elif 'µs' in value or 'μs' in value:
                                            try:
                                                num = float(value.replace('µs', '').replace('μs', '').strip())
                                                parsed_value = num * 1_000  # µs to ns
                                                unit = 'ns'
                                            except ValueError as e:
                                                continue
                                        elif 'ns' in value:
                                            # Handle "0ns", "123ns", etc.
                                            try:
                                                num = float(value.replace('ns', '').strip())
                                                parsed_value = num  # already in ns
                                                unit = 'ns'
                                            except ValueError as e:
                                                continue
                                        elif value.endswith('s'):
                                            # Must be seconds (after ruling out ms, µs, ns)
                                            try:
                                                num = float(value.replace('s', '').strip())
                                                parsed_value = num * 1_000_000_000  # s to ns
                                                unit = 'ns'
                                            except ValueError as e:
                                                continue
                                        else:
                                            continue
                                    else:
                                        continue
                                    
                                    if parsed_value is None:
                                        continue
                                    
                                    # Get or create metric (will be cached after first access)
                                    metric = metrics_cache.get(key)
                                    if not metric:
                                        # Get description for this metric
                                        description = metric_descriptions.get(key, '')
                                        
                                        metric, created = Metric.objects.get_or_create(
                                            name=key, 
                                            defaults={'unit': unit, 'description': description}
                                        )
                                        # Update description if metric already exists but has no description
                                        if not created and not metric.description:
                                            metric.description = description
                                            metric.save()
                                        
                                        # Cache it for reuse
                                        metrics_cache[key] = metric
                                    
                                    # Store normalized numeric value
                                    Measurement.objects.create(test_run=test_run, metric=metric, value=parsed_value)
                        except Exception as e:
                                print(f'Warning: Could not extract metrics from {profiling_path}: {e}')
                        
                        test_run.save()
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
