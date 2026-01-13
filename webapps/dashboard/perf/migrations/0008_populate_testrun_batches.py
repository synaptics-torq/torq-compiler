# Data migration to populate test_run_batch and remove test_session from TestRun

from django.db import migrations


def migrate_testruns_to_batches(apps, schema_editor):
    TestSession = apps.get_model('perf', 'TestSession')
    TestRunBatch = apps.get_model('perf', 'TestRunBatch')
    TestRun = apps.get_model('perf', 'TestRun')
    
    # For each TestSession, create a default batch and reassign test runs
    for session in TestSession.objects.all():
        batch = TestRunBatch.objects.create(
            test_session=session,
            name='Auto-migrated batch',
            processed=True,  # Mark as processed since they're from old data
        )
        
        # Update all TestRuns for this session to use the batch
        TestRun.objects.filter(test_session=session).update(test_run_batch=batch)


def reverse_migrate(apps, schema_editor):
    # Cleanup batches if rolling back
    TestRunBatch = apps.get_model('perf', 'TestRunBatch')
    TestRunBatch.objects.all().delete()


class Migration(migrations.Migration):

    dependencies = [
        ('perf', '0007_testrun_test_run_batch'),
    ]

    operations = [
        migrations.RunPython(migrate_testruns_to_batches, reverse_migrate),
    ]
