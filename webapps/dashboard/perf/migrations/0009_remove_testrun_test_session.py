# Migration to remove test_session from TestRun and make test_run_batch non-nullable

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('perf', '0008_populate_testrun_batches'),
    ]

    operations = [
        migrations.AlterField(
            model_name='testrun',
            name='test_run_batch',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='perf.testrunbatch'),
        ),
        migrations.RemoveField(
            model_name='testrun',
            name='test_session',
        ),
    ]
