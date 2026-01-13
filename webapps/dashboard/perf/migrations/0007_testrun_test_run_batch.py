# Migration to add test_run_batch FK and populate it with default batches

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('perf', '0006_testrunbatch'),
    ]

    operations = [
        migrations.AddField(
            model_name='testrun',
            name='test_run_batch',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='perf.testrunbatch'),
        ),
    ]
