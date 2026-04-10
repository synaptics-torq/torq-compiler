from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('perf', '0010_testrun_failure_log'),
    ]

    operations = [
        migrations.AddField(
            model_name='testrun',
            name='failure_type',
            field=models.CharField(blank=True, max_length=50, null=True),
        ),
    ]

