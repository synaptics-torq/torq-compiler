from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('perf', '0009_remove_testrun_test_session'),
    ]

    operations = [
        migrations.AddField(
            model_name='testrun',
            name='failure_log',
            field=models.TextField(blank=True, null=True),
        ),
    ]
