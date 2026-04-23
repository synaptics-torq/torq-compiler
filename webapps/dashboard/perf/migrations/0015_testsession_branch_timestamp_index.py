from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('perf', '0014_testrun_linked_issue_and_more'),
    ]

    operations = [
        migrations.AddIndex(
            model_name='testsession',
            index=models.Index(fields=['git_branch', '-timestamp'], name='perf_session_branch_ts_idx'),
        ),
    ]