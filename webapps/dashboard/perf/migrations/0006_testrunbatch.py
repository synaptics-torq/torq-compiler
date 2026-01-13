# Generated migration for TestRunBatch model

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('perf', '0005_replace_workflow_fields_with_workflow_url'),
    ]

    operations = [
        migrations.CreateModel(
            name='TestRunBatch',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.TextField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('processed', models.BooleanField(default=False)),
                ('test_session', models.ForeignKey(db_index=True, on_delete=django.db.models.deletion.CASCADE, to='perf.testsession')),
            ],
        ),
    ]
