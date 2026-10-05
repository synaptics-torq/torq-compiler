import gzip
import os
import signal
import subprocess
import threading
import zlib
from tempfile import TemporaryDirectory

from django import forms
from django.contrib import admin, messages
from django.conf import settings
from django.core.files.base import File
from django.http import FileResponse, Http404
from django.shortcuts import redirect, render
from django.utils import timezone

from perf import backup_store

# conditionally import uwsgi spool decorator to make sure this module can be imported also in manage.py
try:
    import uwsgi
    from uwsgidecorators import spool
except ImportError:
    uwsgi = None

    def spool(pass_arguments):
        return lambda x: x


CHUNK_SIZE = 1024 * 1024
RESET_SQL = (
    b'DROP SCHEMA IF EXISTS public CASCADE;\n'
    b'CREATE SCHEMA public;\n'
    b'GRANT ALL ON SCHEMA public TO postgres;\n'
    b'GRANT ALL ON SCHEMA public TO public;\n'
)


def get_pg_db():
    db = settings.DATABASES.get('default', {})
    if 'postgresql' not in db.get('ENGINE', ''):
        return None, 'Configured database engine is not PostgreSQL.'
    if not db.get('NAME'):
        return None, 'PostgreSQL database name is not configured.'
    return db, None


def _build_pg_command_and_env(base_command, db):
    command = list(base_command)
    if db.get('HOST'):
        command.extend(['-h', str(db['HOST'])])
    if db.get('PORT'):
        command.extend(['-p', str(db['PORT'])])
    if db.get('USER'):
        command.extend(['-U', str(db['USER'])])
    command.append(str(db['NAME']))
    env = os.environ.copy()
    if db.get('PASSWORD'):
        env['PGPASSWORD'] = str(db['PASSWORD'])
    return command, env


def _restart_uwsgi_process():
    if uwsgi is not None:
        uwsgi.reload()
        return
    os.kill(os.getpid(), signal.SIGTERM)


def _schedule_uwsgi_restart(delay_seconds=0.5):
    restart_timer = threading.Timer(delay_seconds, _restart_uwsgi_process)
    restart_timer.daemon = True
    restart_timer.start()


def _iter_uploaded_sql_chunks(uploaded_file):
    file_obj = getattr(uploaded_file, 'file', uploaded_file)

    with gzip.GzipFile(fileobj=file_obj, mode='rb') as gz_file:
        while True:
            chunk = gz_file.read(CHUNK_SIZE)
            if not chunk:
                break
            yield chunk


def _dump_pg_to_file(command, env, file_path):
    # lower priority so this doesn't starve the web workers sharing the same CPU
    os.nice(15)
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=lambda: os.nice(15),
    )

    compressor = zlib.compressobj(wbits=31)
    stderr = b''
    try:
        with open(file_path, 'wb') as out_file:
            while True:
                chunk = process.stdout.read(CHUNK_SIZE)
                if not chunk:
                    break
                compressed_chunk = compressor.compress(chunk)
                if compressed_chunk:
                    out_file.write(compressed_chunk)

            final_chunk = compressor.flush()
            if final_chunk:
                out_file.write(final_chunk)
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            stderr = process.stderr.read()
        return_code = process.wait()

    if return_code != 0:
        error_text = stderr.decode(errors='replace').strip() or 'pg_dump failed.'
        raise RuntimeError(error_text)


def _restore_database_from_upload(command, env, uploaded_file):
    # lower priority so this doesn't starve the web workers sharing the same CPU
    os.nice(15)
    process = subprocess.Popen(
        command,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        preexec_fn=lambda: os.nice(15),
    )

    stderr = b''
    try:
        process.stdin.write(RESET_SQL)
        for chunk in _iter_uploaded_sql_chunks(uploaded_file):
            process.stdin.write(chunk)
        process.stdin.close()
    except Exception:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        process.kill()
        stderr = process.stderr.read() if process.stderr is not None else b''
        process.wait()
        raise

    if process.stderr is not None:
        stderr = process.stderr.read()
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command, stderr=stderr)


def _format_restore_error(exc):
    if isinstance(exc, FileNotFoundError):
        return 'psql is not available on this host.'
    if isinstance(exc, subprocess.CalledProcessError):
        return f'Restore failed: {(exc.stderr or b"").decode().strip()}'
    return str(exc)


@spool
def create_database_backup(args):
    """
    Spooled function to run pg_dump in the background and attach the gzipped result to a backup record.

    This is processed in background thanks to the @spool decorator to avoid having a timeout on the admin request.

    Args:
        args: Dictionary containing:
            - backup_id: ID of the pre-created backup record
    """
    record = backup_store.get_backup(args['backup_id'])
    if record is None:
        return

    record.status = backup_store.Status.RUNNING
    backup_store.save_backup(record)

    db, error = get_pg_db()
    if error:
        record.status = backup_store.Status.FAILED
        record.error_message = error
        record.finished_at = backup_store.timestamp()
        backup_store.save_backup(record)
        return

    command, env = _build_pg_command_and_env(
        ['pg_dump', '--clean', '--if-exists', '--no-owner', '--no-privileges'],
        db,
    )
    filename = f"backup_{timezone.now().strftime('%Y%m%d_%H%M%S')}.sql.gz"

    try:
        with TemporaryDirectory() as temp_dir:
            dump_path = os.path.join(temp_dir, filename)
            _dump_pg_to_file(command, env, dump_path)
            with open(dump_path, 'rb') as f:
                backup_store.save_backup_file(record, filename, File(f))
        record.status = backup_store.Status.DONE
    except Exception as exc:
        record.status = backup_store.Status.FAILED
        record.error_message = str(exc)
    finally:
        record.finished_at = backup_store.timestamp()
        backup_store.save_backup(record)


@spool
def restore_database_backup(args):
    """
    Spooled function to restore the database from a backup record in the background.

    This is processed in background thanks to the @spool decorator to avoid having a timeout on the admin request.

    Args:
        args: Dictionary containing:
            - backup_id: ID of the backup record to restore from
    """
    record = backup_store.get_backup(args['backup_id'])
    if record is None:
        return

    # check if the restore has been just scheduled, we may end up with restores
    # that were already completed or that crashed mid-way, we should not attempt 
    # to restore them again.
    if record.restore_status != backup_store.Status.PENDING:

        # the check if the restore was already running and the spooler was killed
        if record.restore_status == backup_store.Status.RUNNING:
            record.restore_status = backup_store.Status.FAILED
            record.restore_finished_at = backup_store.timestamp()
            backup_store.save_backup(record)

        return

    record.restore_status = backup_store.Status.RUNNING
    record.restore_error_message = None
    backup_store.save_backup(record)

    db, error = get_pg_db()
    if error:
        record.restore_status = backup_store.Status.FAILED
        record.restore_error_message = error
        record.restore_finished_at = backup_store.timestamp()
        backup_store.save_backup(record)
        return

    command, env = _build_pg_command_and_env(['psql', '-v', 'ON_ERROR_STOP=1', '--single-transaction'], db)
    restore_succeeded = False
    try:
        with backup_store.open_backup_file(record) as sql_file:
            _restore_database_from_upload(command, env, sql_file)
        record.restore_status = backup_store.Status.DONE
        restore_succeeded = True
    except Exception as exc:
        record.restore_status = backup_store.Status.FAILED
        record.restore_error_message = _format_restore_error(exc)
    finally:
        record.restore_finished_at = backup_store.timestamp()
        backup_store.save_backup(record)

    # only restart once the record save above has completed, so uwsgi can't reload mid-write
    if restore_succeeded:
        record.restart_requested_at = backup_store.timestamp()
        backup_store.save_backup(record)
        _schedule_uwsgi_restart()


class UploadBackupForm(forms.Form):
    backup_file = forms.FileField(
        label='SQL backup file',
        help_text='Upload a .sql.gz file to add it to the list of backups below, from which it can then be restored.',
    )


def backup_database_view(request):
    if request.method == 'POST':
        db, error = get_pg_db()
        if error:
            messages.error(request, error)
            return redirect('admin_database_backup')

        record = backup_store.create_backup_record()
        create_database_backup.spool(backup_id=record.id.encode('utf-8'))
        messages.info(request, 'Backup started in the background. Refresh this page to check its progress.')
        return redirect('admin_database_backup')

    context = {
        **admin.site.each_context(request),
        'backups': backup_store.list_backups()[:20],
        'title': 'Backup Database',
    }
    return render(request, 'admin/perf/backup_database.html', context)


def download_backup_view(request, backup_id):
    record = backup_store.get_backup(backup_id)
    if record is None or record.status != backup_store.Status.DONE or not record.filename:
        raise Http404('Backup is not available for download.')

    response = FileResponse(backup_store.open_backup_file(record), content_type='application/gzip')
    response['Content-Disposition'] = f'attachment; filename="{os.path.basename(record.filename)}"'
    return response


def upload_backup_view(request):
    if request.method == 'POST':
        form = UploadBackupForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['backup_file']
            record = backup_store.create_backup_record()
            record.status = backup_store.Status.DONE
            record.finished_at = backup_store.timestamp()
            backup_store.save_backup_file(record, uploaded_file.name, uploaded_file)
            messages.success(request, f'Uploaded "{uploaded_file.name}" as a new backup.')
            return redirect('admin_database_backup')
    else:
        form = UploadBackupForm()

    context = {
        **admin.site.each_context(request),
        'form': form,
        'title': 'Upload Backup',
    }
    return render(request, 'admin/perf/upload_backup.html', context)


def restore_backup_view(request, backup_id):
    record = backup_store.get_backup(backup_id)
    if record is None or record.status != backup_store.Status.DONE or not record.filename:
        raise Http404('Backup is not available for restore.')

    if request.method == 'POST':
        record.restore_status = backup_store.Status.PENDING
        record.restore_error_message = None
        record.restore_finished_at = None
        backup_store.save_backup(record)

        restore_database_backup.spool(backup_id=record.id.encode('utf-8'))
        messages.info(request, 'Restore started in the background. Refresh the backup list to check its progress.')
        return redirect('admin_database_backup')

    context = {
        **admin.site.each_context(request),
        'backup': record,
        'title': 'Restore Database from Backup',
    }
    return render(request, 'admin/perf/restore_backup_confirm.html', context)


def delete_backup_view(request, backup_id):
    record = backup_store.get_backup(backup_id)
    if record is None:
        raise Http404('Backup not found.')

    if record.restore_status in (backup_store.Status.PENDING, backup_store.Status.RUNNING):
        messages.error(request, 'Cannot delete a backup while a restore from it is in progress.')
        return redirect('admin_database_backup')

    if request.method == 'POST':
        backup_store.delete_backup(record)
        messages.success(request, f'Deleted backup created on {record.created_at}.')
        return redirect('admin_database_backup')

    context = {
        **admin.site.each_context(request),
        'backup': record,
        'title': 'Delete Backup',
    }
    return render(request, 'admin/perf/delete_backup_confirm.html', context)
