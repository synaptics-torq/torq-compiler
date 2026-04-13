import gzip
import os
import signal
import subprocess
import threading
import zlib
from datetime import datetime, timezone

from django import forms
from django.contrib import admin, messages
from django.conf import settings
from django.http import StreamingHttpResponse
from django.shortcuts import redirect, render

try:
    import uwsgi
except ImportError:
    uwsgi = None


CHUNK_SIZE = 1024 * 1024
RESET_SQL = (
    b'DROP SCHEMA IF EXISTS public CASCADE;\n'
    b'CREATE SCHEMA public;\n'
    b'GRANT ALL ON SCHEMA public TO postgres;\n'
    b'GRANT ALL ON SCHEMA public TO public;\n'
)


def _get_pg_db():
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


def _stream_pg_dump_chunks(command, env):
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    def generate_chunks():
        compressor = zlib.compressobj(wbits=31)
        stderr = b''
        try:
            while True:
                chunk = process.stdout.read(CHUNK_SIZE)
                if not chunk:
                    break
                compressed_chunk = compressor.compress(chunk)
                if compressed_chunk:
                    yield compressed_chunk

            final_chunk = compressor.flush()
            if final_chunk:
                yield final_chunk
        finally:
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                stderr = process.stderr.read()
            return_code = process.wait()

        if return_code != 0:
            error_text = stderr.decode(errors='replace').strip() or 'pg_dump failed.'
            raise RuntimeError(error_text)

    return generate_chunks()


def _restore_database_from_upload(command, env, uploaded_file):
    process = subprocess.Popen(
        command,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
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


class RestoreDatabaseForm(forms.Form):
    backup_file = forms.FileField(
        label='SQL backup file',
        help_text='Upload a .sql.gz file previously downloaded via the backup action.',
    )


def backup_database_view(request):
    db, error = _get_pg_db()
    if error:
        messages.error(request, error)
        return redirect('admin:index')

    command, env = _build_pg_command_and_env(
        ['pg_dump', '--clean', '--if-exists', '--no-owner', '--no-privileges'],
        db,
    )
    try:
        stream = _stream_pg_dump_chunks(command, env)
    except FileNotFoundError:
        messages.error(request, 'pg_dump is not available on this host.')
        return redirect('admin:index')

    filename = f"backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.sql.gz"
    response = StreamingHttpResponse(stream, content_type='application/gzip')
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response


def restore_database_view(request):
    if request.method == 'POST':
        form = RestoreDatabaseForm(request.POST, request.FILES)
        if form.is_valid():
            sql_file = request.FILES['backup_file']
            db, error = _get_pg_db()
            if error:
                messages.error(request, error)
            else:
                command, env = _build_pg_command_and_env(['psql', '-v', 'ON_ERROR_STOP=1', '--single-transaction'], db)
                try:
                    _restore_database_from_upload(command, env, sql_file)
                    messages.warning(request, f'Database restored from "{sql_file.name}". Restarting uWSGI.')
                    _schedule_uwsgi_restart()
                    return redirect('admin:index')
                except FileNotFoundError:
                    messages.error(request, 'psql is not available on this host.')
                except OSError as exc:
                    messages.error(request, str(exc))
                except subprocess.CalledProcessError as exc:
                    messages.error(request, f'Restore failed: {(exc.stderr or b"").decode().strip()}')
    else:
        form = RestoreDatabaseForm()

    context = {
        **admin.site.each_context(request),
        'form': form,
        'title': 'Restore Database',
    }
    return render(request, 'admin/perf/restore_database.html', context)
