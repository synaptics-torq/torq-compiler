import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

# Backups are stored directly in the default (S3) storage backend, outside of the
# application database, so that they remain available even after a database restore
# wipes out any bookkeeping that would otherwise live in a DB table.
BACKUP_PREFIX = 'db_backups/'

_ID_RE = re.compile(r'^[0-9]{14}_[0-9a-f]{8}$')


class Status:
    PENDING = 'pending'
    RUNNING = 'running'
    DONE = 'done'
    FAILED = 'failed'


_STATUS_LABELS = {
    Status.PENDING: 'Pending',
    Status.RUNNING: 'Running',
    Status.DONE: 'Done',
    Status.FAILED: 'Failed',
}


@dataclass
class BackupRecord:
    id: str
    created_at: str
    status: str = Status.PENDING
    finished_at: Optional[str] = None
    error_message: Optional[str] = None
    filename: Optional[str] = None
    restore_status: Optional[str] = None
    restore_finished_at: Optional[str] = None
    restore_error_message: Optional[str] = None
    restart_requested_at: Optional[str] = None

    @property
    def status_display(self):
        return _STATUS_LABELS.get(self.status, self.status)

    @property
    def restore_status_display(self):
        return _STATUS_LABELS.get(self.restore_status, self.restore_status)

    @property
    def size(self):
        """Size in bytes of the stored backup file, or None if unavailable."""
        if not self.filename:
            return None
        try:
            return default_storage.size(self.filename)
        except (FileNotFoundError, OSError):
            return None


def _meta_key(backup_id):
    return f'{BACKUP_PREFIX}{backup_id}.json'


def _data_key(backup_id, filename):
    return f'{BACKUP_PREFIX}{backup_id}/{filename}'


def timestamp():
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')


def create_backup_record():
    """Create and persist a new, empty backup record and return it."""
    backup_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid4().hex[:8]}"
    record = BackupRecord(id=backup_id, created_at=timestamp())
    save_backup(record)
    return record


def save_backup(record):
    """Persist (overwrite) the metadata for a backup record."""
    name = _meta_key(record.id)
    if default_storage.exists(name):
        default_storage.delete(name)
    default_storage.save(name, ContentFile(json.dumps(asdict(record)).encode('utf-8')))


def save_backup_file(record, filename, file_obj):
    """Store the dump/upload content for a record and persist the updated metadata."""
    name = _data_key(record.id, filename)
    if default_storage.exists(name):
        default_storage.delete(name)
    record.filename = default_storage.save(name, file_obj)
    save_backup(record)


def open_backup_file(record):
    return default_storage.open(record.filename, 'rb')


def delete_backup(record):
    """Remove the dump/upload file and metadata for a backup record from storage."""
    if record.filename and default_storage.exists(record.filename):
        default_storage.delete(record.filename)

    name = _meta_key(record.id)
    if default_storage.exists(name):
        default_storage.delete(name)


def get_backup(backup_id):
    """Look up a backup record by id. Returns None if not found or if the id is malformed."""
    if not backup_id or not _ID_RE.match(backup_id):
        return None

    name = _meta_key(backup_id)
    if not default_storage.exists(name):
        return None

    with default_storage.open(name, 'rb') as f:
        data = json.loads(f.read().decode('utf-8'))
    return BackupRecord(**data)


def list_backups():
    """Return all backup records, most recently created first."""
    try:
        _, files = default_storage.listdir(BACKUP_PREFIX.rstrip('/'))
    except FileNotFoundError:
        return []

    records = []
    for name in files:
        if not name.endswith('.json'):
            continue
        record = get_backup(name[:-len('.json')])
        if record is not None:
            records.append(record)

    records.sort(key=lambda r: r.created_at, reverse=True)
    return records
