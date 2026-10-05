import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

# Trim progress is tracked directly in the default (S3) storage backend, outside of the
# application database, so the status survives independently of the web/spooler processes.
STATUS_KEY = 'trim_status.json'


@dataclass
class TrimStatus:
    in_progress: bool = False
    retain_days: Optional[int] = None
    sessions_total: Optional[int] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None


def timestamp():
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')


def get_trim_status():
    """Return the current TrimStatus, or a default (idle) one if none is stored."""
    if not default_storage.exists(STATUS_KEY):
        return TrimStatus()

    with default_storage.open(STATUS_KEY, 'rb') as f:
        data = json.loads(f.read().decode('utf-8'))
    return TrimStatus(**data)


def save_trim_status(status):
    """Persist (overwrite) the trim status."""
    if default_storage.exists(STATUS_KEY):
        default_storage.delete(STATUS_KEY)
    default_storage.save(STATUS_KEY, ContentFile(json.dumps(asdict(status)).encode('utf-8')))
