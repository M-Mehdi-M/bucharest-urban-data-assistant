"""Thread-safe collection progress, exposed without credentials or response bodies."""
from copy import deepcopy
from datetime import datetime, timezone
import logging
import threading

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_sources = {}


def record(source, state, message, **details):
    entry = {'state': state, 'message': message,
             'checked_at': datetime.now(timezone.utc).isoformat(), **details}
    with _lock:
        _sources[source] = entry
    logger.info('Collection [%s] %s: %s', source, state, message)


def snapshot():
    with _lock:
        return deepcopy(_sources)


def reset():
    with _lock:
        _sources.clear()


def warning():
    entries = snapshot()
    messages = [item['message'] for item in entries.values() if item['state'] in ('cached', 'partial', 'unavailable')]
    return ' '.join(messages) or None
