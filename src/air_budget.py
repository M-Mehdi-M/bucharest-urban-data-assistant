"""Respect Airly quota headers across refreshes and application restarts."""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
import json
import math
from pathlib import Path
import time
from uuid import uuid4

from . import config


def _read(api_key=None):
    try:
        data = json.loads(Path(config.AIRLY_QUOTA_FILE).read_text(encoding='utf-8'))
        fingerprint = sha256((api_key or config.AIRLY_API_KEY).encode()).hexdigest()
        return data if isinstance(data, dict) and data.get('key_fingerprint') == fingerprint else {}
    except (OSError, ValueError):
        return {}


def blocked(api_key=None):
    data = _read(api_key)
    until = data.get('blocked_until', 0)
    return data if isinstance(until, (float, int)) and until > time.time() else None


def allowance():
    data = _read()
    budget = config.AIRLY_MAX_MEASUREMENTS_PER_REFRESH
    for field in ('remaining_day', 'remaining_minute'):
        if isinstance(data.get(field), int):
            budget = min(budget, max(0, data[field]))
    return budget


def observe(response, api_key):
    headers = response.headers
    def number(name):
        try:
            return int(headers.get(name))
        except (TypeError, ValueError):
            return None
    day, minute = number('x-ratelimit-remaining-day'), number('x-ratelimit-remaining-minute')
    limited = response.status_code == 429 or day == 0 or minute == 0
    delay = 0
    if limited:
        try:
            delay = float(headers.get('Retry-After'))
        except (TypeError, ValueError):
            try:
                delay = parsedate_to_datetime(headers.get('Retry-After')).timestamp() - time.time()
            except (TypeError, ValueError, AttributeError):
                delay = 3600 if day == 0 else 300
        delay = min(max(delay, 1), 86400) if math.isfinite(delay) else 300
    data = {'key_fingerprint': sha256(api_key.encode()).hexdigest(),
            'blocked_until': time.time() + delay if limited else 0,
            'remaining_day': day, 'remaining_minute': minute}
    if limited:
        data['retry_at'] = datetime.fromtimestamp(data['blocked_until'], timezone.utc).isoformat()
    path = Path(config.AIRLY_QUOTA_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}-{uuid4().hex}.tmp')
    try:
        temporary.write_text(json.dumps(data), encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
