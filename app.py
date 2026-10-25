"""Web API with atomic snapshots and stale-while-refresh availability."""
import argparse
import logging
import sys
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

from src import config
from src import source_health
from src.map_view import REFERRER_POLICY
from src.agent import UrbanDataAgent
from src.contracts import validate_query
from src.data_collection import collect_all_data
from src.data_processing import process_and_integrate_data
from src.visualization import UrbanVisualizer

logger = logging.getLogger(__name__)
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024


@dataclass(frozen=True)
class Snapshot:
    data: object
    agent: object
    visualizer: object
    updated_at: str
    map_file: str


@dataclass
class Conversation:
    history: list = field(default_factory=list)
    lock: object = field(default_factory=threading.Lock)


_snapshot = None
_state_lock = threading.Lock()
_load_lock = threading.Lock()
_session_lock = threading.Lock()
_bootstrap_lock = threading.Lock()
_started = False
is_updating = False
last_error = None
_last_manual_update = 0.0
sessions = OrderedDict()
_request_log = OrderedDict()
MAX_SESSIONS = 500
MAX_QUERY_LENGTH = 500
UPDATE_COOLDOWN = 300


def publish(data):
    """Build everything first: failed initialization never replaces working state."""
    global _snapshot
    agent = UrbanDataAgent(data)
    visualizer = UrbanVisualizer(data)
    map_file = f'parks_{uuid.uuid4().hex}.html'
    visualizer.save_map(visualizer.create_simplified_map(), map_file)
    snapshot = Snapshot(data, agent, visualizer, datetime.now(timezone.utc).isoformat(), map_file)
    with _state_lock:
        _snapshot = snapshot


def load_data(force_update=False, reserved=False):
    global is_updating, last_error
    if not reserved and not _load_lock.acquire(blocking=False):
        return False
    with _state_lock:
        is_updating = True
    try:
        collected = collect_all_data(force_update=force_update)
        data = process_and_integrate_data(collected)
        if data is None or data.empty:
            raise ValueError('No usable park records')
        publish(data)
        with _state_lock:
            last_error = None
        return True
    except Exception:
        logger.exception('Data refresh failed; preserving previous snapshot')
        with _state_lock:
            last_error = 'Actualizarea nu a reușit. Ultimul set disponibil este păstrat.'
        return False
    finally:
        with _state_lock:
            is_updating = False
        _load_lock.release()


def _startup():
    global is_updating
    # Fast offline startup from source caches; reprocess to enforce evidence rules.
    if Path(config.PARKS_SECTORS_FILE).exists() and _load_lock.acquire(blocking=False):
        with _state_lock:
            is_updating = True
        try:
            import geopandas as gpd
            cached = {'parks_sectors': gpd.read_file(config.PARKS_SECTORS_FILE)}
            if Path(config.AIR_QUALITY_FILE).exists():
                cached['air_quality'] = gpd.read_file(config.AIR_QUALITY_FILE)
            data = process_and_integrate_data(cached)
            if data is not None and not data.empty:
                publish(data)
        except Exception:
            logger.exception('Cached startup unavailable')
        finally:
            with _state_lock:
                is_updating = False
            _load_lock.release()
    load_data()


def _periodic_refresh():
    while True:
        time.sleep(6 * 60 * 60)
        load_data()  # Static geometry need not be downloaded every six hours.


def create_app(start_background=True):
    """WSGI factory: waitress-serve --call app:create_app (one process)."""
    global _started
    if start_background:
        with _bootstrap_lock:
            if not _started:
                _started = True
                threading.Thread(target=_startup, daemon=True).start()
                threading.Thread(target=_periodic_refresh, daemon=True).start()
    return app


def rate_limited(ip):
    now = time.monotonic()
    with _session_lock:
        log = _request_log.setdefault(ip, deque())
        _request_log.move_to_end(ip)
        while log and now - log[0] >= 60:
            log.popleft()
        if len(log) >= 20:
            return True
        log.append(now)
        while len(_request_log) > 4096:
            _request_log.popitem(last=False)
    return False


@app.after_request
def security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    # Tile requests must carry the app's origin, without exposing paths or queries.
    response.headers['Referrer-Policy'] = REFERRER_POLICY
    if request.path.startswith('/api/'):
        response.headers['Cache-Control'] = 'no-store'
    return response


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error='Cererea este prea mare.'), 413


@app.get('/')
def home():
    snapshot = _snapshot
    sectors = []
    if snapshot is not None:
        counts = snapshot.data.groupby('sector').size().sort_values(ascending=False)
        sectors = [{'name': name, 'count': int(count)} for name, count in counts.items()]
    return render_template('index.html', map_file=snapshot.map_file if snapshot else None,
                           parks_count=len(snapshot.data) if snapshot else 0,
                           sectors=sectors, is_updating=is_updating, ready=snapshot is not None,
                           simulation=config.ENABLE_SIMULATED_DATA,
                           last_update=snapshot.updated_at if snapshot else 'În așteptare',
                           error_message=last_error or source_health.warning())


@app.get('/api/status')
@app.get('/debug-data')
def status():
    snapshot = _snapshot
    counts = snapshot.data.groupby('sector').size().sort_values(ascending=False) if snapshot else {}
    payload = {'ready': snapshot is not None, 'is_updating': is_updating,
               'rows': len(snapshot.data) if snapshot else 0,
               'last_update': snapshot.updated_at if snapshot else None,
               'map_url': f'/maps/{snapshot.map_file}' if snapshot else None,
               'data_mode': 'simulation' if config.ENABLE_SIMULATED_DATA else 'measured',
               'message': last_error or source_health.warning(),
               'llm_configured': bool(config.GEMINI_API_KEY)}
    payload['sources'] = source_health.snapshot()
    payload['air_quality_parks'] = int(snapshot.data.aqi.notna().sum()) if snapshot and 'aqi' in snapshot.data else 0
    payload['degraded'] = bool(source_health.warning())
    payload['sectors'] = [{'name': name, 'count': int(count)} for name, count in counts.items()]
    return jsonify(payload)


@app.post('/api/query')
def process_query():
    try:
        query = validate_query(request.get_json(silent=True), MAX_QUERY_LENGTH)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    snapshot = _snapshot
    if snapshot is None:
        response = jsonify(error='Datele se pregătesc. Te anunțăm când poți începe.')
        response.headers['Retry-After'] = '5'
        return response, 503
    if rate_limited(request.remote_addr):
        response = jsonify(error='Prea multe întrebări. Încearcă din nou peste un minut.')
        response.headers['Retry-After'] = '60'
        return response, 429
    sid = None if request.get_json(silent=True).get('reset_session') is True else request.cookies.get('sid')
    new_cookie = False
    with _session_lock:
        if sid not in sessions:
            sid = uuid.uuid4().hex
            sessions[sid] = Conversation()
            new_cookie = True
        sessions.move_to_end(sid)
        conversation = sessions[sid]
        while len(sessions) > MAX_SESSIONS:
            sessions.popitem(last=False)
    try:
        # Serialize a visitor's history read + update, including simultaneous tabs.
        with conversation.lock:
            result = snapshot.agent.process_query(query, history=list(conversation.history))
            conversation.history = result.get('history', conversation.history)[-config.MAX_HISTORY_LENGTH * 2:]
        response = jsonify(response=result.get('response', ''), visualization=result.get('visualization'),
                           evidence=result.get('evidence', {}), timestamp=datetime.now(timezone.utc).isoformat(),
                           data_updated_at=snapshot.updated_at,
                           data_mode='simulation' if config.ENABLE_SIMULATED_DATA else 'measured')
        if new_cookie:
            response.set_cookie('sid', sid, httponly=True, samesite='Lax', secure=request.is_secure)
        return response
    except Exception:
        logger.exception('Query failed')
        return jsonify(error='Întrebarea nu a putut fi procesată. Încearcă din nou.'), 500


@app.post('/update-data')
def update_data():
    global _last_manual_update, is_updating
    with _state_lock:
        now = time.monotonic()
        if is_updating:
            return jsonify(message='Actualizarea este deja în curs.'), 409
        if _last_manual_update and now - _last_manual_update < UPDATE_COOLDOWN:
            return jsonify(message='Datele au fost actualizate recent. Încearcă peste câteva minute.'), 429
        if not _load_lock.acquire(blocking=False):
            return jsonify(message='Actualizarea este deja în curs.'), 409
        is_updating = True
        _last_manual_update = now
    try:
        threading.Thread(target=load_data, kwargs={'force_update': True, 'reserved': True}, daemon=True).start()
    except Exception:
        with _state_lock:
            is_updating = False
            _last_manual_update = 0.0
        _load_lock.release()
        logger.exception('Unable to start refresh')
        return jsonify(message='Actualizarea nu a putut porni.'), 500
    return jsonify(message='Actualizarea a început. Poți continua să folosești datele disponibile.'), 202


@app.get('/api/map')
def get_map():
    snapshot = _snapshot
    if snapshot is None:
        return jsonify(error='Datele se pregătesc.'), 503
    kind, column = request.args.get('type', 'parks'), request.args.get('column')
    if kind not in ('parks', 'heatmap') or (kind == 'heatmap' and column not in ('aqi', 'noise_level_db')):
        return jsonify(error='Tip de hartă sau indicator invalid.'), 400
    if kind == 'heatmap' and (column not in snapshot.data or not snapshot.data[column].notna().any()):
        return jsonify(error='Nu există măsurători pentru acest indicator.'), 422
    try:
        obj = snapshot.visualizer.create_heatmap(column) if kind == 'heatmap' else snapshot.visualizer.create_simplified_map()
        path = snapshot.visualizer.save_map(obj, f'map_{uuid.uuid4().hex}.html')
        return jsonify(map_url=f'/maps/{Path(path).name}')
    except Exception:
        logger.exception('Map generation failed')
        return jsonify(error='Harta nu a putut fi creată.'), 500


@app.get('/maps/<filename>')
def serve_map(filename):
    if Path(filename).suffix.lower() != '.html':
        return '', 404
    return send_from_directory(config.MAPS_DIR, filename)


@app.get('/visualizations/<filename>')
def serve_visualization(filename):
    if Path(filename).suffix.lower() not in ('.png', '.jpg', '.jpeg'):
        return '', 404
    return send_from_directory(Path(config.BASE_DIR) / 'visualizations', filename)


def main():
    parser = argparse.ArgumentParser(description='Asistent Date Urbane București')
    parser.add_argument('--host', default=config.HOST)
    parser.add_argument('-p', '--port', type=int, default=config.PORT)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-f', '--force-update', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.force_update:
        # Run through the same startup/refresh path while honoring the explicit flag.
        global _started
        _started = True
        threading.Thread(target=load_data, args=(True,), daemon=True).start()
        threading.Thread(target=_periodic_refresh, daemon=True).start()
    create_app()
    host = '127.0.0.1' if args.debug else args.host
    app.run(host=host, port=args.port, debug=args.debug, use_reloader=False, threaded=True)


if __name__ == '__main__':
    main()
