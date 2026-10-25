from types import SimpleNamespace

import pytest

import app as web


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(web, '_snapshot', None)
    monkeypatch.setattr(web, 'is_updating', False)
    monkeypatch.setattr(web, 'last_error', None)
    web.sessions.clear()
    web._request_log.clear()
    web.app.config['TESTING'] = True
    return web.app.test_client()


def ready(monkeypatch, parks):
    histories = []
    def query(text, history):
        histories.append(list(history))
        return {'response': 'Răspuns', 'history': history + [text, 'Răspuns']}
    snapshot = web.Snapshot(parks, SimpleNamespace(process_query=query), None, '2026-10-25T10:00:00Z', 'parks.html')
    monkeypatch.setattr(web, '_snapshot', snapshot)
    return histories, snapshot


@pytest.mark.parametrize('payload', [[], 'hi', 1, None, {'query': 20}, {'query': ''}])
def test_invalid_payloads_are_400(client, payload):
    assert client.post('/api/query', json=payload).status_code == 400


def test_initial_loading_has_retry_hint(client):
    response = client.post('/api/query', json={'query': 'parcuri'})
    assert response.status_code == 503
    assert response.headers['Retry-After'] == '5'
    assert client.get('/api/status').json['ready'] is False


def test_refresh_keeps_queries_available(client, monkeypatch, parks):
    ready(monkeypatch, parks)
    monkeypatch.setattr(web, 'is_updating', True)
    assert client.post('/api/query', json={'query': 'parcuri'}).status_code == 200


def test_histories_are_isolated_and_can_be_reset(client, monkeypatch, parks):
    histories, _ = ready(monkeypatch, parks)
    client.post('/api/query', json={'query': 'first'})
    client.post('/api/query', json={'query': 'follow up'})
    other = web.app.test_client()
    other.post('/api/query', json={'query': 'different visitor'})
    client.post('/api/query', json={'query': 'new', 'reset_session': True})
    assert histories == [[], ['first', 'Răspuns'], [], []]


def test_failed_refresh_keeps_previous_snapshot(client, monkeypatch, parks):
    _, previous = ready(monkeypatch, parks)
    def failed(**kwargs):
        raise RuntimeError('provider unavailable')
    monkeypatch.setattr(web, 'collect_all_data', failed)
    assert web.load_data() is False
    assert web._snapshot is previous
    assert web.is_updating is False
    assert web._load_lock.acquire(blocking=False)
    web._load_lock.release()
    assert client.get('/api/status').json['message']


def test_failed_initialization_cannot_publish_partial_snapshot(client, monkeypatch, parks):
    _, previous = ready(monkeypatch, parks)
    def failed(*args):
        raise RuntimeError('agent initialization failed')
    monkeypatch.setattr(web, 'UrbanDataAgent', failed)
    with pytest.raises(RuntimeError):
        web.publish(parks)
    assert web._snapshot is previous


def test_private_data_is_not_served_as_maps(client, monkeypatch, tmp_path):
    monkeypatch.setattr(web.config, 'MAPS_DIR', str(tmp_path))
    (tmp_path / 'private.geojson').write_text('private')
    (tmp_path / 'map.html').write_text('<html>map</html>')
    assert client.get('/maps/private.geojson').status_code == 404
    assert client.get('/maps/../private.geojson').status_code == 404
    assert client.get('/maps/map.html').status_code == 200


def test_status_and_home_use_actual_sector_counts(client, monkeypatch, parks):
    ready(monkeypatch, parks)
    assert len(client.get('/api/status').json['sectors']) == 2
    assert client.get('/').status_code == 200
    assert b'Verde Bucure' in client.get('/').data


def test_map_input_is_validated(client, monkeypatch, parks):
    ready(monkeypatch, parks)
    assert client.get('/api/map?type=../../bad').status_code == 400
    assert client.get('/api/map?type=heatmap&column=aqi').status_code == 422


def test_wsgi_factory_does_not_start_threads_in_test_mode(client):
    assert web.create_app(start_background=False) is web.app


def test_concurrent_tabs_do_not_lose_history(client, monkeypatch, parks):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    histories, _ = ready(monkeypatch, parks)
    web.sessions['shared'] = web.Conversation()
    barrier = Barrier(2)
    def ask(text):
        visitor = web.app.test_client()
        visitor.set_cookie('sid', 'shared')
        barrier.wait(timeout=3)
        return visitor.post('/api/query', json={'query': text}).status_code
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(ask, ['first tab', 'second tab']))
    assert results == [200, 200]
    assert sorted(map(len, histories)) == [0, 2]
    assert len(web.sessions['shared'].history) == 4
