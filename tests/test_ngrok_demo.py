from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import run_ngrok_demo as demo


def test_startup_accepts_loading_app_without_waiting_for_external_providers(monkeypatch):
    monkeypatch.setattr(demo, 'urlopen', lambda *args, **kwargs: BytesIO(b'{"ready": false}'))
    result = demo.wait_for_json('http://localhost/api/status', {}, lambda data: 'ready' in data)
    assert result == {'ready': False}


def test_dead_process_fails_instead_of_sleeping_for_two_hours():
    with pytest.raises(RuntimeError, match='ngrok exited with code 1'):
        demo.check_processes({'ngrok': SimpleNamespace(poll=lambda: 1)})


def test_wait_times_out_when_agent_never_opens_a_tunnel(monkeypatch):
    clock = iter([0, 0, 2])
    monkeypatch.setattr(demo.time, 'monotonic', lambda: next(clock))
    monkeypatch.setattr(demo.time, 'sleep', lambda duration: None)
    monkeypatch.setattr(demo, 'urlopen', lambda *args, **kwargs: BytesIO(b'{"tunnels": []}'))
    with pytest.raises(RuntimeError, match='Timed out'):
        demo.wait_for_json('http://localhost/tunnels', {}, demo.tunnel_url, timeout=1)


def test_tunnel_selection_requires_https():
    assert demo.tunnel_url({'endpoints': [{'url': 'https://demo.ngrok.app'}]}) == 'https://demo.ngrok.app'
    assert demo.tunnel_url({'tunnels': [{'public_url': 'http://example.test'}, {'public_url': 'https://demo.ngrok.app'}]}) == 'https://demo.ngrok.app'
    assert demo.tunnel_url({'tunnels': []}) is None


def test_older_agent_api_is_supported_when_endpoints_api_is_unavailable(monkeypatch):
    from urllib.error import URLError
    calls = []
    def respond(url, **kwargs):
        calls.append(url)
        if url.endswith('/endpoints'):
            raise URLError('not found')
        return BytesIO(b'{"tunnels": [{"public_url": "https://demo.ngrok.app"}]}')
    monkeypatch.setattr(demo, 'urlopen', respond)
    result = demo.wait_for_json('http://localhost/endpoints', {}, demo.tunnel_url, alternatives=('http://localhost/tunnels',))
    assert demo.tunnel_url(result) == 'https://demo.ngrok.app'
    assert len(calls) == 2


def test_link_is_published_to_job_summary_and_step_output(tmp_path, monkeypatch):
    summary, output = tmp_path / 'summary.md', tmp_path / 'output.txt'
    monkeypatch.setenv('GITHUB_STEP_SUMMARY', str(summary))
    monkeypatch.setenv('GITHUB_OUTPUT', str(output))
    demo.announce('https://demo.ngrok.app', {'ready': False})
    assert '[Open the running demo](https://demo.ngrok.app)' in summary.read_text()
    assert output.read_text() == 'public_url=https://demo.ngrok.app\n'


def test_startup_failure_cleans_up_server_process(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('NGROK_AUTHTOKEN', 'test-token')
    process = Mock()
    process.poll.return_value = None
    monkeypatch.setattr(demo.subprocess, 'Popen', lambda *args, **kwargs: process)
    def failed(*args):
        raise RuntimeError('startup failed')
    monkeypatch.setattr(demo, 'wait_for_json', failed)
    with pytest.raises(RuntimeError, match='startup failed'):
        demo.run(1)
    process.terminate.assert_called_once()
    process.wait.assert_called_once()


def test_missing_token_fails_before_starting_processes(monkeypatch):
    monkeypatch.delenv('NGROK_AUTHTOKEN', raising=False)
    with pytest.raises(RuntimeError, match='NGROK_AUTH_TOKEN'):
        demo.run(1)
