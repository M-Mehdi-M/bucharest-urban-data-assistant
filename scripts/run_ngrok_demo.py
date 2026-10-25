"""Run and supervise the GitHub Actions demo; never print authentication tokens."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


def check_processes(processes):
    for name, process in processes.items():
        code = process.poll()
        if code is not None:
            raise RuntimeError(f'{name} exited with code {code}. Check the uploaded logs.')


def wait_for_json(url, processes, predicate, timeout=60, alternatives=()):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        check_processes(processes)
        for candidate in (url, *alternatives):
            try:
                with urlopen(candidate, timeout=5) as response:
                    data = json.load(response)
                if predicate(data):
                    return data
            except (URLError, OSError, ValueError):
                pass
        time.sleep(1)
    raise RuntimeError(f'Timed out waiting for {url}. Check the uploaded logs.')


def tunnel_url(payload):
    if not isinstance(payload, dict):
        return None
    for tunnel in payload.get('endpoints', []) + payload.get('tunnels', []):
        url = tunnel.get('url', tunnel.get('public_url', ''))
        if isinstance(url, str) and url.startswith('https://'):
            return url
    return None


def stop_processes(processes):
    for process in reversed(list(processes.values())):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def announce(url, status):
    print(f'Public demo URL: {url}', flush=True)
    print('Data ready.' if status.get('ready') else 'Data is loading; the page shows collection progress.', flush=True)
    summary_path = os.getenv('GITHUB_STEP_SUMMARY')
    if summary_path:
        with open(summary_path, 'a', encoding='utf-8') as handle:
            handle.write(f'## Bucharest Urban Data Assistant\n\n[Open the running demo]({url})\n\n'
                         'The URL remains available while this job runs. Source availability and collection progress appear in the app.\n')
    output_path = os.getenv('GITHUB_OUTPUT')
    if output_path:
        with open(output_path, 'a', encoding='utf-8') as handle:
            handle.write(f'public_url={url}\n')


def run(duration_minutes):
    if not os.getenv('NGROK_AUTHTOKEN'):
        raise RuntimeError('Set the repository secret NGROK_AUTH_TOKEN before running the demo.')
    processes = {}
    environment = {**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONUTF8': '1'}
    with open('server.log', 'w', encoding='utf-8') as server_log, open('ngrok.log', 'w', encoding='utf-8') as ngrok_log:
        try:
            processes['App'] = subprocess.Popen(
                [sys.executable, '-m', 'waitress', '--call', '--host=127.0.0.1', '--port=5000', 'app:create_app'],
                stdout=server_log, stderr=subprocess.STDOUT, env=environment)
            status = wait_for_json('http://127.0.0.1:5000/api/status', processes,
                                   lambda data: isinstance(data, dict) and 'ready' in data)
            processes['ngrok'] = subprocess.Popen(
                ['ngrok', 'http', 'http://127.0.0.1:5000', '--log=stdout', '--log-format=json'],
                stdout=ngrok_log, stderr=subprocess.STDOUT, env=environment)
            tunnel = wait_for_json('http://127.0.0.1:4040/api/endpoints', processes, tunnel_url,
                                   alternatives=('http://127.0.0.1:4040/api/tunnels',))
            announce(tunnel_url(tunnel), status)
            deadline = time.monotonic() + duration_minutes * 60
            while time.monotonic() < deadline:
                check_processes(processes)
                time.sleep(min(10, max(0, deadline - time.monotonic())))
        finally:
            stop_processes(processes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration-minutes', type=int, choices=range(1, 121), default=120)
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parents[1])
    def cancelled(_signum, _frame):
        raise SystemExit(130)
    signal.signal(signal.SIGTERM, cancelled)
    try:
        run(args.duration_minutes)
    except Exception as exc:
        print(f'Demo failed: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1) from exc


if __name__ == '__main__':
    main()
