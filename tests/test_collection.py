import json
from threading import Event

import geopandas as gpd
import pandas as pd
import pytest
import requests

from src import config, source_health
from src.air_source import collect_air_quality
from src.data_collection import collect_all_data, _get_airly_measurements
from src.evidence import fresh_air_mask
from src.osm_source import download_features


def observations(parks):
    return [{'id': index, 'longitude': point.x, 'latitude': point.y,
             'aqi': 20 + index, 'air_observed_at': pd.Timestamp.now(tz='UTC').isoformat()}
            for index, point in enumerate(parks.geometry)]


def test_legacy_cache_refreshes_even_when_file_is_young(parks, monkeypatch):
    old = parks.copy()
    old['aqi'] = [10, 20]
    old.to_file(config.AIR_QUALITY_FILE, driver='GeoJSON')
    monkeypatch.setattr(config, 'AIRLY_API_KEY', 'test-key')
    calls = []
    result = collect_air_quality(False, lambda key: calls.append('installations') or [1, 2],
                                 lambda installations, key: observations(parks))
    assert calls == ['installations']
    assert fresh_air_mask(result).all()
    assert source_health.snapshot()['air_quality']['state'] == 'ready'
    assert 'air_observed_at' in gpd.read_file(config.AIR_QUALITY_FILE)


def test_fresh_observation_cache_does_not_consume_api_quota(parks, monkeypatch):
    monkeypatch.setattr(config, 'AIRLY_API_KEY', 'test-key')
    collect_air_quality(False, lambda key: [1, 2], lambda installations, key: observations(parks))
    def forbidden(*args):
        pytest.fail('Fresh cache must not call Airly')
    result = collect_air_quality(False, forbidden, forbidden)
    assert len(result) == 2
    assert source_health.snapshot()['air_quality']['origin'] == 'cache'


def test_failed_refresh_keeps_good_air_cache(parks, monkeypatch):
    monkeypatch.setattr(config, 'AIRLY_API_KEY', 'test-key')
    collect_air_quality(False, lambda key: [1, 2], lambda installations, key: observations(parks))
    def offline(*args):
        raise requests.ConnectionError('offline')
    result = collect_air_quality(True, offline, offline)
    assert fresh_air_mask(result).all()
    assert source_health.snapshot()['air_quality']['state'] == 'cached'


def test_partial_air_download_is_reported(parks, monkeypatch):
    monkeypatch.setattr(config, 'AIRLY_API_KEY', 'test-key')
    collect_air_quality(False, lambda key: [1, 2, 3], lambda installations, key: observations(parks))
    assert source_health.snapshot()['air_quality']['state'] == 'partial'


def test_all_providers_run_even_when_parks_fail(parks, monkeypatch):
    air_started, weather_started = Event(), Event()
    def parks_loader(force):
        assert air_started.wait(2) and weather_started.wait(2)
        raise RuntimeError('Overpass is busy')
    monkeypatch.setattr('src.data_collection.fetch_parks_and_sectors', parks_loader)
    monkeypatch.setattr('src.data_collection.fetch_air_quality_data', lambda force: air_started.set() or parks)
    monkeypatch.setattr('src.data_collection.fetch_weather_forecast', lambda: weather_started.set() or {'hourly': parks})
    result = collect_all_data()
    assert result['parks_sectors'] is None
    assert result['air_quality'] is parks
    assert result['weather_forecast']['hourly'] is parks


def test_quota_failure_stops_measurement_loop(monkeypatch):
    calls = []
    response = requests.Response()
    response.status_code = 429
    monkeypatch.setattr('src.data_collection.requests.get', lambda *args, **kwargs: calls.append(1) or response)
    result = _get_airly_measurements([{'id': 1}, {'id': 2}], 'test-key')
    assert result == []
    assert len(calls) == 1


def test_mixed_iso_timestamp_precision_remains_usable(parks):
    from datetime import datetime, timezone, timedelta
    parks['aqi'] = [10, 20]
    now = pd.Timestamp(datetime.now(timezone.utc) - timedelta(seconds=1))
    parks['air_observed_at'] = [now.isoformat(), now.floor('s').isoformat()]
    assert fresh_air_mask(parks).all()


XML = b'''<osm version="0.6"><node id="1" lat="44.43" lon="26.09"/><node id="2" lat="44.44" lon="26.09"/><node id="3" lat="44.44" lon="26.10"/><way id="5"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="1"/><tag k="leisure" v="park"/><tag k="name" v="Test Park"/></way></osm>'''


@pytest.mark.parametrize('first_status,first_content', [(502, b'busy'), (200, b'<osm><remark>timeout</remark></osm>')])
def test_overpass_fallback_handles_http_and_partial_xml(monkeypatch, first_status, first_content):
    monkeypatch.setattr(config, 'OVERPASS_ENDPOINTS', ('https://primary.invalid/api', 'https://fallback.invalid/api'))
    monkeypatch.setattr('src.osm_source._preferred_endpoint', None)
    monkeypatch.setattr('src.osm_source.time.sleep', lambda delay: None)
    calls = []
    def response(url, **kwargs):
        calls.append((url, kwargs))
        result = requests.Response()
        result.status_code = first_status if len(calls) == 1 else 200
        result._content = first_content if len(calls) == 1 else XML
        return result
    monkeypatch.setattr('src.osm_source.requests.post', response)
    result = download_features('query', {'leisure': 'park'})
    assert len(result) == 1
    assert result.geometry.iloc[0].geom_type == 'Polygon'
    assert calls[1][0] == 'https://fallback.invalid/api'
    assert calls[0][1]['timeout'] == (5, config.OSM_REQUEST_TIMEOUT)


def test_source_status_never_contains_api_credentials(parks, monkeypatch):
    monkeypatch.setattr(config, 'AIRLY_API_KEY', 'secret-test-key')
    def failed(*args):
        raise RuntimeError('exception text might contain secret-test-key')
    collect_air_quality(False, failed, failed)
    assert 'secret-test-key' not in json.dumps(source_health.snapshot())


def test_retry_after_is_persisted_and_blocks_manual_refresh(parks, monkeypatch):
    from src import air_budget
    monkeypatch.setattr(config, 'AIRLY_API_KEY', 'test-key')
    collect_air_quality(False, lambda key: [1, 2], lambda installations, key: observations(parks))
    response = requests.Response()
    response.status_code = 429
    response.headers['Retry-After'] = '3600'
    response.headers['x-ratelimit-remaining-day'] = '0'
    air_budget.observe(response, 'test-key')
    def forbidden(*args):
        pytest.fail('Manual refresh must respect provider cooldown')
    result = collect_air_quality(True, forbidden, forbidden)
    assert fresh_air_mask(result).all()
    assert source_health.snapshot()['air_quality']['retry_at']
    from pathlib import Path
    quota_text = Path(config.AIRLY_QUOTA_FILE).read_text(encoding='utf-8')
    assert 'test-key' not in quota_text


def test_sensor_selection_respects_budget_and_spreads_coverage(parks):
    from src.air_source import select_installations
    parks.to_file(config.PARKS_SECTORS_FILE, driver='GeoJSON')
    installations = [{'id': index, 'location': {'longitude': 26.09 + index * 0.05, 'latitude': 44.43}} for index in range(10)]
    selected = select_installations(installations, 3)
    assert len(selected) == 3
    assert len({item['id'] for item in selected}) == 3
    assert any(item['id'] == 0 for item in selected)


def test_new_air_batch_preserves_other_recent_cached_sensors(parks, monkeypatch):
    monkeypatch.setattr(config, 'AIRLY_API_KEY', 'test-key')
    collect_air_quality(False, lambda key: [1, 2], lambda installations, key: observations(parks))
    result = collect_air_quality(True, lambda key: [1], lambda installations, key: observations(parks)[:1])
    assert len(result) == 2
    assert fresh_air_mask(result).all()
