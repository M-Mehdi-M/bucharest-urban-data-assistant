import geopandas as gpd
import pytest
import requests
from shapely.geometry import Point

from src import config
from src import source_health


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    source_health.reset()
    def blocked(*args, **kwargs):
        raise AssertionError('Tests must not access external services')
    monkeypatch.setattr(requests.sessions.Session, 'request', blocked)
    monkeypatch.setattr(config, 'ENABLE_SIMULATED_DATA', False)
    monkeypatch.setattr(config, 'GEMINI_API_KEY', '')
    monkeypatch.setattr(config, 'PROCESSED_DATA_FILE', str(tmp_path / 'processed.geojson'))
    monkeypatch.setattr(config, 'MAPS_DIR', str(tmp_path / 'maps'))
    monkeypatch.setattr(config, 'STATIC_DATA_DIR', str(tmp_path))
    monkeypatch.setattr(config, 'SECTORS_FILE', str(tmp_path / 'sectors.geojson'))
    monkeypatch.setattr(config, 'CITY_BOUNDARY_FILE', str(tmp_path / 'boundary.geojson'))
    monkeypatch.setattr(config, 'AIR_QUALITY_FILE', str(tmp_path / 'air.geojson'))
    monkeypatch.setattr(config, 'AIRLY_QUOTA_FILE', str(tmp_path / 'quota.json'))
    monkeypatch.setattr(config, 'PARKS_SECTORS_FILE', str(tmp_path / 'parks.geojson'))
    monkeypatch.setattr(config, 'WEATHER_FORECAST_DAILY_FILE', str(tmp_path / 'daily.geojson'))
    monkeypatch.setattr(config, 'WEATHER_FORECAST_HOURLY_FILE', str(tmp_path / 'hourly.geojson'))


@pytest.fixture
def parks():
    return gpd.GeoDataFrame({'park_id': ['osm:way:1', 'osm:way:2'],
                             'name': ['Parcul Izvor', 'Parcul Cișmigiu'],
                             'sector': ['Sector 5', 'Sector 1']},
                            geometry=[Point(26.09, 44.43), Point(26.095, 44.435)], crs='EPSG:4326')
