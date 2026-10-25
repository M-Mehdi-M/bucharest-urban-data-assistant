from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import os
import time

import geopandas as gpd
import pytest
from shapely.geometry import Point

from src.forecast import load_forecast, combine_hourly_forecasts


def fixture_file(tmp_path, offsets):
    today = datetime.now(ZoneInfo('Europe/Bucharest')).date()
    data = gpd.GeoDataFrame({'time': [(today + timedelta(days=offset)).isoformat() for offset in offsets]},
                            geometry=[Point(26, 44)] * len(offsets), crs='EPSG:4326')
    path = tmp_path / 'forecast.geojson'
    data.to_file(path, driver='GeoJSON')
    return path


def test_forecast_removes_past_days_before_today_tomorrow_selection(tmp_path):
    path = fixture_file(tmp_path, [-1, 0, 1])
    data = load_forecast(path)
    assert len(data) == 2
    assert data.iloc[0]['time'] == datetime.now(ZoneInfo('Europe/Bucharest')).date().isoformat()


def test_old_forecast_is_unavailable(tmp_path):
    path = fixture_file(tmp_path, [0, 1])
    old = time.time() - 24 * 3600
    os.utime(path, (old, old))
    with pytest.raises(ValueError):
        load_forecast(path)


def test_forecast_missing_today_is_not_called_today(tmp_path):
    path = fixture_file(tmp_path, [1, 2])
    with pytest.raises(ValueError):
        load_forecast(path)


def test_weather_and_air_are_joined_by_timestamp():
    weather = {'hourly': {'time': ['2026-10-25T10:00', '2026-10-25T11:00'], 'temperature_2m': [20, 21]}}
    air = {'hourly': {'time': ['2026-10-25T11:00'], 'pm10': [12], 'pm2_5': [6], 'european_aqi': [25]}}
    result = combine_hourly_forecasts(weather, air, 26, 44)
    assert result[0]['pm10'] is None
    assert result[1]['pm10'] == 12
    assert result[1]['temperature'] == 21
