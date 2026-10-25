"""Read current city forecasts without relabeling yesterday as today."""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import time

import geopandas as gpd
from shapely.geometry import Point


def load_forecast(path):
    path = Path(path)
    if time.time() - path.stat().st_mtime > 12 * 3600:
        raise ValueError('Prognoza din cache este prea veche. Actualizează datele.')
    data = gpd.read_file(path)
    if 'time' not in data:
        raise ValueError('Prognoza nu conține date calendaristice.')
    today = datetime.now(ZoneInfo('Europe/Bucharest')).date().isoformat()
    data['time'] = data['time'].astype(str).str.replace(' ', 'T', regex=False)
    dates = data['time'].str.slice(0, 10)
    data = data[dates >= today].copy().sort_values('time').reset_index(drop=True)
    if data.empty or not data['time'].str.startswith(today).any():
        raise ValueError('Prognoza nu acoperă ziua curentă. Actualizează datele.')
    return data


def combine_hourly_forecasts(weather, air, longitude, latitude):
    """Join predictions by timestamp, never by coincidental array position."""
    weather_hours = weather['hourly']
    air_hours = air['hourly']
    air_positions = {timestamp: index for index, timestamp in enumerate(air_hours['time'])}

    def value(series, field, index):
        values = series.get(field, [])
        return values[index] if index is not None and index < len(values) else None

    records = []
    for index, timestamp in enumerate(weather_hours['time']):
        air_index = air_positions.get(timestamp)
        records.append({
            'time': timestamp, 'geometry': Point(longitude, latitude),
            'temperature': value(weather_hours, 'temperature_2m', index),
            'humidity': value(weather_hours, 'relative_humidity_2m', index),
            'wind_speed': value(weather_hours, 'wind_speed_10m', index),
            'precipitation': value(weather_hours, 'precipitation', index),
            'cloudcover': value(weather_hours, 'cloud_cover', index) if 'cloud_cover' in weather_hours else value(weather_hours, 'cloudcover', index),
            'pm10': value(air_hours, 'pm10', air_index),
            'pm25': value(air_hours, 'pm2_5', air_index),
            'aqi': value(air_hours, 'european_aqi', air_index),
            'forecast_type': 'hourly', 'forecast_source': 'Open-Meteo',
            'air_index': 'European AQI',
        })
    return records
