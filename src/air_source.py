"""Air cache validity follows observation metadata, not just file modification time."""
from pathlib import Path
import time

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from . import config
from .evidence import fresh_air_mask
from .source_health import record
from .storage import save_geojson
from . import air_budget


def select_installations(installations, budget, cached=None):
    """Spend a small request budget on sensors spread across the park dataset."""
    if len(installations) <= budget:
        return installations
    try:
        sensors = gpd.GeoDataFrame(installations, geometry=[Point(item['location']['longitude'], item['location']['latitude']) for item in installations], crs='EPSG:4326').to_crs(config.METRIC_CRS)
        parks = gpd.read_file(config.PARKS_SECTORS_FILE).to_crs(config.METRIC_CRS).geometry.representative_point()
        distances = np.hypot(parks.x.to_numpy()[:, None] - sensors.geometry.x.to_numpy(),
                             parks.y.to_numpy()[:, None] - sensors.geometry.y.to_numpy())
        coverage = distances <= config.MAX_SENSOR_DISTANCE_METERS
        known = set(cached.loc[fresh_air_mask(cached), 'id']) if cached is not None and 'id' in cached else set()
        selected, covered = [], np.zeros(len(parks), dtype=bool)
        for _ in range(budget):
            scores = coverage[~covered].sum(axis=0).astype(float)
            scores += np.array([0.1 if item['id'] in known else 0 for item in installations])
            # Prefer unrefreshed, spatially distinct sensors after city coverage is met.
            if covered.all() and selected:
                points = np.column_stack([sensors.geometry.x, sensors.geometry.y])
                scores = np.min(np.linalg.norm(points[:, None, :] - points[selected][None, :, :], axis=2), axis=1)
            scores[selected] = -np.inf
            index = int(scores.argmax())
            selected.append(index)
            covered |= coverage[:, index]
        return [installations[index] for index in selected]
    except (KeyError, ValueError, OSError):
        # Sensor order returned by Airly is still valid if park geometry is missing.
        return installations[:budget]


def collect_air_quality(force_update, installations_loader, measurements_loader):
    path = Path(config.AIR_QUALITY_FILE)
    cached = None
    if path.exists():
        try:
            cached = gpd.read_file(path)
        except Exception:
            # A corrupt cache must not stop a new network attempt or other sources.
            pass
    quota = air_budget.blocked()
    if not force_update and cached is not None and not quota:
        age = time.time() - path.stat().st_mtime
        usable = int(fresh_air_mask(cached).sum())
        if age < config.AIR_REFRESH_MINUTES * 60 and usable:
            record('air_quality', 'ready', f'Airly: {usable} senzori cu observații recente în cache.',
                   origin='cache', count=len(cached), usable_sensors=usable)
            return cached

    def fallback(message):
        active_quota = air_budget.blocked()
        usable = int(fresh_air_mask(cached).sum()) if cached is not None else 0
        state = 'cached' if usable else 'unavailable'
        if active_quota and 'limita' not in message:
            message += ' Airly a atins limita de cereri; se respectă pauza furnizorului.'
        record('air_quality', state, message + (' Se folosesc observațiile recente din cache.' if usable else ' Nu există observații recente utilizabile.'),
               origin='cache' if cached is not None else None, count=len(cached) if cached is not None else 0,
               usable_sensors=usable, retry_at=active_quota.get('retry_at') if active_quota else None)
        return cached if cached is not None else gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')

    if quota:
        return fallback('Airly: limita de cereri este atinsă; reîncercarea așteaptă termenul indicat de furnizor.')
    if not config.AIRLY_API_KEY:
        return fallback('Airly: lipsește cheia AIRLY_API_KEY.')
    record('air_quality', 'loading', 'Airly: descărcare de observații recente.')
    try:
        installations = installations_loader(config.AIRLY_API_KEY)
        if not installations:
            return fallback('Airly: nu au fost găsiți senzori în zona configurată.')
        selected = select_installations(installations, air_budget.allowance(), cached)
        measurements = measurements_loader(selected, config.AIRLY_API_KEY)
        if not measurements:
            return fallback('Airly: nu au fost primite măsurători.')
        frame = gpd.GeoDataFrame(measurements,
                                 geometry=[Point(row['longitude'], row['latitude']) for row in measurements],
                                 crs='EPSG:4326')
        usable = int(fresh_air_mask(frame).sum())
        if not usable:
            return fallback('Airly: răspunsul nu conține observații recente cu CAQI valid.')
        downloaded = len(frame)
        if cached is not None and not cached.empty and 'id' in cached:
            frame = gpd.GeoDataFrame(pd.concat([frame, cached], ignore_index=True), geometry='geometry', crs='EPSG:4326').drop_duplicates('id', keep='first')
            usable = int(fresh_air_mask(frame).sum())
        save_geojson(frame, path)
        quota = air_budget.blocked()
        partial = downloaded < len(selected) or quota is not None
        record('air_quality', 'partial' if partial else 'ready',
               f'Airly: {usable} senzori cu observații recente; {downloaded}/{len(selected)} senzori verificați din {len(installations)} instalații găsite.' + (' Descărcarea este parțială.' if partial else ''),
               origin='network', count=len(frame), usable_sensors=usable,
               retry_at=quota.get('retry_at') if quota else None)
        return frame
    except Exception as exc:
        status = getattr(getattr(exc, 'response', None), 'status_code', None)
        reason = f'HTTP {status}' if status else type(exc).__name__
        return fallback(f'Airly: descărcarea a eșuat ({reason}).')
