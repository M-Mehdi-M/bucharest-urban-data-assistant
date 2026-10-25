"""Bounded OSM feature downloads, independent of OSMnx's recursive HTTP retries."""
from pathlib import Path
import tempfile
import time
import xml.etree.ElementTree as ET

import geopandas as gpd
import osmnx as ox
import requests
from shapely.geometry import shape

from . import config
from .storage import save_geojson
from .source_health import record

_preferred_endpoint = None


def city_boundary():
    path = Path(config.CITY_BOUNDARY_FILE)
    try:
        response = requests.get('https://nominatim.openstreetmap.org/search',
                                params={'q': 'București, România', 'format': 'json', 'polygon_geojson': 1, 'limit': 1},
                                headers={'User-Agent': config.HTTP_USER_AGENT}, timeout=(5, config.OSM_REQUEST_TIMEOUT))
        response.raise_for_status()
        records = response.json()
        if not records or records[0].get('geojson', {}).get('type') not in ('Polygon', 'MultiPolygon'):
            raise ValueError('Nominatim did not return a city polygon')
        boundary = gpd.GeoDataFrame(geometry=[shape(records[0]['geojson'])], crs='EPSG:4326')
        save_geojson(boundary, path)
        return boundary
    except (requests.RequestException, ValueError):
        if path.exists():
            return gpd.read_file(path)
        raise


def download_features(query, tags, polygon=None):
    """Try each configured provider once; HTTP/XML errors cannot become empty success."""
    global _preferred_endpoint
    failures = []
    endpoints = sorted(config.OVERPASS_ENDPOINTS, key=lambda item: item != _preferred_endpoint)
    for index, endpoint in enumerate(endpoints):
        record('parks', 'loading', 'OpenStreetMap: descărcare de geometrie și limite administrative.', provider=endpoint)
        try:
            response = requests.post(endpoint, data={'data': query},
                                     headers={'User-Agent': config.HTTP_USER_AGENT},
                                     timeout=(5, config.OSM_REQUEST_TIMEOUT))
            response.raise_for_status()
            root = ET.fromstring(response.content)
            if root.tag != 'osm' or root.find('remark') is not None:
                raise ValueError('Overpass returned an incomplete query result')
            with tempfile.NamedTemporaryFile(suffix='.osm', delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(response.content)
            try:
                features = ox.features_from_xml(temporary, polygon=polygon, tags=tags)
            finally:
                temporary.unlink(missing_ok=True)
            if features.empty:
                raise ValueError('Overpass returned no matching features')
            _preferred_endpoint = endpoint
            return features
        except (requests.RequestException, ET.ParseError, ValueError) as exc:
            status = getattr(getattr(exc, 'response', None), 'status_code', None)
            failures.append(f'{endpoint}: HTTP {status}' if status else f'{endpoint}: {type(exc).__name__}')
            if index + 1 < len(endpoints):
                time.sleep(1)
    raise RuntimeError('OpenStreetMap indisponibil (' + '; '.join(failures) + ').')
