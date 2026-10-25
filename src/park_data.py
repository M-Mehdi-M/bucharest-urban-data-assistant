"""Collect polygon geometry and stable identities instead of discarding them."""
import logging
import re

import geopandas as gpd

from . import config
from .osm_source import city_boundary, download_features
from .storage import save_geojson

logger = logging.getLogger(__name__)


def fetch_park_geometries():
    boundary = city_boundary()
    polygon = boundary.geometry.iloc[0]
    west, south, east, north = boundary.total_bounds
    bbox = f'{south},{west},{north},{east}'
    parks = download_features(
        f'[out:xml][timeout:25];nwr["leisure"="park"]({bbox});(._;>;);out body;',
        tags={'leisure': 'park'}, polygon=polygon)
    parks = parks[parks.geometry.notna() & ~parks.geometry.is_empty].copy()
    parks["park_id"] = [f"osm:{kind}:{identifier}" for kind, identifier in parks.index]
    if "name" not in parks:
        parks["name"] = None
    parks["name"] = parks["name"].fillna("Parc fără nume")
    parks["area_hectares"] = parks.to_crs(config.METRIC_CRS).geometry.area / 10000
    parks.loc[~parks.geometry.geom_type.isin(["Polygon", "MultiPolygon"]), "area_hectares"] = float("nan")
    parks["geometry_source"] = "OpenStreetMap"
    parks = parks[["park_id", "name", "area_hectares", "geometry_source", "geometry"]].reset_index(drop=True)
    sectors = download_features(
        f'[out:xml][timeout:25];rel["boundary"="administrative"]["admin_level"="9"]({bbox});(._;>;);out body;',
        tags={'admin_level': '9'}, polygon=polygon)
    sectors = sectors[sectors.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    sectors["sector"] = sectors["name"].map(lambda name: re.search(r"sector(?:ul)?\s*([1-6])\b", str(name), re.I))
    sectors = sectors[sectors.sector.notna()].copy()
    sectors["sector"] = sectors["sector"].map(lambda match: f"Sector {match.group(1)}")
    sectors = sectors[["sector", "geometry"]].dissolve(by="sector").reset_index()
    if len(sectors) != 6:
        raise ValueError("Could not verify all six sector boundaries")
    save_geojson(sectors, config.SECTORS_FILE)
    # Representative points assign sectors; original park geometry is retained.
    points = parks.copy()
    points.geometry = points.geometry.representative_point()
    joined = gpd.sjoin(points, sectors, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    parks["sector"] = joined["sector"].reindex(parks.index).fillna("Necunoscut")
    return parks.to_crs("EPSG:4326")
