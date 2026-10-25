import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from src import config
from src.data_processing import process_and_integrate_data
from src.recommendations import rank_parks
from src.spatial_analysis import SpatialAnalyst


def air_at(geometry, values, observed=None):
    return gpd.GeoDataFrame({'id': list(range(len(values))), 'aqi': values,
                             'air_observed_at': observed or [pd.Timestamp.now(tz='UTC').isoformat()] * len(values)},
                            geometry=geometry, crs='EPSG:4326')


def test_offline_sensor_does_not_hide_a_functioning_nearby_sensor(parks):
    air = air_at(list(parks.geometry), [None, 20])
    result = process_and_integrate_data({'parks_sectors': parks, 'air_quality': air})
    assert len(result) == 2
    assert result.iloc[0].aqi == 20
    assert result.iloc[0].sensor_id == 1
    assert result.iloc[0].sensor_distance_meters > 0
    assert result.iloc[1].quality_score == 5
    assert set(result.park_id) == set(parks.park_id)


def test_stale_unknown_and_distant_measurements_are_unusable(parks):
    air = air_at([Point(27, 45)], [10], ['2000-01-01T00:00:00Z'])
    result = process_and_integrate_data({'parks_sectors': parks, 'air_quality': air})
    assert result.aqi.isna().all()
    air = air_at(list(parks.geometry), [10, 20]).drop(columns='air_observed_at')
    result = process_and_integrate_data({'parks_sectors': parks, 'air_quality': air})
    assert result.aqi.isna().all()


def test_tied_sensors_keep_one_row_per_identity(parks):
    air = air_at([parks.geometry.iloc[0], parks.geometry.iloc[0]], [10, 20])
    result = process_and_integrate_data({'parks_sectors': parks, 'air_quality': air})
    assert len(result) == len(parks)
    assert result.park_id.is_unique


def test_simulation_does_not_leak_into_default_data(parks):
    noise = gpd.GeoDataFrame({'noise_level_db': [40]}, geometry=[parks.geometry.iloc[0]], crs=parks.crs)
    result = process_and_integrate_data({'parks_sectors': parks, 'noise': noise})
    assert 'noise_level_db' not in result
    assert result.quality_score.isna().all()


def test_same_name_different_osm_ids_survive(parks):
    parks['name'] = 'Parc fără nume'
    result = process_and_integrate_data({'parks_sectors': parks})
    assert len(result) == 2


def test_recommendations_require_recent_nearby_complete_evidence(parks):
    parks['aqi'] = [5, 20]
    parks['sensor_distance_meters'] = [5000, 100]
    parks['air_observed_at'] = pd.Timestamp.now(tz='UTC').isoformat()
    assert rank_parks(parks).iloc[0]['name'] == 'Parcul Cișmigiu'
    with pytest.raises(ValueError):
        rank_parks(parks, ['congestion'])
    parks['air_observed_at'] = '2000-01-01T00:00:00Z'
    with pytest.raises(ValueError):
        rank_parks(parks)


def test_polygon_geometry_and_area_survive_integration(parks):
    polygons = parks.to_crs(config.METRIC_CRS)
    polygons.geometry = polygons.geometry.buffer(100)
    polygons['area_hectares'] = polygons.geometry.area / 10000
    result = process_and_integrate_data({'parks_sectors': polygons.to_crs('EPSG:4326')})
    assert set(result.geometry.geom_type) == {'Polygon'}
    assert result.area_hectares.min() > 3


def test_coverage_uses_local_metres_not_degrees():
    data = gpd.GeoDataFrame({'name': ['Park']}, geometry=[Point(430000, 4920000)], crs=config.METRIC_CRS)
    analyst = SpatialAnalyst(data)
    analyst._cached_bucharest_boundary = gpd.GeoDataFrame(geometry=[box(429000, 4919000, 431000, 4921000)], crs=config.METRIC_CRS)
    result = analyst.calculate_park_accessibility(500)
    assert result['city_area_sqkm'] == 4
    assert result['accessible_area_sqkm'] == pytest.approx(0.78, abs=0.02)
    assert 19 < result['percentage_accessible'] < 20


def test_sector_coverage_requires_verified_boundary(parks):
    analyst = SpatialAnalyst(parks)
    analyst._cached_bucharest_boundary = gpd.GeoDataFrame(geometry=[box(26, 44, 27, 45)], crs=parks.crs)
    assert 'error' in analyst.calculate_park_accessibility(500, region='Sector 1')
