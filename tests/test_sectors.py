import pytest

from src import config
from src.sectors import report, sector_metrics


def test_density_requires_verified_boundaries(parks):
    with pytest.raises(ValueError):
        report(parks, 'density')


def test_density_uses_polygon_area(parks):
    boundaries = parks.to_crs(config.METRIC_CRS)
    boundaries.geometry = boundaries.geometry.buffer(1000, cap_style=3)
    boundaries[['sector', 'geometry']].to_file(config.SECTORS_FILE, driver='GeoJSON')
    table = sector_metrics(parks)
    assert table.area_sqkm.tolist() == pytest.approx([4, 4])
    assert table.parks_per_sqkm.tolist() == pytest.approx([0.25, 0.25])


def test_population_statistics_are_not_fabricated(parks):
    assert 'population' not in sector_metrics(parks)
    with pytest.raises(ValueError):
        report(parks, 'parks_per_100k_residents')
