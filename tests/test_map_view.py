from src.map_view import create_map
from src.visualization import UrbanVisualizer


def test_map_preserves_referrer_and_visible_provider_attribution():
    html = create_map(location=[44.43, 26.10]).get_root().render()
    assert '<meta name="referrer" content="strict-origin-when-cross-origin">' in html
    assert '"referrerPolicy": "strict-origin-when-cross-origin"' in html
    assert 'openstreetmap.org/copyright' in html
    assert 'contributors' in html


def test_home_map_uses_compliant_tile_configuration(parks):
    html = UrbanVisualizer(parks).create_simplified_map().get_root().render()
    assert '"referrerPolicy": "strict-origin-when-cross-origin"' in html
    assert 'https://tile.openstreetmap.org/{z}/{x}/{y}.png' in html


def test_no_basemap_does_not_create_tile_requests():
    html = create_map(location=[44.43, 26.10], tiles=None).get_root().render()
    assert 'L.tileLayer(' not in html
