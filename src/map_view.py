"""Shared browser-map setup that preserves tile attribution and referrers."""
import folium

REFERRER_POLICY = 'strict-origin-when-cross-origin'


def create_map(*args, tiles='OpenStreetMap', **kwargs):
    map_obj = folium.Map(*args, tiles=None, **kwargs)
    # Keep exports and embedded maps compliant even outside the Flask server.
    map_obj.get_root().header.add_child(
        folium.Element(f'<meta name="referrer" content="{REFERRER_POLICY}">'),
        name='map_referrer_policy',
    )
    if tiles:
        folium.TileLayer(tiles=tiles, referrer_policy=REFERRER_POLICY).add_to(map_obj)
    return map_obj
