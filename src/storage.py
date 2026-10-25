"""Publish cache files only after a complete successful write."""
from pathlib import Path
from uuid import uuid4


def save_geojson(data, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f'.{destination.stem}-{uuid4().hex}.geojson')
    try:
        data.to_file(temporary, driver='GeoJSON')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
