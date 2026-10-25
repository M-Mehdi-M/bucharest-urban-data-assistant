"""Routes over real, mode-specific OSM networks; no synthetic street fallbacks."""
from pathlib import Path
from html import escape
from uuid import uuid4

import folium
import networkx as nx
import osmnx as ox
from pyproj import Geod

from . import config
from .contracts import normalize_park_name
from .map_view import create_map


class RouteService:
    def __init__(self, data):
        self.data = data
        self.graphs = {}

    def match_park(self, query):
        names = self.data["name"].map(normalize_park_name)
        query = normalize_park_name(query)
        exact = self.data[names == query]
        matches = exact if len(exact) else self.data[names.str.contains(query, regex=False, na=False)]
        if len(matches) != 1:
            choices = ", ".join(matches["name"].head(5))
            raise ValueError(f"Precizează parcul: {choices}" if len(matches) else f"Nu am găsit parcul «{query}».")
        return matches.iloc[0]

    def get_graph(self, mode):
        network_type = {"walking": "walk", "bicycle": "bike", "driving": "drive"}[mode]
        if mode not in self.graphs:
            cache = Path(config.STATIC_DATA_DIR) / f"network_{network_type}.graphml"
            if cache.exists():
                graph = ox.load_graphml(cache)
            else:
                graph = ox.graph_from_place("București, Romania", network_type=network_type)
                ox.save_graphml(graph, cache)
            if not graph.number_of_edges():
                raise ValueError("Rețeaua stradală nu este disponibilă.")
            self.graphs[mode] = graph
        return self.graphs[mode]

    def route(self, origin_name, destination_name, mode="walking"):
        mode = {"cycling": "bicycle", "bike": "bicycle", "walk": "walking", "drive": "driving"}.get(mode, mode)
        if mode not in ("walking", "bicycle", "driving"):
            raise ValueError("Alege walking, bicycle sau driving.")
        origin, destination = self.match_park(origin_name), self.match_park(destination_name)
        if origin.name == destination.name:
            raise ValueError("Alege două parcuri diferite.")
        graph = self.get_graph(mode)
        start, end = origin.geometry.representative_point(), destination.geometry.representative_point()
        nodes = [ox.distance.nearest_nodes(graph, p.x, p.y) for p in (start, end)]
        geod = Geod(ellps="WGS84")
        offsets = [abs(geod.inv(p.x, p.y, graph.nodes[node]["x"], graph.nodes[node]["y"])[2])
                   for p, node in zip((start, end), nodes)]
        if max(offsets) > 1000:
            raise ValueError("Parcul este prea departe de rețeaua disponibilă pentru o rută sigură.")
        path = nx.shortest_path(graph, *nodes, weight="length")
        if len(path) < 2:
            raise ValueError("Parcurile se conectează la același nod; nu pot estima o rută distinctă.")
        coordinates, distance = [], 0.0
        for u, v in zip(path, path[1:]):
            edge = min(graph.get_edge_data(u, v).values(), key=lambda item: item["length"])
            distance += float(edge["length"])
            if edge.get("geometry") is not None:
                coords = list(edge["geometry"].coords)
                # Edge geometry can be stored in the opposite direction.
                first = graph.nodes[u]
                if (coords[-1][0] - first["x"]) ** 2 + (coords[-1][1] - first["y"]) ** 2 < (coords[0][0] - first["x"]) ** 2 + (coords[0][1] - first["y"]) ** 2:
                    coords.reverse()
            else:
                coords = [(graph.nodes[n]["x"], graph.nodes[n]["y"]) for n in (u, v)]
            coordinates.extend((lat, lon) for lon, lat in coords)
        map_obj = create_map(location=coordinates[0], zoom_start=13)
        folium.PolyLine(coordinates, color="#166854", weight=5).add_to(map_obj)
        for point, park in ((start, origin), (end, destination)):
            folium.Marker([point.y, point.x], tooltip=escape(str(park["name"]))).add_to(map_obj)
        map_obj.fit_bounds(coordinates)
        directory = Path(config.MAPS_DIR)
        directory.mkdir(parents=True, exist_ok=True)
        artifact = directory / f"route_{uuid4().hex}.html"
        map_obj.save(str(artifact))
        # Driving time requires real speed data; do not invent it.
        speed = {"walking": 5.0, "bicycle": 15.0}.get(mode)
        return {
            "origin": str(origin["name"]), "destination": str(destination["name"]),
            "mode": mode, "distance_meters": round(distance),
            "duration_minutes": round(distance / (speed * 1000 / 60), 1) if speed else None,
            "snap_distance_meters": [round(value) for value in offsets],
            "source": "OpenStreetMap", "artifact": str(artifact),
            "coordinates": coordinates,
            "limitations": "Rută între nodurile stradale apropiate de punctele reprezentative ale parcurilor; accesul exact la intrare nu este verificat. Durata este estimată, fără trafic în timp real.",
        }
