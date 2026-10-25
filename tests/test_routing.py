import networkx as nx
import pytest

from src.routing import RouteService


def graph():
    graph = nx.MultiDiGraph(crs='EPSG:4326')
    graph.add_node(1, x=26.09, y=44.43)
    graph.add_node(2, x=26.095, y=44.435)
    graph.add_edge(1, 2, key=7, length=1200)
    graph.add_edge(1, 2, key=9, length=800)
    return graph


def test_route_selects_correct_parallel_edge_and_real_distance(parks, monkeypatch):
    service = RouteService(parks)
    service.graphs['walking'] = graph()
    monkeypatch.setattr('src.routing.ox.distance.nearest_nodes', lambda graph, x, y: 1 if x < 26.092 else 2)
    result = service.route('izvor', 'cismigiu')
    assert result['distance_meters'] == 800
    assert result['duration_minutes'] == 9.6
    assert result['source'] == 'OpenStreetMap'


def test_missing_network_never_fabricates_route(parks, monkeypatch):
    service = RouteService(parks)
    def failed(*args, **kwargs):
        raise RuntimeError('offline')
    monkeypatch.setattr('src.routing.ox.graph_from_place', failed)
    with pytest.raises(RuntimeError):
        service.route('izvor', 'cismigiu')
    assert not service.graphs


def test_network_type_matches_travel_mode(parks, monkeypatch):
    service = RouteService(parks)
    modes = []
    def build(place, network_type):
        modes.append(network_type)
        return graph()
    monkeypatch.setattr('src.routing.ox.graph_from_place', build)
    monkeypatch.setattr('src.routing.ox.save_graphml', lambda *args: None)
    for mode in ('walking', 'bicycle', 'driving'):
        service.get_graph(mode)
    assert modes == ['walk', 'bike', 'drive']


def test_ambiguous_names_require_clarification(parks):
    with pytest.raises(ValueError, match='Precizează'):
        RouteService(parks).match_park('Parcul')
