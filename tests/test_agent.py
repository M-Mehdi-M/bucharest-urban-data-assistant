import json

from src.agent import UrbanDataAgent


def test_invalid_model_output_requests_clarification(parks, monkeypatch):
    monkeypatch.setattr('src.agent.call_gemini', lambda *args, **kwargs: 'not json')
    result = UrbanDataAgent(parks).process_query('Câte parcuri există?')
    assert 'Precizează' in result['response']
    assert 'Ora curentă' not in result['response']


def test_model_arguments_are_validated_before_execution(parks, monkeypatch):
    monkeypatch.setattr('src.agent.call_gemini', lambda *args, **kwargs: json.dumps({'tool': 'find_parks_with_best_air_quality', 'args': {'top_n': 1000000}}))
    result = UrbanDataAgent(parks).process_query('Parcuri cu aer bun')
    assert 'nu poate fi executată' in result['response']


def test_traffic_word_does_not_hijack_a_route_question(parks, monkeypatch):
    calls = []
    def model(*args, **kwargs):
        calls.append(kwargs)
        return json.dumps({'tool': 'get_total_parks', 'args': {}}) if len(calls) == 1 else 'Două parcuri.'
    monkeypatch.setattr('src.agent.call_gemini', model)
    result = UrbanDataAgent(parks).process_query('Compară parcurile fără trafic')
    assert result['response'] == 'Două parcuri.'
    assert isinstance(calls[0]['json_mode'], dict)


def test_missing_route_provider_returns_unavailable(parks, monkeypatch):
    agent = UrbanDataAgent(parks)
    def failed(*args):
        raise RuntimeError('offline')
    monkeypatch.setattr(agent.route_service, 'route', failed)
    result = agent._find_route_between_parks('Izvor', 'Cișmigiu')
    assert 'nu este disponibilă' in result
    assert 'Grid Road' not in result


def test_park_details_do_not_apply_us_aqi_categories(parks):
    parks['aqi'] = [20, 30]
    text = UrbanDataAgent(parks)._get_park_details('Izvor')
    assert 'Airly CAQI' in text
    assert 'Good' not in text


def test_unknown_tool_requests_clarification(parks, monkeypatch):
    monkeypatch.setattr('src.agent.call_gemini', lambda *args, **kwargs: json.dumps({'tool': 'delete_data', 'args': {}}))
    result = UrbanDataAgent(parks).process_query('Do something')
    assert 'Precizează' in result['response']


def test_snapshot_measurements_expire_before_analysis(parks, monkeypatch):
    parks['aqi'] = [20, 30]
    parks['air_observed_at'] = '2000-01-01T00:00:00Z'
    monkeypatch.setattr('src.agent.call_gemini', lambda *args, **kwargs: json.dumps({'tool': 'get_park_details', 'args': {'park_name': 'Izvor'}}))
    result = UrbanDataAgent(parks).process_query('Detalii despre Izvor')
    assert 'Nu există măsurători' in result['response']
