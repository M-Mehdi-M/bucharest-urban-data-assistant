import pytest

from src.contracts import validate_query, validate_tool_arguments


@pytest.mark.parametrize('payload', [None, [], ['hello'], 'hello', 1, {'query': None}, {'query': {}}, {'query': ' '}, {'query': 'a' * 501}])
def test_reject_invalid_queries(payload):
    with pytest.raises(ValueError):
        validate_query(payload)


def test_trim_query():
    assert validate_query({'query': '  parc  '}) == 'parc'


@pytest.mark.parametrize('arguments', [{'top_n': 0}, {'top_n': True}, {'top_n': 100000}, {'top_n': '5'}, {'unexpected': 1}, []])
def test_reject_invalid_tool_arguments(arguments):
    def tool(top_n=5):
        pass
    with pytest.raises((TypeError, ValueError)):
        validate_tool_arguments(tool, arguments)


def test_optional_and_hour_arguments():
    def tool(day='today', hours=None):
        pass
    assert validate_tool_arguments(tool, {'hours': ['10:00', '23']})
    with pytest.raises(ValueError):
        validate_tool_arguments(tool, {'hours': ['24:00']})
