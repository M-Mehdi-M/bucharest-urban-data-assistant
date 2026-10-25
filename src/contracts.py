"""Validation and evidence contracts shared by the agent and HTTP interface."""
import inspect
import math
from datetime import datetime, timezone


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def validate_tool_arguments(function, arguments):
    """Reject unexpected, missing, wrongly typed and unbounded tool inputs."""
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be an object")
    signature = inspect.signature(function)
    signature.bind(**arguments)
    integer_bounds = {
        "top_n": (1, 50), "num_parks": (2, 10), "num_locations": (1, 10),
        "days": (1, 5), "page": (1, 1000), "page_size": (1, 100),
        "buffer_distance": (50, 5000),
    }
    for name, value in arguments.items():
        parameter = signature.parameters[name]
        if value is None and parameter.default is None:
            continue
        if name in integer_bounds:
            low, high = integer_bounds[name]
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer between {low} and {high}")
        elif name in ("reverse", "show_poorly_served"):
            if type(value) is not bool:
                raise ValueError(f"{name} must be a boolean")
        elif name in ("criteria", "hours"):
            if not isinstance(value, list) or len(value) > 24:
                raise ValueError(f"{name} must be a list with at most 24 items")
            if any(not isinstance(item, str) or len(item) > 100 for item in value):
                raise ValueError(f"{name} must contain short strings")
            if name == "hours" and any(not item.removesuffix(':00').isdigit() or not 0 <= int(item.removesuffix(':00')) <= 23 for item in value):
                raise ValueError("hours must contain hours from 0 to 23")
        elif not isinstance(value, str) or not value.strip() or len(value) > 200:
            raise ValueError(f"{name} must be a nonempty string of at most 200 characters")
    return arguments


def selection_schema(tools):
    properties = {}
    for tool in tools.values():
        for name, kind in tool['args'].items():
            if kind.startswith('list'):
                properties[name] = {'type': 'ARRAY', 'items': {'type': 'STRING'}}
            else:
                properties[name] = {'type': {'int': 'INTEGER', 'bool': 'BOOLEAN'}.get(kind, 'STRING')}
            # Nullable optional arguments are common in the existing tools.
            parameter = inspect.signature(tool['function']).parameters.get(name)
            if parameter and parameter.default is None:
                properties[name]['nullable'] = True
    return {'type': 'OBJECT', 'properties': {
        'tool': {'type': 'STRING', 'enum': list(tools)},
        'args': {'type': 'OBJECT', 'properties': properties},
        'reasoning': {'type': 'STRING'},
    }, 'required': ['tool', 'args']}


def validate_query(payload, max_length=500):
    if not isinstance(payload, dict) or not isinstance(payload.get("query"), str):
        raise ValueError("Trimite o întrebare ca text în câmpul «query».")
    query = payload["query"].strip()
    if not query or len(query) > max_length:
        raise ValueError(f"Întrebarea trebuie să aibă între 1 și {max_length} caractere.")
    return query


def normalize_park_name(value):
    import unicodedata
    text = unicodedata.normalize("NFKD", str(value))
    return "".join(c for c in text if not unicodedata.combining(c)).casefold().strip()


def finite_number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None
