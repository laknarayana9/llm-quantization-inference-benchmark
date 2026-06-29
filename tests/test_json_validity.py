from inferbench.evals.json_validity import extract_json, is_valid, validity_rate

SCHEMA = {"type": "object", "required": ["name", "age"],
          "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}


def test_extract_handles_fences():
    assert extract_json('```json\n{"name":"A","age":3}\n```') == {"name": "A", "age": 3}


def test_is_valid_checks_required_and_types():
    assert is_valid('{"name":"A","age":3}', SCHEMA)
    assert not is_valid('{"name":"A"}', SCHEMA)            # missing required
    assert not is_valid('{"name":"A","age":"x"}', SCHEMA)  # wrong type
    assert not is_valid('not json', SCHEMA)


def test_validity_rate():
    assert validity_rate(['{"name":"A","age":3}', 'bad'], [SCHEMA, SCHEMA]) == 0.5
