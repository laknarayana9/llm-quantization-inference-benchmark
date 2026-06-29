from __future__ import annotations

import json
import re

_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
          "object": dict, "array": list}


def extract_json(text: str) -> dict | None:
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    start = t.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(t[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def is_valid(text: str, schema: dict) -> bool:
    obj = extract_json(text)
    if obj is None:
        return False
    for key in schema.get("required", []):
        if key not in obj:
            return False
    for key, spec in schema.get("properties", {}).items():
        if key in obj and spec.get("type") in _TYPES:
            if not isinstance(obj[key], _TYPES[spec["type"]]):
                return False
    return True


def validity_rate(outputs: list[str], schemas: list[dict]) -> float:
    if not outputs:
        return 0.0
    return sum(is_valid(o, s) for o, s in zip(outputs, schemas)) / len(outputs)
