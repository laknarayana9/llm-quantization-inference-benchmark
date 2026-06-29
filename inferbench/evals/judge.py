from __future__ import annotations

import re
from typing import Callable

JudgeFn = Callable[[str], float]


def parse_score(reply: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)", reply)
    if not m:
        return 0.0
    val = float(m.group(1))
    return max(0.0, min(1.0, val / 10.0 if val > 1.0 else val))


def build_judge(api_complete_fn) -> JudgeFn:
    """api_complete_fn(prompt:str)->str calls a strong model; we parse a 0-1 score."""
    def judge(prompt: str) -> float:
        return parse_score(api_complete_fn(prompt))
    return judge
