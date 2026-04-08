"""
Validation logic for the agentic 10-K itemization pipeline.

Validates the agent's current assignments and returns specific, actionable
issues the agent can use to self-correct.
"""

from __future__ import annotations

from pipeline.agent.state import ItemAssignment
from pipeline.config import EXPECTED_ITEMS, ITEM_SEQ_INDEX, ITEM_SEQ_ORDER


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_LARGE_SPAN_PCT = 0.40   # flag spans > 40 % of the document
_SMALL_SPAN_CHARS = 100   # flag spans < 100 chars (likely empty)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_assignments(
    assignments: list[ItemAssignment],
    html_length: int,
) -> dict:
    """Validate current assignments and return structured diagnostics.

    Returns
    -------
    dict
        ``is_valid``  : bool -- True when zero issues detected.
        ``issues``    : list[dict] -- Each has *type*, *item*, *description*.
        ``stats``     : dict -- Summary counters.
    """
    issues: list[dict] = []

    seen_names: dict[str, int] = {}
    prev_name: str | None = None
    prev_pos: int | None = None

    for assignment in assignments:
        name = assignment.item_name
        pos = assignment.char_position

        # 1. Duplicate item names
        if name in seen_names:
            issues.append({
                "type": "duplicate",
                "item": name,
                "description": (
                    f"{name} assigned twice "
                    f"(positions {seen_names[name]} and {pos})"
                ),
            })
        seen_names[name] = pos

        # Skip position-based checks for incorporated-by-reference items
        if pos is None:
            prev_name = name
            continue

        # 2. Position within valid range
        if pos < 0 or pos > html_length:
            issues.append({
                "type": "out_of_range",
                "item": name,
                "description": (
                    f"{name} position {pos} outside valid range "
                    f"[0, {html_length}]"
                ),
            })

        # 3. Canonical order check
        if prev_name is not None:
            prev_idx = ITEM_SEQ_INDEX.get(prev_name)
            curr_idx = ITEM_SEQ_INDEX.get(name)
            if prev_idx is not None and curr_idx is not None:
                if curr_idx <= prev_idx:
                    issues.append({
                        "type": "order_violation",
                        "item": name,
                        "description": (
                            f"{name} (seq {curr_idx}) appears after "
                            f"{prev_name} (seq {prev_idx}) -- "
                            "violates canonical ITEM_SEQ_ORDER"
                        ),
                    })

        # 4. Monotonic positions
        if prev_pos is not None and pos <= prev_pos:
            issues.append({
                "type": "non_monotonic",
                "item": name,
                "description": (
                    f"{name} at position {pos} is not after "
                    f"{prev_name} at position {prev_pos}"
                ),
            })

        prev_name = name
        prev_pos = pos

    # 5. Expected items present
    assigned_names = frozenset(seen_names)
    for expected in sorted(EXPECTED_ITEMS):
        if expected not in assigned_names:
            issues.append({
                "type": "missing_expected",
                "item": expected,
                "description": (
                    f"{expected} is expected in most 10-K filings "
                    "but has not been assigned"
                ),
            })

    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "stats": {
            "assigned_count": len(seen_names),
            "expected_min": len(EXPECTED_ITEMS),
            "total_possible": len(ITEM_SEQ_ORDER),
            "issue_count": len(issues),
        },
    }


def check_span_sizes(
    assignments: list[ItemAssignment],
    html_length: int,
) -> list[dict]:
    """Calculate span sizes between consecutive assignments.

    Returns
    -------
    list[dict]
        Each entry has *item_name*, *span_chars*, *span_pct*, and *flag*
        (``"ok"``, ``"large"``, or ``"small"``).
    """
    if not assignments or html_length <= 0:
        return []

    # Filter out incorporated-by-reference items (None positions)
    positioned = [a for a in assignments if a.char_position is not None]
    if not positioned:
        return []

    spans: list[dict] = []

    for i, assignment in enumerate(positioned):
        if i + 1 < len(positioned):
            next_pos = positioned[i + 1].char_position
        else:
            next_pos = html_length

        span_chars = next_pos - assignment.char_position
        span_pct = span_chars / html_length if html_length > 0 else 0.0

        if span_pct > _LARGE_SPAN_PCT:
            flag = "large"
        elif span_chars < _SMALL_SPAN_CHARS:
            flag = "small"
        else:
            flag = "ok"

        spans.append({
            "item_name": assignment.item_name,
            "span_chars": span_chars,
            "span_pct": round(span_pct, 4),
            "flag": flag,
        })

    return spans
