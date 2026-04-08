"""
Hybrid runner: DP solver + targeted LLM calls for ambiguous items only.

Architecture:
1. Build structural index (includes DP solver result)
2. Validate DP result — if clean, output directly (no LLM)
3. Identify ambiguous items (low confidence, validation issues, bad spans)
4. Make single-shot LLM calls to resolve each ambiguous item
5. Apply post-processing and output

No multi-turn agent loop. LLM is a scalpel, not a sledgehammer.
"""

import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from openai import AsyncOpenAI

from .index import StructuralIndex, build_structural_index, Candidate
from .validation import validate_assignments, check_span_sizes
from .state import AssignmentState, ItemAssignment
from .runner import slice_html_from_assignments
from ..config import OPENROUTER_BASE_URL, LLM_TIMEOUT_SECONDS, ITEM_SEQ_INDEX

_archive_dir = str(Path(__file__).resolve().parent.parent.parent / "archive")
if _archive_dir not in sys.path:
    sys.path.insert(0, _archive_dir)

from extract import _is_placeholder_item16  # noqa: E402


# ---------------------------------------------------------------------------
# Confidence scoring for DP assignments
# ---------------------------------------------------------------------------

@dataclass
class AmbiguousItem:
    """An item that needs LLM resolution."""
    item_name: str
    reason: str
    candidates: list[Candidate]
    current_position: int | None  # DP's assignment, if any
    context_text: str  # text around the boundary for LLM


def _score_dp_confidence(index: StructuralIndex) -> tuple[list[AmbiguousItem], dict]:
    """Score the DP solver's assignments and identify ambiguous items.

    Returns (ambiguous_items, stats).
    """
    if not index.dp_assignments:
        return [], {"dp_count": 0, "confidence": "none"}

    dp_map = {dp.item_name: dp for dp in index.dp_assignments}
    ambiguous = []

    # Build assignment list for validation
    assignments = [
        ItemAssignment(dp.item_name, dp.anchor_id, dp.char_position, "dp")
        for dp in index.dp_assignments
    ]

    # Run validation
    validation = validate_assignments(assignments, index.html_length)
    spans = check_span_sizes(assignments, index.html_length)

    # Check each assigned item for ambiguity
    for dp in index.dp_assignments:
        candidates = index.item_candidates.get(dp.item_name, [])
        reasons = []

        # 1. Low confidence: best candidate has confidence < 7
        if candidates:
            best_conf = max(c.confidence for c in candidates)
            if best_conf < 7:
                reasons.append(f"low_confidence({best_conf})")

            # 2. Ambiguous: multiple candidates with same top confidence
            #    at very different positions (>5% of doc apart)
            top_conf_cands = [c for c in candidates if c.confidence == best_conf]
            if len(top_conf_cands) > 1:
                positions = sorted(set(c.char_position for c in top_conf_cands))
                if len(positions) > 1:
                    spread = max(positions) - min(positions)
                    spread_pct = spread / html_length if html_length > 0 else 0
                    if spread_pct > 0.05:
                        reasons.append(f"ambiguous_position({len(top_conf_cands)}@conf={best_conf},spread={spread_pct:.0%})")

        # 3. Validation issues for this specific item
        for issue in validation.get("issues", []):
            if issue["item"] == dp.item_name:
                reasons.append(f"validation:{issue['type']}")

        # 4. Bad span size (item8 is expected to be large, skip it)
        for span in spans:
            if span["item_name"] == dp.item_name and span["flag"] != "ok":
                if dp.item_name == "item8" and span["flag"] == "large":
                    continue  # item8 is supposed to be the largest section
                reasons.append(f"span:{span['flag']}({span['span_chars']:,})")

        if reasons:
            # Get context text around the DP position
            start = max(0, dp.char_position - 200)
            end = min(index.html_length, dp.char_position + 500)
            raw = index.html[start:end]
            context = re.sub(r'<[^>]+>', ' ', raw)
            context = re.sub(r'\s+', ' ', context).strip()[:400]

            ambiguous.append(AmbiguousItem(
                item_name=dp.item_name,
                reason="; ".join(reasons),
                candidates=candidates[:5],
                current_position=dp.char_position,
                context_text=context,
            ))

    # Also check for missing expected items
    from ..config import EXPECTED_ITEMS
    assigned_names = {dp.item_name for dp in index.dp_assignments}
    for expected in sorted(EXPECTED_ITEMS - assigned_names):
        candidates = index.item_candidates.get(expected, [])
        ambiguous.append(AmbiguousItem(
            item_name=expected,
            reason="missing_expected",
            candidates=candidates[:5],
            current_position=None,
            context_text="",
        ))

    stats = {
        "dp_count": len(index.dp_assignments),
        "validation_issues": len(validation.get("issues", [])),
        "ambiguous_count": len(ambiguous),
        "clean": len(ambiguous) == 0,
    }
    return ambiguous, stats


# ---------------------------------------------------------------------------
# Targeted LLM resolution
# ---------------------------------------------------------------------------

_RESOLVE_PROMPT = """You are resolving an ambiguous item boundary in a SEC 10-K filing.

Item: {item_name}
Problem: {reason}
Current position: {current_pos} ({rel_pos} into the document)

Candidates (sorted by confidence):
{candidates_text}

Text around current boundary:
"{context_text}"

Respond with ONLY a JSON object: {{"action": "keep"|"move"|"remove", "anchor_id": "...", "char_position": N, "reasoning": "..."}}
- "keep" = current assignment is correct
- "move" = reassign to a different candidate (specify which)
- "remove" = this item should not be assigned (incorporated by reference or not present)"""


async def _resolve_ambiguous_item(
    client: AsyncOpenAI,
    model_id: str,
    item: AmbiguousItem,
    html_length: int,
) -> dict | None:
    """Make a single LLM call to resolve one ambiguous item."""
    candidates_text = "\n".join(
        f"  [{i+1}] anchor={c.anchor_id} pos={c.char_position} "
        f"({c.char_position/html_length:.1%}) conf={c.confidence} "
        f"src={c.source} text=\"{c.nearby_text[:100]}\""
        for i, c in enumerate(item.candidates)
    )
    if not candidates_text:
        candidates_text = "  (no candidates found)"

    rel_pos = (
        f"{item.current_position/html_length:.1%}"
        if item.current_position is not None else "unassigned"
    )

    prompt = _RESOLVE_PROMPT.format(
        item_name=item.item_name,
        reason=item.reason,
        current_pos=item.current_position,
        rel_pos=rel_pos,
        candidates_text=candidates_text,
        context_text=item.context_text[:400],
    )

    try:
        response = await client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=256,
            timeout=LLM_TIMEOUT_SECONDS,
        )
        text = response.choices[0].message.content or ""

        # Parse JSON response
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r'```(?:json)?\s*', '', text).rstrip('`').strip()

        result = json.loads(text)
        tokens = 0
        if response.usage:
            tokens = (response.usage.prompt_tokens or 0) + (response.usage.completion_tokens or 0)
        result["_tokens"] = tokens
        return result

    except Exception as e:
        return {"action": "keep", "reasoning": f"LLM error: {e}", "_tokens": 0}


# ---------------------------------------------------------------------------
# Hybrid runner
# ---------------------------------------------------------------------------

@dataclass
class HybridResult:
    accession: str
    predictions: dict[str, str]
    dp_count: int
    ambiguous_count: int
    llm_calls: int
    total_tokens: int
    total_latency_ms: int
    used_llm: bool


async def run_hybrid(
    source_path: str,
    accession: str,
    model_id: str,
    api_key: str,
) -> HybridResult:
    """Run the hybrid pipeline on a single filing."""
    t0 = time.monotonic()

    # Step 1: Build index (includes DP solver)
    index = build_structural_index(source_path, accession)

    # Step 2: Score confidence
    ambiguous, stats = _score_dp_confidence(index)

    total_tokens = 0
    llm_calls = 0

    if not ambiguous and index.dp_assignments:
        # Clean DP result — output directly, no LLM needed
        assignments = [
            ItemAssignment(dp.item_name, dp.anchor_id, dp.char_position, "dp")
            for dp in index.dp_assignments
        ]
    else:
        # Start with DP assignments
        state = AssignmentState()
        for dp in (index.dp_assignments or []):
            state.assign_item(dp.item_name, dp.anchor_id, dp.char_position, "dp")

        # Step 3: Resolve ambiguous items with targeted LLM calls
        if ambiguous:
            client = AsyncOpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=api_key,
            )
            for item in ambiguous:
                result = await _resolve_ambiguous_item(
                    client, model_id, item, index.html_length,
                )
                llm_calls += 1
                if result:
                    total_tokens += result.get("_tokens", 0)
                    action = result.get("action", "keep")

                    if action == "move" and result.get("char_position") is not None:
                        pos = result["char_position"]
                        aid = result.get("anchor_id", "")
                        reasoning = result.get("reasoning", "LLM resolution")
                        if pos >= 0:
                            state.assign_item(item.item_name, aid, pos, reasoning)
                        else:
                            state.assign_item(item.item_name, aid, None, reasoning)

                    elif action == "remove":
                        state.unassign_item(
                            item.item_name,
                            result.get("reasoning", "LLM: item not present"),
                        )

                    # "keep" = do nothing, current assignment stands

        assignments = state.get_assignments()

    # Step 4: Slice and post-process
    predictions = slice_html_from_assignments(
        index.html, assignments, accession,
    )

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    return HybridResult(
        accession=accession,
        predictions=predictions,
        dp_count=stats["dp_count"],
        ambiguous_count=stats["ambiguous_count"],
        llm_calls=llm_calls,
        total_tokens=total_tokens,
        total_latency_ms=elapsed_ms,
        used_llm=llm_calls > 0,
    )
