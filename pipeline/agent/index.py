"""
Structural Index Builder
========================
Precomputes structural signals from a 10-K filing HTML by reusing
functions from extract.py.  The agent queries this index through
tools instead of exploring raw HTML.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Import extraction helpers from archive/extract.py
# ---------------------------------------------------------------------------
_archive_dir = str(Path(__file__).resolve().parent.parent.parent / "archive")
if _archive_dir not in sys.path:
    sys.path.insert(0, _archive_dir)

from extract import (  # noqa: E402
    extract_10k_text,
    collect_toc_referenced_ids,
    find_all_id_elements,
    parse_toc_links,
    normalize_text,
    ITEM_PATTERNS,
    TITLE_PATTERNS,
    _ANCHOR_ID_PATTERNS,
    _classify_tier1,
    _classify_tier2,
    _detect_part3_incorporation,
    _find_signatures_fallback,
    classify_anchors,
    ITEM_SEQ_ORDER,
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TocLink:
    anchor_id: str
    link_text: str
    classified_item: str | None


@dataclass(frozen=True)
class Candidate:
    anchor_id: str
    char_position: int
    relative_position: float
    confidence: int          # 0-10
    source: str              # "anchor_id", "toc_text", "tier1", "tier2", "title_keyword"
    nearby_text: str
    tag_name: str


@dataclass(frozen=True)
class PartBoundary:
    part_number: int         # 1, 2, 3, or 4
    char_position: int
    relative_position: float


@dataclass(frozen=True)
class DPAssignment:
    """One assignment from the DP solver (archive baseline)."""
    item_name: str
    anchor_id: str
    char_position: int


@dataclass(frozen=True)
class StructuralIndex:
    html: str
    html_length: int
    accession: str
    toc_links: list[TocLink]
    toc_referenced_ids: set[str]
    anchor_positions: dict[str, tuple]   # anchor_id -> (offset, tag_name, attr_name)
    item_candidates: dict[str, list[Candidate]]  # item_name -> scored candidates
    part_boundaries: list[PartBoundary]
    part3_incorporated: bool             # whether Part III is incorporated by reference
    dp_assignments: list[DPAssignment] | None = None  # DP solver's proposed assignments


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_PART_HEADING_RE = re.compile(
    r'<(?:div|p|h[1-6]|b|strong|span|td|tr)[^>]*>'
    r'\s*(?:<(?:b|strong|font|span)[^>]*>\s*)*'
    r'PART\s+(I{1,3}V?|IV|[1-4])\b',
    re.I,
)

_PART_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4}
_PART_ARABIC = {"1": 1, "2": 2, "3": 3, "4": 4}


def _classify_anchor_id(anchor_id: str) -> str | None:
    """Classify an anchor by its ID string using _ANCHOR_ID_PATTERNS."""
    for item_name, pattern in _ANCHOR_ID_PATTERNS:
        if pattern.search(anchor_id):
            return item_name
    return None


def _extract_nearby_text(html: str, offset: int, window: int = 300) -> str:
    """Extract and clean text near an offset for human-readable context."""
    start = max(0, offset)
    end = min(len(html), offset + window)
    raw = html[start:end]
    text = re.sub(r'<[^>]+>', ' ', raw)
    return normalize_text(text)[:200]


# ---------------------------------------------------------------------------
# Candidate collection (adapted from classify_anchors in extract.py)
# ---------------------------------------------------------------------------

def _collect_all_candidates(
    html: str,
    id_positions: dict,
    toc_mappings: dict | None,
    anchor_to_items: dict | None,
) -> dict[str, list[Candidate]]:
    """
    Run the same tiered classification as classify_anchors but collect
    ALL candidates per item with confidence scores, rather than picking
    winners via the DP assignment.

    Returns {item_name: [Candidate, ...]}.
    """
    html_length = len(html)
    sorted_anchors = sorted(id_positions.items(), key=lambda x: x[1][0])
    candidates: dict[str, list[Candidate]] = {}

    for idx, (anchor_id, anchor_meta) in enumerate(sorted_anchors):
        offset, tag_name, _attr_name = anchor_meta

        # Determine bounded forward window
        if idx + 1 < len(sorted_anchors):
            next_offset = sorted_anchors[idx + 1][1][0]
            max_forward = min(2000, next_offset - offset)
        else:
            max_forward = 2000

        lookahead_html = html[offset:offset + max_forward]
        lookahead_text = normalize_text(re.sub(r'<[^>]+>', ' ', lookahead_html))

        back_start = max(0, offset - 500)
        back_html = html[back_start:offset]
        back_text = normalize_text(re.sub(r'<[^>]+>', ' ', back_html))

        nearby = _extract_nearby_text(html, offset)
        rel_pos = offset / html_length if html_length > 0 else 0.0

        # Collect every classification that matches, not just the first.
        # Each hit becomes a separate Candidate.

        # --- Tier 0a: Anchor ID classification ---
        id_class = _classify_anchor_id(anchor_id)
        if id_class:
            _add_candidate(candidates, id_class, Candidate(
                anchor_id=anchor_id,
                char_position=offset,
                relative_position=rel_pos,
                confidence=9,
                source="anchor_id",
                nearby_text=nearby,
                tag_name=tag_name,
            ))

        # --- Tier 0b: TOC link text ---
        if toc_mappings:
            for toc_item, toc_aid in toc_mappings.items():
                if toc_aid == anchor_id:
                    _add_candidate(candidates, toc_item, Candidate(
                        anchor_id=anchor_id,
                        char_position=offset,
                        relative_position=rel_pos,
                        confidence=8,
                        source="toc_text",
                        nearby_text=nearby,
                        tag_name=tag_name,
                    ))

        # --- Tier 1: "Item X" regex on nearby text ---
        text_150 = lookahead_text[:150]
        t1_name = _classify_tier1(text_150)
        if t1_name:
            _add_candidate(candidates, t1_name, Candidate(
                anchor_id=anchor_id,
                char_position=offset,
                relative_position=rel_pos,
                confidence=6,
                source="tier1",
                nearby_text=nearby,
                tag_name=tag_name,
            ))

        # Also check the full lookahead at a lower confidence
        if not t1_name:
            t1_full = _classify_tier1(lookahead_text)
            if t1_full:
                _add_candidate(candidates, t1_full, Candidate(
                    anchor_id=anchor_id,
                    char_position=offset,
                    relative_position=rel_pos,
                    confidence=5,
                    source="tier1",
                    nearby_text=nearby,
                    tag_name=tag_name,
                ))

        # --- Tier 2: Descriptive title keywords ---
        t2_name = _classify_tier2(text_150)
        if t2_name:
            _add_candidate(candidates, t2_name, Candidate(
                anchor_id=anchor_id,
                char_position=offset,
                relative_position=rel_pos,
                confidence=3,
                source="tier2",
                nearby_text=nearby,
                tag_name=tag_name,
            ))

        # Broader combined context at lower confidence
        combined = back_text[-200:] + " " + lookahead_text[:500]
        t2_combined = _classify_tier2(combined)
        if t2_combined and t2_combined != t2_name:
            _add_candidate(candidates, t2_combined, Candidate(
                anchor_id=anchor_id,
                char_position=offset,
                relative_position=rel_pos,
                confidence=1,
                source="title_keyword",
                nearby_text=nearby,
                tag_name=tag_name,
            ))

    # Sort each item's candidates: highest confidence first, then by position
    for item_name in candidates:
        candidates[item_name] = sorted(
            candidates[item_name],
            key=lambda c: (-c.confidence, c.char_position),
        )

    return candidates


# ---------------------------------------------------------------------------
# Part-region confidence adjustment
# ---------------------------------------------------------------------------

# Which Part each item belongs to
_ITEM_TO_PART = {
    "item1": 1, "item1a": 1, "item1b": 1, "item2": 1, "item3": 1, "item4": 1,
    "item5": 2, "item6": 2, "item7": 2, "item7a": 2, "item8": 2,
    "item9": 2, "item9a": 2, "item9b": 2, "item9c": 2,
    "item10": 3, "item11": 3, "item12": 3, "item13": 3, "item14": 3,
    "item15": 4, "item16": 4,
    "signatures": 4, "crossReference": 4,
}


def _apply_part_region_scoring(
    candidates: dict[str, list[Candidate]],
    part_boundaries: list[PartBoundary],
    html_length: int,
) -> dict[str, list[Candidate]]:
    """Adjust candidate confidence based on Part-region position.

    +1 confidence if candidate falls in the expected Part region.
    -2 confidence if candidate falls in the wrong Part region.
    Only applies to candidates with confidence < 9 (don't override
    high-confidence anchor_id/TOC matches).
    """
    if not part_boundaries or html_length == 0:
        return candidates

    # Build Part region ranges: {part_number: (start, end)}
    regions = {}
    sorted_parts = sorted(part_boundaries, key=lambda b: b.char_position)
    for i, pb in enumerate(sorted_parts):
        start = pb.char_position
        end = sorted_parts[i + 1].char_position if i + 1 < len(sorted_parts) else html_length
        regions[pb.part_number] = (start, end)

    adjusted = {}
    for item_name, cands in candidates.items():
        expected_part = _ITEM_TO_PART.get(item_name)
        if expected_part is None or expected_part not in regions:
            adjusted[item_name] = cands
            continue

        region_start, region_end = regions[expected_part]
        new_cands = []
        for c in cands:
            if c.confidence >= 9:
                # Don't touch high-confidence matches
                new_cands.append(c)
                continue

            in_region = region_start <= c.char_position < region_end
            if in_region:
                # Boost: candidate is in expected Part region
                new_conf = min(c.confidence + 1, 8)
            else:
                # Penalize: candidate is in wrong Part region
                new_conf = max(c.confidence - 2, 0)

            new_cands.append(Candidate(
                anchor_id=c.anchor_id,
                char_position=c.char_position,
                relative_position=c.relative_position,
                confidence=new_conf,
                source=c.source,
                nearby_text=c.nearby_text,
                tag_name=c.tag_name,
            ))

        # Re-sort after adjustment
        new_cands.sort(key=lambda c: (-c.confidence, c.char_position))
        adjusted[item_name] = new_cands

    return adjusted


def _add_candidate(
    store: dict[str, list[Candidate]],
    item_name: str,
    candidate: Candidate,
) -> None:
    """Append a candidate, deduplicating by (anchor_id, source)."""
    items = store.setdefault(item_name, [])
    for existing in items:
        if (existing.anchor_id == candidate.anchor_id
                and existing.source == candidate.source):
            return
    items.append(candidate)


# ---------------------------------------------------------------------------
# TOC link extraction
# ---------------------------------------------------------------------------

def _build_toc_links(
    html: str,
    referenced_ids: set[str],
) -> list[TocLink]:
    """Parse all TOC <a href="#..."> links into TocLink records."""
    links: list[TocLink] = []
    seen: set[tuple[str, str]] = set()

    pattern = re.compile(
        r'<a\s[^>]*href=["\']#([^"\'>\s]+)["\'][^>]*>(.*?)</a>',
        re.I | re.DOTALL,
    )
    for m in pattern.finditer(html):
        anchor_id = m.group(1)
        if anchor_id not in referenced_ids:
            continue

        link_html = m.group(2)
        link_text = normalize_text(re.sub(r'<[^>]+>', ' ', link_html))
        if not link_text or len(link_text) < 2:
            continue

        key = (anchor_id, link_text)
        if key in seen:
            continue
        seen.add(key)

        # Classify the link text
        classified = _classify_tier1(link_text) or _classify_tier2(link_text)

        links.append(TocLink(
            anchor_id=anchor_id,
            link_text=link_text,
            classified_item=classified,
        ))

    return links


# ---------------------------------------------------------------------------
# Part boundary detection
# ---------------------------------------------------------------------------

def _detect_part_boundaries(html: str) -> list[PartBoundary]:
    """
    Best-effort scan for PART I/II/III/IV headings in bold or heading
    elements.  Returns sorted list; empty if none found.
    """
    html_length = len(html)
    if html_length == 0:
        return []

    boundaries: list[PartBoundary] = []
    seen_parts: set[int] = set()

    for m in _PART_HEADING_RE.finditer(html):
        raw_numeral = m.group(1).strip().lower()
        part_num = _PART_ROMAN.get(raw_numeral) or _PART_ARABIC.get(raw_numeral)
        if part_num is None:
            continue

        # Keep the last occurrence of each part (body, not TOC)
        offset = m.start()
        rel_pos = offset / html_length

        # Skip very early occurrences (likely TOC) unless it's the only one
        if rel_pos < 0.05 and part_num in seen_parts:
            continue

        # Replace earlier occurrence with this one (prefer later = body)
        if part_num in seen_parts:
            boundaries = [b for b in boundaries if b.part_number != part_num]

        boundaries.append(PartBoundary(
            part_number=part_num,
            char_position=offset,
            relative_position=rel_pos,
        ))
        seen_parts.add(part_num)

    boundaries.sort(key=lambda b: b.char_position)
    return boundaries


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_structural_index(filepath: str, accession: str) -> StructuralIndex:
    """Build the complete structural index for a filing."""
    html = extract_10k_text(filepath)
    html_length = len(html)

    # Step 1: TOC-referenced anchor IDs
    referenced_ids = collect_toc_referenced_ids(html)

    # Step 2: Locate anchor elements
    id_positions = find_all_id_elements(html, referenced_ids)

    # Step 3: Parse TOC links for direct mappings
    toc_mappings, anchor_to_items = parse_toc_links(html, referenced_ids)

    # Step 4: Build TOC link records
    toc_links = _build_toc_links(html, referenced_ids)

    # Step 5: Collect all candidates (no DP winner selection)
    item_candidates = _collect_all_candidates(
        html, id_positions, toc_mappings, anchor_to_items,
    )

    # Step 6: Part boundary detection
    part_boundaries = _detect_part_boundaries(html)

    # Step 6b: Apply Part-region scoring to candidates
    item_candidates = _apply_part_region_scoring(
        item_candidates, part_boundaries, html_length,
    )

    # Step 7: Part III incorporation check
    # Build the set of items that have at least one high-confidence candidate
    found_items: set[str] = set()
    for item_name, cands in item_candidates.items():
        if any(c.confidence >= 5 for c in cands):
            found_items.add(item_name)

    part3_incorporated = bool(_detect_part3_incorporation(html, found_items))

    # Step 8: Run the DP solver for a proposed assignment
    dp_assignments = None
    try:
        dp_result = classify_anchors(html, id_positions, toc_mappings, anchor_to_items)
        dp_assignments = [
            DPAssignment(
                item_name=item_name,
                anchor_id=anchor_id,
                char_position=offset,
            )
            for offset, item_name, anchor_id in dp_result
        ]
    except Exception:
        pass

    return StructuralIndex(
        html=html,
        html_length=html_length,
        accession=accession,
        toc_links=toc_links,
        toc_referenced_ids=referenced_ids,
        anchor_positions=id_positions,
        item_candidates=item_candidates,
        part_boundaries=part_boundaries,
        part3_incorporated=part3_incorporated,
        dp_assignments=dp_assignments,
    )
