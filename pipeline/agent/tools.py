"""
Tool registry for the agentic 10-K extraction pipeline.

Each tool is a focused, narrow operation — dedicated tools over generic
commands (Claude Code Lesson #3). Tools operate on the precomputed
structural index, not raw HTML.
"""

import json
import re
import html as html_module
from dataclasses import dataclass, asdict

from .state import AssignmentState, ItemAssignment
from .validation import validate_assignments, check_span_sizes
from .index import StructuralIndex


# ---------------------------------------------------------------------------
# Tool schemas for OpenAI-compatible function calling
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_filing_overview",
            "description": "Get an overview of the filing's structure: size, TOC link count, anchor count, Part boundaries, and difficulty estimate. Call this FIRST to orient yourself.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_toc_links",
            "description": "Get all table-of-contents entries with their anchor IDs and pre-classified item names. Most items are directly identifiable from TOC text.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_item_candidates",
            "description": "Get all anchor candidates for a specific item, scored by confidence. Returns positions, confidence levels, source type, and nearby text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {
                        "type": "string",
                        "description": "The item to search for (e.g., 'item1', 'item7a', 'signatures')",
                    },
                },
                "required": ["item_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_text_at",
            "description": "Read cleaned text at a specific position in the HTML. Use this to verify what's at a boundary position.",
            "parameters": {
                "type": "object",
                "properties": {
                    "position": {
                        "type": "integer",
                        "description": "Character position in the HTML to read from",
                    },
                    "length": {
                        "type": "integer",
                        "description": "Number of characters to read (default 500, max 2000)",
                    },
                },
                "required": ["position"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_part_boundaries",
            "description": "Get detected Part I/II/III/IV header positions. These provide hard constraints on where items can appear.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "classify_text",
            "description": "Run deterministic regex classification on text. Returns Tier 1 (Item X pattern) and Tier 2 (keyword) matches. Use this to verify your own reasoning.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to classify (will be normalized automatically)",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_incorporation",
            "description": "Check if a specific item or Part III items are incorporated by reference (common for items 10-14).",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {
                        "type": "string",
                        "description": "Item to check (e.g., 'item10'). Use 'part3' to check all Part III items.",
                    },
                },
                "required": ["item_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assign_item",
            "description": "Assign an item to an anchor position. Use char_position=-1 for items incorporated by reference (null assignment).",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {
                        "type": "string",
                        "description": "Item name (e.g., 'item1', 'item7a', 'signatures')",
                    },
                    "anchor_id": {
                        "type": "string",
                        "description": "The anchor ID to assign (from get_item_candidates)",
                    },
                    "char_position": {
                        "type": "integer",
                        "description": "Character position in HTML. Use -1 for incorporated by reference.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why this assignment is correct",
                    },
                },
                "required": ["item_name", "anchor_id", "char_position", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unassign_item",
            "description": "Remove an item assignment (for self-correction).",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {
                        "type": "string",
                        "description": "Item name to unassign",
                    },
                },
                "required": ["item_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_assignments",
            "description": "Get all current item assignments sorted by position. Use this to review your work before validating.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_assignments",
            "description": "Validate all current assignments. Checks canonical order, monotonic positions, expected items, span sizes. Returns specific issues to fix.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_span_sizes",
            "description": "Check the size of each item's HTML span. Flags suspiciously large (>40% of doc) or small (<100 chars) spans.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_top_candidates",
            "description": "Get the TOP candidate for EVERY item in one call. Returns the highest-confidence candidate per item with position and text. Use this for efficient bulk discovery instead of calling get_item_candidates 22 times.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "batch_assign",
            "description": "Assign multiple items at once. Much faster than calling assign_item 22 times. Each entry needs item_name, anchor_id, char_position, and reasoning.",
            "parameters": {
                "type": "object",
                "properties": {
                    "assignments": {
                        "type": "array",
                        "description": "List of assignments, each with item_name, anchor_id, char_position, reasoning",
                        "items": {
                            "type": "object",
                            "properties": {
                                "item_name": {"type": "string"},
                                "anchor_id": {"type": "string"},
                                "char_position": {"type": "integer"},
                                "reasoning": {"type": "string"},
                            },
                            "required": ["item_name", "anchor_id", "char_position", "reasoning"],
                        },
                    },
                },
                "required": ["assignments"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "refine_boundary",
            "description": "Search for anchors near an assigned item's position (±radius) to find a more precise boundary. Use when an item's span looks wrong or F1 might be low due to a slightly off boundary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {
                        "type": "string",
                        "description": "The assigned item to refine",
                    },
                    "search_radius": {
                        "type": "integer",
                        "description": "Characters to search in each direction (default 5000, max 20000)",
                    },
                },
                "required": ["item_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scan_for_heading",
            "description": "Search for an 'Item X' heading between two document positions. Use this when adjacent items (like item6/item7/item7a or item9a/item9b) might have swapped boundaries or when a short item is missing between two assigned items.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_name": {
                        "type": "string",
                        "description": "The item heading to search for (e.g., 'item7a', 'item9b')",
                    },
                    "start_position": {
                        "type": "integer",
                        "description": "Start of search range (char position)",
                    },
                    "end_position": {
                        "type": "integer",
                        "description": "End of search range (char position)",
                    },
                },
                "required": ["item_name", "start_position", "end_position"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize",
            "description": "Finalize the extraction and produce output. Only succeeds if validate_assignments passes. Call this when you're done.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def _strip_tags(html_fragment: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r'<[^>]+>', ' ', html_fragment)
    text = html_module.unescape(text)
    return re.sub(r'\s+', ' ', text).strip()


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Manages tool execution and state for the agent loop."""

    def __init__(self, index: StructuralIndex):
        self._index = index
        self._state = AssignmentState()
        self._turn = 0
        self._finalized = False

    def set_turn(self, turn: int) -> None:
        self._turn = turn

    def openai_schemas(self) -> list[dict]:
        return TOOL_SCHEMAS

    def get_state(self) -> AssignmentState:
        return self._state

    def is_finalized(self) -> bool:
        return self._finalized

    def execute(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool and return the result as a JSON string."""
        handler = getattr(self, f"_tool_{tool_name}", None)
        if handler is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        try:
            result = handler(**arguments)
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def pre_populate_dp(self) -> dict | None:
        """Pre-populate assignments from the DP solver's proposal.

        Returns the proposal summary, or None if no DP result available.
        """
        if not self._index.dp_assignments:
            return None

        assigned = []
        for dp in self._index.dp_assignments:
            self._state.assign_item(
                dp.item_name, dp.anchor_id, dp.char_position,
                reasoning="DP solver proposal",
            )
            assigned.append({
                "item_name": dp.item_name,
                "anchor_id": dp.anchor_id,
                "char_position": dp.char_position,
                "relative_position": (
                    f"{dp.char_position / self._index.html_length:.1%}"
                    if self._index.html_length > 0 else "0%"
                ),
            })
        return {"dp_assignments": assigned, "count": len(assigned)}

    # ----- Discovery Tools -----

    def _tool_get_filing_overview(self) -> dict:
        idx = self._index
        part_info = [
            {"part": pb.part_number, "position": pb.char_position,
             "relative_position": f"{pb.relative_position:.1%}"}
            for pb in idx.part_boundaries
        ]
        toc_classified = sum(
            1 for tl in idx.toc_links if tl.classified_item is not None
        )
        total_candidates = sum(
            len(cands) for cands in idx.item_candidates.values()
        )
        difficulty = "easy"
        if total_candidates > 100:
            difficulty = "hard"
        elif total_candidates > 50:
            difficulty = "medium"

        return {
            "accession": idx.accession,
            "html_length": idx.html_length,
            "toc_link_count": len(idx.toc_links),
            "toc_classified_count": toc_classified,
            "anchor_count": len(idx.anchor_positions),
            "total_item_candidates": total_candidates,
            "items_with_candidates": list(idx.item_candidates.keys()),
            "part_boundaries": part_info,
            "part3_incorporated": idx.part3_incorporated,
            "difficulty_estimate": difficulty,
        }

    def _tool_get_toc_links(self) -> dict:
        links = []
        for tl in self._index.toc_links:
            links.append({
                "anchor_id": tl.anchor_id,
                "link_text": tl.link_text[:150],
                "classified_item": tl.classified_item,
            })
        return {"toc_links": links, "count": len(links)}

    def _tool_get_item_candidates(self, item_name: str) -> dict:
        candidates = self._index.item_candidates.get(item_name, [])
        result = []
        for c in candidates:
            result.append({
                "anchor_id": c.anchor_id,
                "char_position": c.char_position,
                "relative_position": f"{c.relative_position:.1%}",
                "confidence": c.confidence,
                "source": c.source,
                "nearby_text": c.nearby_text[:200],
                "tag_name": c.tag_name,
            })
        return {
            "item_name": item_name,
            "candidates": result,
            "count": len(result),
        }

    def _tool_read_text_at(self, position: int, length: int = 500) -> dict:
        length = min(length, 2000)
        position = max(0, min(position, self._index.html_length - 1))
        raw = self._index.html[position:position + length]
        text = _strip_tags(raw)
        return {
            "position": position,
            "length": length,
            "text": text[:500],
            "relative_position": f"{position / self._index.html_length:.1%}",
        }

    def _tool_get_part_boundaries(self) -> dict:
        boundaries = [
            {"part": pb.part_number, "char_position": pb.char_position,
             "relative_position": f"{pb.relative_position:.1%}"}
            for pb in self._index.part_boundaries
        ]
        return {"part_boundaries": boundaries, "count": len(boundaries)}

    # ----- Classification Tools -----

    def _tool_classify_text(self, text: str) -> dict:
        import sys
        from pathlib import Path
        _project_root = str(Path(__file__).resolve().parent.parent.parent)
        if _project_root not in sys.path:
            sys.path.insert(0, _project_root)
        from extract import _classify_tier1, _classify_tier2, normalize_text, _classify_anchor_id

        normed = normalize_text(text)
        tier1 = _classify_tier1(normed)
        tier2 = _classify_tier2(normed)
        return {
            "text_preview": text[:100],
            "tier1_match": tier1,
            "tier2_match": tier2,
        }

    def _tool_detect_incorporation(self, item_name: str) -> dict:
        if item_name == "part3" or item_name in {
            "item10", "item11", "item12", "item13", "item14"
        }:
            return {
                "item_name": item_name,
                "is_incorporated": self._index.part3_incorporated,
                "evidence": "Part III incorporation detected in filing"
                if self._index.part3_incorporated
                else "No incorporation language found",
            }
        return {
            "item_name": item_name,
            "is_incorporated": False,
            "evidence": "Only Part III items (10-14) are typically incorporated",
        }

    # ----- Assignment Tools -----

    def _tool_assign_item(
        self, item_name: str, anchor_id: str, char_position: int, reasoning: str
    ) -> dict:
        pos = char_position if char_position >= 0 else None
        self._state.assign_item(item_name, anchor_id, pos, reasoning)
        return {
            "success": True,
            "item_name": item_name,
            "char_position": pos,
            "current_assignment_count": len(self._state.get_assignments()),
        }

    def _tool_unassign_item(self, item_name: str) -> dict:
        self._state.unassign_item(item_name)
        return {
            "success": True,
            "item_name": item_name,
            "current_assignment_count": len(self._state.get_assignments()),
        }

    def _tool_get_current_assignments(self) -> dict:
        assignments = self._state.get_assignments()
        result = []
        for a in assignments:
            result.append({
                "item_name": a.item_name,
                "anchor_id": a.anchor_id,
                "char_position": a.char_position,
                "relative_position": (
                    f"{a.char_position / self._index.html_length:.1%}"
                    if a.char_position is not None else "null"
                ),
                "reasoning": a.reasoning,
            })
        return {"assignments": result, "count": len(result)}

    # ----- Validation Tools -----

    def _tool_validate_assignments(self) -> dict:
        assignments = self._state.get_assignments()
        return validate_assignments(assignments, self._index.html_length)

    def _tool_check_span_sizes(self) -> dict:
        assignments = self._state.get_assignments()
        spans = check_span_sizes(assignments, self._index.html_length)
        return {"spans": spans}

    # ----- Bulk Tools -----

    def _tool_get_all_top_candidates(self) -> dict:
        """Return the top 3 candidates per item — one call instead of 22."""
        results = {}
        for item_name, candidates in sorted(self._index.item_candidates.items()):
            if not candidates:
                continue
            top3 = sorted(candidates, key=lambda c: -c.confidence)[:3]
            best = top3[0]
            entry = {
                "anchor_id": best.anchor_id,
                "char_position": best.char_position,
                "relative_position": f"{best.relative_position:.1%}",
                "confidence": best.confidence,
                "source": best.source,
                "nearby_text": best.nearby_text[:120],
            }
            if len(top3) > 1:
                entry["alternatives"] = [
                    {
                        "anchor_id": c.anchor_id,
                        "char_position": c.char_position,
                        "relative_position": f"{c.relative_position:.1%}",
                        "confidence": c.confidence,
                        "source": c.source,
                        "nearby_text": c.nearby_text[:80],
                    }
                    for c in top3[1:]
                ]
            results[item_name] = entry
        return {"items": results, "count": len(results)}

    def _tool_batch_assign(self, assignments: list[dict]) -> dict:
        """Assign multiple items at once."""
        assigned = []
        errors = []
        for entry in assignments:
            try:
                item_name = entry["item_name"]
                anchor_id = entry.get("anchor_id", "")
                char_position = entry.get("char_position", -1)
                reasoning = entry.get("reasoning", "")
                pos = char_position if char_position >= 0 else None
                self._state.assign_item(item_name, anchor_id, pos, reasoning)
                assigned.append(item_name)
            except Exception as e:
                errors.append(f"{entry.get('item_name', '?')}: {e}")
        return {
            "assigned": assigned,
            "assigned_count": len(assigned),
            "errors": errors,
            "total_assignments": len(self._state.get_assignments()),
        }

    # ----- Refinement Tools -----

    def _tool_refine_boundary(self, item_name: str, search_radius: int = 5000) -> dict:
        """Search around current assignment position for a better anchor.

        Looks for all anchors within ±search_radius of the current position
        and returns them with context text, so you can pick the most precise one.
        """
        search_radius = min(search_radius, 20000)
        assignment = self._state.get_assignment(item_name)
        if assignment is None:
            return {"error": f"{item_name} is not currently assigned"}
        if assignment.char_position is None:
            return {"error": f"{item_name} is incorporated by reference (no position)"}

        center = assignment.char_position
        lo = max(0, center - search_radius)
        hi = min(self._index.html_length, center + search_radius)

        # Find all anchors in the window
        nearby_anchors = []
        for anchor_id, (offset, tag_name, _attr) in self._index.anchor_positions.items():
            if lo <= offset <= hi:
                text = _strip_tags(self._index.html[offset:offset + 300])[:150]
                nearby_anchors.append({
                    "anchor_id": anchor_id,
                    "char_position": offset,
                    "distance_from_current": offset - center,
                    "relative_position": f"{offset / self._index.html_length:.1%}",
                    "tag_name": tag_name,
                    "text_preview": text,
                })

        nearby_anchors.sort(key=lambda a: a["char_position"])

        return {
            "item_name": item_name,
            "current_position": center,
            "search_range": [lo, hi],
            "nearby_anchors": nearby_anchors,
            "count": len(nearby_anchors),
        }

    # ----- Heading Scan Tool -----

    # Patterns to find item headings in HTML (must be in heading/bold tags)
    _HEADING_SCAN_PATTERNS = {
        "item1": re.compile(r'item\s*1\b(?!\s*[0-9a])', re.I),
        "item1a": re.compile(r'item\s*1\s*[\.\-]?\s*a\b', re.I),
        "item1b": re.compile(r'item\s*1\s*[\.\-]?\s*b\b', re.I),
        "item2": re.compile(r'item\s*2\b', re.I),
        "item3": re.compile(r'item\s*3\b', re.I),
        "item4": re.compile(r'item\s*4\b', re.I),
        "item5": re.compile(r'item\s*5\b', re.I),
        "item6": re.compile(r'item\s*6\b', re.I),
        "item7": re.compile(r'item\s*7\b(?!\s*[0-9a])', re.I),
        "item7a": re.compile(r'item\s*7\s*[\.\-\u2013\u2014]?\s*a\b', re.I),
        "item8": re.compile(r'item\s*8\b', re.I),
        "item9": re.compile(r'item\s*9\b(?!\s*[0-9a])', re.I),
        "item9a": re.compile(r'item\s*9\s*[\.\-]?\s*a\b', re.I),
        "item9b": re.compile(r'item\s*9\s*[\.\-]?\s*b\b', re.I),
        "item9c": re.compile(r'item\s*9\s*[\.\-]?\s*c\b', re.I),
        "item10": re.compile(r'item\s*10\b', re.I),
        "item11": re.compile(r'item\s*11\b', re.I),
        "item12": re.compile(r'item\s*12\b', re.I),
        "item13": re.compile(r'item\s*13\b', re.I),
        "item14": re.compile(r'item\s*14\b', re.I),
        "item15": re.compile(r'item\s*15\b', re.I),
        "item16": re.compile(r'item\s*16\b', re.I),
        "signatures": re.compile(r'\bsignatures?\s*$', re.I),
    }

    def _tool_scan_for_heading(
        self, item_name: str, start_position: int, end_position: int
    ) -> dict:
        """Search for an item heading between two positions in the HTML."""
        pattern = self._HEADING_SCAN_PATTERNS.get(item_name)
        if pattern is None:
            return {"error": f"Unknown item: {item_name}"}

        start_position = max(0, start_position)
        end_position = min(self._index.html_length, end_position)
        if end_position <= start_position:
            return {"found": False, "item_name": item_name, "matches": []}

        search_html = self._index.html[start_position:end_position]
        search_text = _strip_tags(search_html)

        matches = []
        for m in pattern.finditer(search_text):
            # Map text position back to approximate HTML position
            text_pos = m.start()
            text_len = len(search_text)
            html_len = len(search_html)
            approx_html_pos = start_position + int(text_pos / max(text_len, 1) * html_len)

            # Get context around the match
            ctx_start = max(0, text_pos - 50)
            ctx_end = min(len(search_text), text_pos + 150)
            context = search_text[ctx_start:ctx_end].strip()

            # Find the nearest anchor to this position
            nearest_anchor = None
            nearest_dist = float("inf")
            for aid, (offset, tag, _) in self._index.anchor_positions.items():
                dist = abs(offset - approx_html_pos)
                if dist < nearest_dist and dist < 2000:
                    nearest_dist = dist
                    nearest_anchor = {"anchor_id": aid, "char_position": offset, "distance": dist}

            matches.append({
                "approx_position": approx_html_pos,
                "relative_position": f"{approx_html_pos / self._index.html_length:.1%}",
                "context": context,
                "nearest_anchor": nearest_anchor,
            })

        return {
            "found": len(matches) > 0,
            "item_name": item_name,
            "search_range": [start_position, end_position],
            "matches": matches[:5],  # Limit to 5 matches
        }

    # ----- Output Tools -----

    def _tool_finalize(self) -> dict:
        assignments = self._state.get_assignments()
        validation = validate_assignments(assignments, self._index.html_length)
        if not validation["is_valid"]:
            return {
                "success": False,
                "error": "Validation failed. Fix issues before finalizing.",
                "issues": validation["issues"],
            }
        self._finalized = True
        return {
            "success": True,
            "item_count": len(assignments),
            "validation": validation,
        }
