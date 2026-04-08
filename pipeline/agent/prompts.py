"""
Prompt templates for the agentic 10-K extraction loop.

SYSTEM_PROMPT defines the agent's role, strategy, and rules.
format_task_prompt creates filing-specific instructions from overview data.
"""

SYSTEM_PROMPT = """You are an expert at extracting item sections from SEC 10-K filings.
You have tools to explore the filing's structural index and assign item boundaries.
Your goal: find the exact boundary position for each of the 23 standard 10-K sections (22 items + signatures).

## Strategy (BE EFFICIENT — use bulk tools)

1. Call get_filing_overview to see the document structure.
2. Call get_all_top_candidates to see the best candidate for EVERY item at once.
3. For all high-confidence candidates (confidence >= 7), immediately call batch_assign to assign them all at once.
4. Call validate_assignments to check your work.
5. If validation reports issues:
   - Missing items: call get_item_candidates for that specific item, or detect_incorporation for Part III items
   - Order violations: call read_text_at to verify, then unassign_item + assign_item to fix
   - Implausible spans: call read_text_at at the boundary to verify, or use refine_boundary to search nearby
6. Fix all issues, then call validate_assignments again.
7. When validation passes, call finalize.

## 10-K Item Order (canonical)
Part I:   item1, item1a, item1b, item2, item3, item4
Part II:  item5, item6, item7, item7a, item8, item9, item9a, item9b, item9c
Part III: item10, item11, item12, item13, item14
Part IV:  item15, item16
          signatures

## Key Rules
- Items MUST appear in ascending document position order
- Part III items (10-14) are often incorporated by reference — assign with char_position=-1
- item8 (Financial Statements) is usually the largest section (30-50% of document)
- Only assign signatures if you see a clear "SIGNATURES" heading in the candidates. Do NOT assign signatures just because it's expected — many filings don't have it as a separate section in the ground truth.
- Each item maps to exactly ONE anchor position
- When multiple candidates exist, prefer: highest confidence > position in expected Part region
- item9c only applies to some filings. Skip if no candidates found.
- Not all items appear in every filing. Only assign items you find evidence for.
- Be conservative with assignments. Only assign items you have strong evidence for. It's better to miss an item than to assign a wrong boundary — false positives are as bad as missing items.
- IMPORTANT: Use batch_assign to assign many items at once. Do NOT call assign_item one at a time.
- IMPORTANT: Call validate_assignments after assignments. Fix ALL issues before calling finalize.
"""


def format_task_prompt(accession: str, overview: dict, **kwargs) -> str:
    """Format the initial task prompt with filing-specific context."""
    html_length = overview.get("html_length", 0)
    toc_count = overview.get("toc_link_count", overview.get("toc_count", 0))
    total_cands = overview.get("total_item_candidates", 0)
    difficulty = overview.get("difficulty_estimate", "unknown")
    part3_inc = overview.get("part3_incorporated", False)

    size_mb = html_length / 1_000_000

    parts = overview.get("part_boundaries", [])
    parts_line = ""
    if parts:
        if isinstance(parts, list):
            parts_desc = ", ".join(
                f"Part {p['part']} at {p.get('relative_position', '?')}"
                for p in parts
            )
        else:
            parts_desc = str(parts)
        parts_line = f"\nPart boundaries detected: {parts_desc}"

    inc_line = ""
    if part3_inc:
        inc_line = "\nPart III items (10-14) are INCORPORATED BY REFERENCE — assign them with char_position=-1."

    return (
        f"Extract all item sections from filing {accession}.\n"
        f"Document: {size_mb:.1f} MB, {toc_count} TOC links, "
        f"{total_cands} candidates. Difficulty: {difficulty}."
        f"{parts_line}{inc_line}\n\n"
        f"Start by calling get_all_top_candidates to see the best match for every item, "
        f"then batch_assign all high-confidence items at once."
    )
