"""
Batch runner: run the agentic pipeline on multiple filings and evaluate.
"""

import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from .loop import AgentLoop, AgentResult
from .index import build_structural_index
from ..config import OPENROUTER_API_KEY_ENV, FILING_FOLDERS, GT_DIRS, ITEM_SEQ_INDEX
from evaluate import evaluate_pair, strip_html


# ---------------------------------------------------------------------------
# Filing discovery (self-contained, no dependency on old select_filings.py)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FilingInfo:
    accession: str
    source_path: str
    gt_path: str
    html_size: int
    gt_item_count: int
    difficulty: str


def _find_source_file(accession: str) -> str | None:
    """Find the .txt source file across all folders."""
    for folder_path in FILING_FOLDERS.values():
        candidate = folder_path / f"{accession}.txt"
        if candidate.exists():
            return str(candidate)
    return None


def find_test_filings() -> list[FilingInfo]:
    """Find all filings that have both source .txt and non-empty GT."""
    from extract import extract_10k_text

    filings = []
    for gt_name, gt_dir in GT_DIRS.items():
        if not gt_dir.exists():
            continue
        for gt_file in sorted(gt_dir.glob("*.json")):
            accession = gt_file.stem
            source_path = _find_source_file(accession)
            if source_path is None:
                continue

            try:
                with open(gt_file, "r", encoding="utf-8") as f:
                    gt = json.load(f)
                if not gt:
                    continue
            except (json.JSONDecodeError, OSError):
                continue

            try:
                html = extract_10k_text(source_path)
                html_size = len(html)
            except Exception:
                continue

            # Simple difficulty estimate
            bold_count = len(re.findall(
                r'font-weight:\s*(?:bold|700)[^>]*>\s*item\s*\d',
                html[:html_size // 3], re.I
            ))
            difficulty = "easy" if bold_count >= 10 else "medium" if bold_count >= 3 else "hard"

            filings.append(FilingInfo(
                accession=accession,
                source_path=source_path,
                gt_path=str(gt_file),
                html_size=html_size,
                gt_item_count=len(gt),
                difficulty=difficulty,
            ))

    return filings


# ---------------------------------------------------------------------------
# HTML slicer (self-contained, no dependency on old slicer.py)
# ---------------------------------------------------------------------------

def slice_html_from_assignments(
    html: str,
    assignments: list,
    accession: str,
) -> dict[str, str]:
    """Slice HTML at assignment positions to produce GT-compatible output."""
    # Filter to valid assignments with positions, sort by position
    valid = [
        a for a in assignments
        if a.char_position is not None and 0 <= a.char_position < len(html)
    ]
    valid.sort(key=lambda a: a.char_position)

    # Dedup by item name (keep first)
    seen = set()
    deduped = []
    for a in valid:
        if a.item_name not in seen:
            seen.add(a.item_name)
            deduped.append(a)

    # Enforce ascending positions
    ordered = []
    last_pos = -1
    for a in deduped:
        if a.char_position > last_pos:
            ordered.append(a)
            last_pos = a.char_position

    if not ordered:
        return {}

    # Slice between consecutive boundaries
    result = {}
    for i, a in enumerate(ordered):
        start = a.char_position
        end = ordered[i + 1].char_position if i + 1 < len(ordered) else len(html)
        key = f"{accession}#{a.item_name}"
        result[key] = html[start:end]

    # Add null items (incorporated by reference) as empty strings
    for a in assignments:
        if a.char_position is None:
            key = f"{accession}#{a.item_name}"
            if key not in result:
                result[key] = ""

    return result


# ---------------------------------------------------------------------------
# Single filing runner
# ---------------------------------------------------------------------------

async def run_single_filing(
    filing: FilingInfo,
    model_id: str,
    api_key: str,
    use_native_tools: bool = True,
    max_turns: int = 30,
) -> dict:
    """Run the agent on a single filing and evaluate against GT."""
    print(f"\n  Processing: {filing.accession} ({filing.difficulty})")

    # Build structural index
    t0 = time.monotonic()
    index = build_structural_index(filing.source_path, filing.accession)
    index_time = int((time.monotonic() - t0) * 1000)
    n_candidates = sum(len(c) for c in index.item_candidates.values())
    print(f"    Index: {index.html_length:,} chars, {len(index.toc_links)} TOC links, "
          f"{n_candidates} candidates ({index_time}ms)")

    # Run agent
    agent = AgentLoop(
        model_id=model_id,
        api_key=api_key,
        max_turns=max_turns,
        use_native_tools=use_native_tools,
    )
    result = await agent.run(index, filing.accession)
    total_tok = result.total_prompt_tokens + result.total_completion_tokens
    print(f"    Agent: {result.turns_used} turns, {total_tok} tokens, "
          f"{result.total_latency_ms}ms"
          f"{' FINALIZED' if result.finalized else ' BEST-EFFORT'}"
          f"{f' ERROR: {result.error}' if result.error else ''}")

    # Slice HTML from assignments
    predictions = slice_html_from_assignments(
        index.html, result.assignments, filing.accession
    )
    print(f"    Sliced: {len(predictions)} items")

    # Evaluate against GT
    with open(filing.gt_path, "r", encoding="utf-8") as f:
        gt_dict = json.load(f)

    if gt_dict:
        eval_result = evaluate_pair(predictions, gt_dict, filing.accession)
        doc_ret = eval_result.get("doc_retrieved", False)
        mean_f1 = eval_result["mean_char_f1"]
        ext_rate = eval_result["extraction_rate"]
        false_pos = eval_result.get("false_positives", [])
        per_item = eval_result.get("per_item", {})

        print(f"    Eval: F1={mean_f1:.1%} DocRet={'YES' if doc_ret else 'NO'} "
              f"ExtRate={ext_rate:.1%} FP={len(false_pos)}")

        # Show per-item failures
        fails = []
        for item_name, metrics in sorted(per_item.items()):
            if metrics["char_f1"] < 0.90:
                fails.append(
                    f"      {item_name:15s} F1={metrics['char_f1']:.3f} "
                    f"(pred={metrics['pred_len']:>9,} gt={metrics['truth_len']:>9,})"
                )
        if fails:
            print(f"    Failures ({len(fails)}):")
            for f_line in fails:
                print(f_line)
    else:
        eval_result = {"mean_char_f1": 0, "doc_retrieved": False, "extraction_rate": 0}

    return {
        "accession": filing.accession,
        "difficulty": filing.difficulty,
        "turns": result.turns_used,
        "tokens": total_tok,
        "latency_ms": result.total_latency_ms,
        "finalized": result.finalized,
        "error": result.error,
        "predictions_count": len(predictions),
        "eval": eval_result,
    }


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

async def run_all_filings(
    model_id: str = "qwen/qwen3.6-plus:free",
    use_native_tools: bool = True,
    max_turns: int = 30,
) -> list[dict]:
    """Run the agent on all available GT filings and print report."""
    api_key = os.environ.get(OPENROUTER_API_KEY_ENV)
    if not api_key:
        print(f"Set {OPENROUTER_API_KEY_ENV} environment variable")
        sys.exit(1)

    print(f"Model: {model_id}")
    print(f"Native tools: {use_native_tools}")
    print(f"Max turns: {max_turns}")

    filings = find_test_filings()
    print(f"\nFound {len(filings)} test filings:")
    for f in filings:
        print(f"  {f.accession} ({f.difficulty}) -- "
              f"{f.html_size:,} bytes, {f.gt_item_count} GT items")

    results = []
    total_start = time.monotonic()

    for filing in filings:
        result = await run_single_filing(
            filing, model_id, api_key,
            use_native_tools=use_native_tools,
            max_turns=max_turns,
        )
        results.append(result)

    total_ms = int((time.monotonic() - total_start) * 1000)

    # Summary report
    print("\n" + "=" * 60)
    print("  AGENTIC PIPELINE -- RESULTS")
    print("=" * 60)

    n = max(len(results), 1)
    doc_retrieved = sum(1 for r in results if r["eval"].get("doc_retrieved", False))
    avg_f1 = sum(r["eval"]["mean_char_f1"] for r in results) / n
    avg_turns = sum(r["turns"] for r in results) / n
    avg_tokens = sum(r["tokens"] for r in results) / n
    total_tokens = sum(r["tokens"] for r in results)

    print(f"\n  Doc Retrieval Rate: {doc_retrieved}/{len(results)} "
          f"({doc_retrieved / n:.0%})")
    print(f"  Mean Char F1:      {avg_f1:.1%}")
    print(f"  Avg Turns/Filing:  {avg_turns:.1f}")
    print(f"  Avg Tokens/Filing: {avg_tokens:,.0f}")
    print(f"  Total Tokens:      {total_tokens:,}")
    print(f"  Total Time:        {total_ms / 1000:.1f}s")

    header = f"\n  {'Accession':<30} {'Diff':<8} {'F1':>6} {'DocRet':>7} {'Turns':>6} {'Tokens':>8}"
    print(header)
    print(f"  {'-' * 68}")
    for r in results:
        f1 = r["eval"]["mean_char_f1"]
        dr = "YES" if r["eval"].get("doc_retrieved", False) else "NO"
        print(f"  {r['accession']:<30} {r['difficulty']:<8} "
              f"{f1:>5.1%} {dr:>7} {r['turns']:>6} {r['tokens']:>8,}")

    print()
    return results


def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Agentic 10-K Extraction")
    parser.add_argument("--model", default="qwen/qwen3.6-plus:free",
                        help="Model ID on OpenRouter")
    parser.add_argument("--no-native-tools", action="store_true",
                        help="Use JSON-mode fallback instead of native function calling")
    parser.add_argument("--max-turns", type=int, default=30,
                        help="Max agent turns per filing")
    args = parser.parse_args()

    asyncio.run(run_all_filings(
        model_id=args.model,
        use_native_tools=not args.no_native_tools,
        max_turns=args.max_turns,
    ))


if __name__ == "__main__":
    main()
