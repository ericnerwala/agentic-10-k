"""Run agentic pipeline on 100 filings with parallel processing."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.agent.loop import AgentLoop
from pipeline.agent.index import build_structural_index
from pipeline.agent.runner import slice_html_from_assignments
from evaluate import evaluate_pair

MODEL_ID = "deepseek/deepseek-v3.2"
MAX_TURNS = 30
CONCURRENCY = 5
DATA_DIR = Path("data")
OUTPUT_DIR = Path("test50_v5")
OUTPUT_DIR.mkdir(exist_ok=True)


def find_filings(n=100):
    filings = []
    for gt_name in ["ground_truth_1", "ground_truth_2", "ground_truth_3"]:
        gt_dir = DATA_DIR / gt_name
        if not gt_dir.exists():
            continue
        for gt_file in sorted(gt_dir.glob("*.json")):
            acc = gt_file.stem
            for folder in ["folder_1", "folder_2", "folder_3"]:
                src = DATA_DIR / folder / f"{acc}.txt"
                if src.exists():
                    try:
                        with open(gt_file) as f:
                            gt = json.load(f)
                        if gt:
                            filings.append((acc, str(src), str(gt_file)))
                    except Exception:
                        pass
                    break
            if len(filings) >= n:
                break
        if len(filings) >= n:
            break
    return filings[:n]


async def process_filing(sem, i, total, acc, src, gt_path, api_key, results_lock, all_results):
    """Process a single filing under a semaphore."""
    async with sem:
        print(f"[{i+1:3d}/{total}] {acc} ...", flush=True)

        try:
            idx = build_structural_index(src, acc)
        except Exception as e:
            print(f"[{i+1:3d}/{total}] {acc} INDEX_ERR: {e}")
            entry = {
                "accession": acc, "error": f"index: {e}",
                "f1": 0, "doc_retrieved": False,
            }
            async with results_lock:
                all_results[i] = entry
            return

        n_cands = sum(len(c) for c in idx.item_candidates.values())

        agent = AgentLoop(
            model_id=MODEL_ID, api_key=api_key, max_turns=MAX_TURNS
        )
        try:
            result = await agent.run(idx, acc)
        except Exception as e:
            print(f"[{i+1:3d}/{total}] {acc} AGENT_ERR: {e}")
            entry = {
                "accession": acc, "error": f"agent: {e}",
                "f1": 0, "doc_retrieved": False,
            }
            async with results_lock:
                all_results[i] = entry
            return

        tok = result.total_prompt_tokens + result.total_completion_tokens
        preds = slice_html_from_assignments(idx.html, result.assignments, acc)

        with open(gt_path) as f:
            gt = json.load(f)

        ev = evaluate_pair(preds, gt, acc)
        dr = ev.get("doc_retrieved", False)
        f1 = ev["mean_char_f1"]

        failed_items = {}
        for item_name, metrics in sorted(ev.get("per_item", {}).items()):
            if metrics["char_f1"] < 0.90:
                failed_items[item_name] = {
                    "f1": round(metrics["char_f1"], 4),
                    "pred_len": metrics["pred_len"],
                    "truth_len": metrics["truth_len"],
                }

        status = "YES" if dr else "NO"
        fin = "FIN" if result.finalized else "BEST"
        err = f" ERR:{result.error[:40]}" if result.error else ""
        fail_str = ""
        if failed_items:
            fail_str = " FAILS: " + ", ".join(
                f"{k}({v['f1']:.2f})" for k, v in failed_items.items()
            )
        print(f"[{i+1:3d}/{total}] {acc} F1={f1:.1%} DR={status} {result.turns_used}t {tok}tok {fin}{err}{fail_str}")

        entry = {
            "accession": acc,
            "model": MODEL_ID,
            "f1": round(f1, 4),
            "doc_retrieved": dr,
            "extraction_rate": round(ev["extraction_rate"], 4),
            "turns": result.turns_used,
            "tokens": tok,
            "latency_ms": result.total_latency_ms,
            "finalized": result.finalized,
            "error": result.error,
            "pred_items": len(preds),
            "gt_items": ev["truth_item_count"],
            "false_positives": ev.get("false_positives", []),
            "failed_items": failed_items,
            "candidates": n_cands,
        }
        async with results_lock:
            all_results[i] = entry
            # Save incrementally (ordered by index, skip None placeholders)
            ordered = [r for r in all_results if r is not None]
            with open(OUTPUT_DIR / "results.json", "w") as f:
                json.dump(ordered, f, indent=2, default=str)


async def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Set OPENROUTER_API_KEY")
        sys.exit(1)

    filings = find_filings(50)
    total = len(filings)
    print(f"Model: {MODEL_ID}")
    print(f"Filings: {total}")
    print(f"Max turns: {MAX_TURNS}")
    print(f"Concurrency: {CONCURRENCY}\n")

    sem = asyncio.Semaphore(CONCURRENCY)
    results_lock = asyncio.Lock()
    all_results = [None] * total

    total_start = time.monotonic()

    tasks = [
        process_filing(sem, i, total, acc, src, gt_path, api_key, results_lock, all_results)
        for i, (acc, src, gt_path) in enumerate(filings)
    ]
    await asyncio.gather(*tasks)

    total_ms = int((time.monotonic() - total_start) * 1000)

    # Filter out None placeholders (shouldn't happen but be safe)
    final_results = [r for r in all_results if r is not None]

    # Final summary
    n = len(final_results)
    dr_count = sum(1 for r in final_results if r.get("doc_retrieved", False))
    avg_f1 = sum(r.get("f1", 0) for r in final_results) / max(n, 1)
    errors = sum(1 for r in final_results if r.get("error"))

    report_lines = [
        f"# Test100 Benchmark: {MODEL_ID}",
        f"",
        f"Date: {time.strftime('%Y-%m-%d %H:%M')}",
        f"Model: {MODEL_ID}",
        f"Max turns: {MAX_TURNS}",
        f"Concurrency: {CONCURRENCY}",
        f"Total filings: {n}",
        f"Total time: {total_ms/1000:.0f}s",
        f"",
        f"## Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Doc Retrieval Rate | {dr_count}/{n} ({dr_count/max(n,1):.1%}) |",
        f"| Mean Char F1 | {avg_f1:.1%} |",
        f"| Errors | {errors} |",
        f"| Avg Turns | {sum(r.get('turns',0) for r in final_results)/max(n,1):.1f} |",
        f"| Avg Tokens | {sum(r.get('tokens',0) for r in final_results)/max(n,1):,.0f} |",
        f"",
        f"## Per-Filing Results",
        f"",
        f"| # | Accession | F1 | DocRet | Turns | Tokens | Failed Items |",
        f"|---|-----------|-----|--------|-------|--------|-------------|",
    ]

    for i, r in enumerate(final_results):
        fails = ", ".join(
            f"{k}({v['f1']:.2f})" for k, v in r.get("failed_items", {}).items()
        )
        if r.get("error") and not fails:
            fails = f"ERROR: {r['error'][:40]}"
        report_lines.append(
            f"| {i+1} | {r['accession']} | {r.get('f1',0):.1%} | "
            f"{'YES' if r.get('doc_retrieved') else 'NO'} | "
            f"{r.get('turns',0)} | {r.get('tokens',0):,} | {fails} |"
        )

    # Failed docs detail
    failed_docs = [r for r in final_results if not r.get("doc_retrieved", False)]
    if failed_docs:
        report_lines.extend([
            f"",
            f"## Failed Documents ({len(failed_docs)})",
            f"",
        ])
        for r in failed_docs:
            report_lines.append(f"### {r['accession']}")
            report_lines.append(f"- F1: {r.get('f1',0):.1%}")
            report_lines.append(f"- Turns: {r.get('turns',0)}, Tokens: {r.get('tokens',0):,}")
            if r.get("error"):
                report_lines.append(f"- Error: {r['error']}")
            if r.get("false_positives"):
                report_lines.append(f"- False positives: {r['false_positives']}")
            for item, m in r.get("failed_items", {}).items():
                report_lines.append(
                    f"- **{item}**: F1={m['f1']:.3f} (pred={m['pred_len']:,} gt={m['truth_len']:,})"
                )
            report_lines.append("")

    report = "\n".join(report_lines)
    with open(OUTPUT_DIR / "report.md", "w") as f:
        f.write(report)

    # Save final results
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  TEST100 RESULTS — {MODEL_ID}")
    print(f"{'='*60}")
    print(f"  Doc Retrieval: {dr_count}/{n} ({dr_count/max(n,1):.1%})")
    print(f"  Mean Char F1:  {avg_f1:.1%}")
    print(f"  Errors:        {errors}")
    print(f"  Total Time:    {total_ms/1000:.0f}s (vs ~{n*120}s sequential)")
    print(f"  Speedup:       ~{n*120*1000/max(total_ms,1):.1f}x")
    print(f"\n  Report: {OUTPUT_DIR}/report.md")
    print(f"  Results: {OUTPUT_DIR}/results.json")


asyncio.run(main())
