"""Resume v5 run — skip already-completed filings."""

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
CONCURRENCY = 8
DATA_DIR = Path("data")
OUTPUT_DIR = Path("test_all_v5")
OUTPUT_DIR.mkdir(exist_ok=True)


def find_filings(n=391):
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


async def process_filing(sem, i, total, acc, src, gt_path, api_key, all_results, results_lock):
    async with sem:
        try:
            idx = build_structural_index(src, acc)
        except Exception as e:
            print(f"[{i+1:3d}/{total}] {acc} INDEX_ERR: {e}")
            entry = {"accession": acc, "error": f"index: {e}", "f1": 0, "doc_retrieved": False}
            async with results_lock:
                all_results.append(entry)
            return

        agent = AgentLoop(model_id=MODEL_ID, api_key=api_key, max_turns=MAX_TURNS)
        try:
            result = await agent.run(idx, acc)
        except Exception as e:
            print(f"[{i+1:3d}/{total}] {acc} AGENT_ERR: {e}")
            entry = {"accession": acc, "error": f"agent: {e}", "f1": 0, "doc_retrieved": False}
            async with results_lock:
                all_results.append(entry)
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
            fail_str = " FAILS: " + ", ".join(f"{k}({v['f1']:.2f})" for k, v in failed_items.items())
        print(f"[{i+1:3d}/{total}] {acc} F1={f1:.1%} DR={status} {result.turns_used}t {tok}tok {fin}{err}{fail_str}", flush=True)

        entry = {
            "accession": acc, "model": MODEL_ID,
            "f1": round(f1, 4), "doc_retrieved": dr,
            "extraction_rate": round(ev["extraction_rate"], 4),
            "turns": result.turns_used, "tokens": tok,
            "latency_ms": result.total_latency_ms,
            "finalized": result.finalized, "error": result.error,
            "pred_items": len(preds), "gt_items": ev["truth_item_count"],
            "false_positives": ev.get("false_positives", []),
            "failed_items": failed_items,
            "candidates": sum(len(c) for c in idx.item_candidates.values()),
        }
        async with results_lock:
            all_results.append(entry)
            with open(OUTPUT_DIR / "results.json", "w") as f:
                json.dump(all_results, f, indent=2, default=str)


async def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Set OPENROUTER_API_KEY")
        sys.exit(1)

    # Load existing results
    existing = []
    results_path = OUTPUT_DIR / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            existing = json.load(f)
    done_accs = {r["accession"] for r in existing}
    print(f"Already completed: {len(done_accs)}")

    all_filings = find_filings(391)
    remaining = [(acc, src, gt) for acc, src, gt in all_filings if acc not in done_accs]
    total = len(remaining)
    print(f"Remaining: {total}")
    print(f"Concurrency: {CONCURRENCY}\n")

    sem = asyncio.Semaphore(CONCURRENCY)
    results_lock = asyncio.Lock()
    all_results = list(existing)  # Start with existing results

    total_start = time.monotonic()
    tasks = [
        process_filing(sem, i, total, acc, src, gt_path, api_key, all_results, results_lock)
        for i, (acc, src, gt_path) in enumerate(remaining)
    ]
    await asyncio.gather(*tasks)
    total_ms = int((time.monotonic() - total_start) * 1000)

    n = len(all_results)
    dr_count = sum(1 for r in all_results if r.get("doc_retrieved", False))
    avg_f1 = sum(r.get("f1", 0) for r in all_results) / max(n, 1)

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  V5 FULL RESULTS — {MODEL_ID}")
    print(f"{'='*60}")
    print(f"  Doc Retrieval: {dr_count}/{n} ({dr_count/max(n,1):.1%})")
    print(f"  Mean Char F1:  {avg_f1:.1%}")
    print(f"  Resume time:   {total_ms/1000:.0f}s")


asyncio.run(main())
