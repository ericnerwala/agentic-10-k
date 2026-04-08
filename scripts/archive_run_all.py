"""
Batch Processor
===============
Runs extract.py on all 3 folder pairs and optionally evaluates against ground truth.

Usage:
  python run_all.py [--eval] [--set 1|2|3] [--workers N]

  --eval         Run evaluate.py after extraction
  --set 1|2|3    Only process a specific set (default: all)
  --workers N    Parallel workers (default: 4)
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from extract import process_file
from evaluate import run_evaluation

BASE_DIR = Path(__file__).parent / "data"

SETS = [
    {
        "id": 1,
        "input_dir": BASE_DIR / "folder_1" / "folder_1",
        "truth_dir": BASE_DIR / "ground_truth_1" / "ground_truth_1",
        "pred_dir":  BASE_DIR / "predictions_1",
    },
    {
        "id": 2,
        "input_dir": BASE_DIR / "folder_2" / "folder_2",
        "truth_dir": BASE_DIR / "ground_truth_2" / "ground_truth_2",
        "pred_dir":  BASE_DIR / "predictions_2",
    },
    {
        "id": 3,
        "input_dir": BASE_DIR / "folder_3" / "folder_3",
        "truth_dir": BASE_DIR / "ground_truth_3" / "ground_truth_3",
        "pred_dir":  BASE_DIR / "predictions_3",
    },
]


def process_one(args):
    txt_path, out_path = args
    try:
        result = process_file(str(txt_path))
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False)
        return str(txt_path.name), len(result), None
    except Exception as e:
        return str(txt_path.name), 0, str(e)


def run_set(s: dict, workers: int, do_eval: bool):
    input_dir = s["input_dir"]
    pred_dir = s["pred_dir"]
    truth_dir = s["truth_dir"]
    set_id = s["id"]

    pred_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(input_dir.glob("*.txt"))
    print(f"\n{'='*60}")
    print(f"Set {set_id}: {len(txt_files)} files in {input_dir}")
    print(f"          predictions → {pred_dir}")

    tasks = []
    for txt_path in txt_files:
        out_path = pred_dir / f"{txt_path.stem}.json"
        tasks.append((txt_path, out_path))

    errors = []
    start = time.time()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_one, t): t for t in tasks}
        done = 0
        for future in as_completed(futures):
            name, n_items, err = future.result()
            done += 1
            if err:
                errors.append((name, err))
                print(f"  ERROR {name}: {err}")
            elif done % 20 == 0 or done == len(tasks):
                elapsed = time.time() - start
                print(f"  [{done}/{len(tasks)}] elapsed={elapsed:.0f}s  last={name} ({n_items} items)")

    elapsed = time.time() - start
    print(f"  Extraction done in {elapsed:.1f}s.  Errors: {len(errors)}")

    if do_eval and truth_dir.exists():
        print(f"\n  Running evaluation for set {set_id}...")
        run_evaluation(str(pred_dir), str(truth_dir), verbose=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true', help='Run evaluation after extraction')
    parser.add_argument('--set', type=int, choices=[1, 2, 3], help='Only process set 1, 2, or 3')
    parser.add_argument('--workers', type=int, default=4, help='Parallel workers')
    args = parser.parse_args()

    sets_to_run = [s for s in SETS if args.set is None or s["id"] == args.set]

    for s in sets_to_run:
        run_set(s, workers=args.workers, do_eval=args.eval)

    print("\nDone.")


if __name__ == '__main__':
    main()
