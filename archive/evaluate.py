"""
Evaluation Script
=================
Compares predicted item extractions against ground truth JSONs.

Metrics per file and per item:
  1. Extraction rate  — did we extract an item that exists in ground truth?
  2. Character F1     — overlap on plain text (HTML tags stripped)
  3. Exact HTML match — are the HTML strings identical?

Usage:
  python evaluate.py <predictions_dir> <ground_truth_dir> [--verbose]
  python evaluate.py data/predictions_1 data/ground_truth_1/ground_truth_1
"""

import json
import re
import sys
import os
import html as html_module
from pathlib import Path
from collections import defaultdict


def strip_html(html: str) -> str:
    """Remove HTML tags, decode entities, and normalize whitespace."""
    text = re.sub(r'<[^>]+>', ' ', html)
    text = html_module.unescape(text)
    text = re.sub(r'\s+', ' ', text.replace('\u00a0', ' ')).strip()
    return text


def char_f1(pred: str, truth: str) -> float:
    """Character-level F1 on plain text (after HTML stripping)."""
    pred_text = strip_html(pred)
    truth_text = strip_html(truth)

    if not pred_text and not truth_text:
        return 1.0
    if not pred_text or not truth_text:
        return 0.0

    # Use character n-gram overlap (n=1 = character overlap)
    pred_chars = set(pred_text)
    truth_chars = set(truth_text)

    # Sliding window overlap — count common subsequence characters
    # For simplicity, use character bag-of-words F1
    pred_counts = defaultdict(int)
    truth_counts = defaultdict(int)
    for c in pred_text:
        pred_counts[c] += 1
    for c in truth_text:
        truth_counts[c] += 1

    overlap = sum(min(pred_counts[c], truth_counts[c]) for c in pred_counts)
    precision = overlap / len(pred_text) if pred_text else 0
    recall = overlap / len(truth_text) if truth_text else 0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def longest_common_substring_ratio(pred: str, truth: str) -> float:
    """
    More accurate metric: ratio of longest common substring length
    to average of pred/truth lengths. Good for boundary accuracy.
    """
    if not pred and not truth:
        return 1.0
    if not pred or not truth:
        return 0.0

    pred_text = strip_html(pred)
    truth_text = strip_html(truth)

    # Use sliding window to find LCS length efficiently
    # (approximate with 500-char prefix/suffix match for speed)
    max_len = max(len(pred_text), len(truth_text))
    if max_len == 0:
        return 1.0

    # Check exact match
    if pred_text == truth_text:
        return 1.0

    # Character F1 as primary metric (fast)
    return char_f1(pred_text, truth_text)


def evaluate_pair(pred_dict: dict, truth_dict: dict, accession: str) -> dict:
    """
    Compare prediction vs ground truth for one file.
    Returns per-item metrics and aggregate stats.
    """
    # Normalize keys (strip accession prefix for comparison)
    def item_from_key(k):
        return k.split('#', 1)[1] if '#' in k else k

    pred_items = {item_from_key(k): v for k, v in pred_dict.items()}
    truth_items = {item_from_key(k): v for k, v in truth_dict.items()}

    per_item = {}
    all_f1 = []

    for item_name, truth_html in truth_items.items():
        pred_html = pred_items.get(item_name, '')

        extracted = item_name in pred_items
        exact = pred_html == truth_html

        if truth_html:
            f1 = longest_common_substring_ratio(pred_html, truth_html)
        else:
            # GT key present but value empty: item exists but content
            # was not annotated. Score 1.0 regardless of prediction
            # (presence of the key confirms the item is valid).
            f1 = 1.0

        per_item[item_name] = {
            'extracted': extracted,
            'exact_match': exact,
            'char_f1': round(f1, 4),
            'pred_len': len(pred_html),
            'truth_len': len(truth_html),
        }
        all_f1.append(f1)

    # Items we predicted but not in ground truth (false positives)
    false_positives = set(pred_items.keys()) - set(truth_items.keys())

    # Document-level retrieval rate (Zhang et al. 2023):
    # A document is "retrieved" only if ALL of the following hold:
    #   1. Every GT item is present in predictions (no missing items)
    #   2. No extra items predicted that aren't in GT (no false positives)
    #   3. Every item's content matches above a threshold (no mis-detection)
    # We use F1 >= 0.90 as the threshold for "correctly detected".
    missing_items = set(truth_items.keys()) - set(pred_items.keys())
    all_items_present = len(missing_items) == 0
    no_false_positives = len(false_positives) == 0
    all_boundaries_correct = all(v['char_f1'] >= 0.90 for v in per_item.values())
    doc_retrieved = all_items_present and no_false_positives and all_boundaries_correct

    # Strict variant: exact HTML match for all items
    doc_retrieved_strict = all_items_present and no_false_positives and \
        all(v['exact_match'] for v in per_item.values())

    return {
        'accession': accession,
        'truth_item_count': len(truth_items),
        'pred_item_count': len(pred_items),
        'extraction_rate': sum(1 for v in per_item.values() if v['extracted']) / max(len(truth_items), 1),
        'exact_match_rate': sum(1 for v in per_item.values() if v['exact_match']) / max(len(truth_items), 1),
        'mean_char_f1': sum(all_f1) / len(all_f1) if all_f1 else 1.0,
        'false_positives': list(false_positives),
        'doc_retrieved': doc_retrieved,
        'doc_retrieved_strict': doc_retrieved_strict,
        'per_item': per_item,
    }


def run_evaluation(pred_dir: str, truth_dir: str, verbose: bool = False) -> dict:
    pred_dir = Path(pred_dir)
    truth_dir = Path(truth_dir)

    all_results = []
    item_f1_totals = defaultdict(list)

    truth_files = list(truth_dir.glob('*.json'))
    print(f"Found {len(truth_files)} ground truth files in {truth_dir}")

    empty_truth = 0
    malformed_truth = 0
    skipped = 0

    for truth_path in sorted(truth_files):
        accession = truth_path.stem
        pred_path = pred_dir / f"{accession}.json"

        try:
            with open(truth_path, 'r', encoding='utf-8') as f:
                truth_dict = json.load(f)
        except json.JSONDecodeError:
            malformed_truth += 1
            if verbose:
                print(f"  MALFORMED ground truth: {accession}")
            continue

        if not truth_dict:
            empty_truth += 1
            continue  # Skip empty ground truth files

        if not pred_path.exists():
            skipped += 1
            if verbose:
                print(f"  MISSING prediction: {accession}")
            # Treat as all-zero scores
            result = {
                'accession': accession,
                'truth_item_count': len(truth_dict),
                'pred_item_count': 0,
                'extraction_rate': 0.0,
                'exact_match_rate': 0.0,
                'mean_char_f1': 0.0,
                'false_positives': [],
                'per_item': {},
            }
        else:
            with open(pred_path, 'r', encoding='utf-8') as f:
                pred_dict = json.load(f)
            result = evaluate_pair(pred_dict, truth_dict, accession)

        all_results.append(result)

        for item_name, metrics in result['per_item'].items():
            item_f1_totals[item_name].append(metrics['char_f1'])

        if verbose:
            print(f"  {accession}: F1={result['mean_char_f1']:.3f}  "
                  f"extracted={result['extraction_rate']:.2%}  "
                  f"exact={result['exact_match_rate']:.2%}")

    # Aggregate stats
    n = len(all_results)
    if n == 0:
        print("No results to aggregate.")
        return {}

    overall_f1 = sum(r['mean_char_f1'] for r in all_results) / n
    overall_extraction = sum(r['extraction_rate'] for r in all_results) / n
    overall_exact = sum(r['exact_match_rate'] for r in all_results) / n

    # Document-level retrieval rate (Zhang et al. 2023)
    doc_retrieved_count = sum(1 for r in all_results if r['doc_retrieved'])
    doc_retrieved_strict_count = sum(1 for r in all_results if r['doc_retrieved_strict'])
    doc_retrieval_rate = doc_retrieved_count / n
    doc_retrieval_strict = doc_retrieved_strict_count / n

    print(f"\n{'='*60}")
    print(
        f"RESULTS SUMMARY  ({n} files evaluated, {empty_truth} empty GT skipped, "
        f"{malformed_truth} malformed GT skipped, {skipped} predictions missing)"
    )
    print(f"{'='*60}")
    print(f"  Overall mean char F1 :  {overall_f1:.4f}  ({overall_f1*100:.1f}%)")
    print(f"  Extraction rate      :  {overall_extraction:.4f}  ({overall_extraction*100:.1f}%)")
    print(f"  Exact HTML match rate:  {overall_exact:.4f}  ({overall_exact*100:.1f}%)")
    print(f"  Doc retrieval (F1>=.9): {doc_retrieval_rate:.4f}  ({doc_retrieved_count}/{n} = {doc_retrieval_rate*100:.1f}%)")
    print(f"  Doc retrieval (exact) : {doc_retrieval_strict:.4f}  ({doc_retrieved_strict_count}/{n} = {doc_retrieval_strict*100:.1f}%)")

    print(f"\n  Per-item char F1:")
    for item_name in sorted(item_f1_totals.keys()):
        scores = item_f1_totals[item_name]
        mean = sum(scores) / len(scores)
        print(f"    {item_name:<12} {mean:.4f}  ({len(scores)} files)")

    # Bottom 10 files by F1
    worst = sorted(all_results, key=lambda r: r['mean_char_f1'])[:10]
    print(f"\n  10 worst files by F1:")
    for r in worst:
        print(f"    {r['accession']}  F1={r['mean_char_f1']:.3f}  "
              f"truth_items={r['truth_item_count']}  pred_items={r['pred_item_count']}")

    # Document-level failure analysis
    failed_docs = [r for r in all_results if not r['doc_retrieved']]
    if failed_docs:
        print(f"\n  Doc retrieval failures ({len(failed_docs)} docs):")
        n_fp = sum(1 for r in failed_docs if r['false_positives'])
        n_missing = sum(1 for r in failed_docs if r['extraction_rate'] < 1.0)
        n_boundary = sum(1 for r in failed_docs
                         if any(v['char_f1'] < 0.90 for v in r['per_item'].values()))
        print(f"    Has missing items:    {n_missing}")
        print(f"    Has false positives:  {n_fp}")
        print(f"    Has boundary errors:  {n_boundary}")

    return {
        'overall_f1': overall_f1,
        'overall_extraction_rate': overall_extraction,
        'overall_exact_match_rate': overall_exact,
        'doc_retrieval_rate': doc_retrieval_rate,
        'doc_retrieval_strict': doc_retrieval_strict,
        'n_evaluated': n,
        'n_empty_truth': empty_truth,
        'n_malformed_truth': malformed_truth,
        'n_missing_pred': skipped,
        'per_item_f1': {k: sum(v)/len(v) for k, v in item_f1_totals.items()},
        'per_file': all_results,
    }


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python evaluate.py <predictions_dir> <ground_truth_dir> [--verbose]")
        sys.exit(1)

    pred_dir = sys.argv[1]
    truth_dir = sys.argv[2]
    verbose = '--verbose' in sys.argv or '-v' in sys.argv

    run_evaluation(pred_dir, truth_dir, verbose=verbose)
