"""
GT Noise Analysis
==================
Evaluate the adjusted retrieval rate by classifying each failure as
GT annotation noise vs genuine pipeline error.

GT noise criteria:
  - signatures missing: GT has empty/trivial signatures, or GT source file differs
  - signatures FP: filing has clear SIGNATURES section but GT omits the key
  - signatures boundary: GT has 1M+ chars (source file mismatch)
  - item16 FP: GT omits item16 despite it being in TOC with placeholder content
  - item16 boundary: both GT and pred are short placeholders, or GT source mismatch
  - item16 missing: GT has trivial/short item16
"""

import json
import re
import html as html_module
from pathlib import Path
from collections import defaultdict, Counter

from extract import (
    extract_10k_text, collect_toc_referenced_ids, parse_toc_links,
    find_all_id_elements, classify_anchors, _is_placeholder_item16,
)


def strip_html(h):
    t = re.sub(r'<[^>]+>', ' ', h)
    t = html_module.unescape(t)
    return re.sub(r'\s+', ' ', t.replace('\u00a0', ' ')).strip()


def char_f1(pred, truth):
    pt, tt = strip_html(pred), strip_html(truth)
    if not pt and not tt:
        return 1.0
    if not pt or not tt:
        return 0.0
    pc, tc = defaultdict(int), defaultdict(int)
    for c in pt:
        pc[c] += 1
    for c in tt:
        tc[c] += 1
    ov = sum(min(pc[c], tc[c]) for c in pc)
    p, r = ov / len(pt), ov / len(tt)
    return 2 * p * r / (p + r) if p + r else 0


def classify_failure(item, failure_type, gi, pi, folder, acc):
    """Classify a single item failure as 'noise' or 'real'.
    Returns (classification, reason) where classification is 'noise' or 'real'.
    """
    if failure_type == 'missing':
        if item == 'signatures':
            gt_val = gi.get('signatures', '')
            if not gt_val or len(gt_val) < 100:
                return 'noise', 'missing signatures (GT empty/trivial)'
            txt_path = folder / f'{acc}.txt'
            file_size = txt_path.stat().st_size if txt_path.exists() else 0
            if len(gt_val) > file_size * 0.3:
                return 'noise', f'missing signatures (GT {len(gt_val):,} chars, source file mismatch)'
            # Check if we detect signatures but suppress it
            if txt_path.exists():
                html_text = extract_10k_text(str(txt_path))
                refs = collect_toc_referenced_ids(html_text)
                toc, a2i = parse_toc_links(html_text, refs)
                id_pos = find_all_id_elements(html_text, refs)
                anchors = classify_anchors(html_text, id_pos, toc, a2i)
                has_sig = any(i == 'signatures' for _, i, _ in anchors)
                if has_sig:
                    return 'noise', 'missing signatures (detected but suppressed, GT inconsistent)'
            return 'real', 'missing signatures (not detected)'

        elif item == 'item16':
            gt_val = gi.get('item16', '')
            gt_text = strip_html(gt_val)
            if len(gt_text) < 100:
                return 'noise', f'missing item16 (GT trivial: {len(gt_text)} chars text)'
            return 'real', f'missing item16 (GT has {len(gt_text)} chars text)'

        else:
            return 'real', f'missing {item}'

    elif failure_type == 'fp':
        if item == 'signatures':
            return 'noise', 'FP signatures (filing has SIGNATURES heading, GT omits key)'
        elif item == 'item16':
            pred_val = pi.get('item16', '')
            if _is_placeholder_item16(pred_val) or len(pred_val) < 3000:
                return 'noise', 'FP item16 (placeholder content, GT inconsistently omits)'
            return 'real', f'FP item16 (substantial content: {len(pred_val)} chars)'
        else:
            return 'real', f'FP {item}'

    elif failure_type == 'boundary':
        gt_val = gi.get(item, '')
        pred_val = pi.get(item, '')
        f1 = char_f1(pred_val, gt_val)

        if item == 'signatures':
            if len(gt_val) > 1_000_000:
                return 'noise', f'boundary signatures (GT {len(gt_val):,} chars, source file mismatch)'
            return 'real', f'boundary signatures (F1={f1:.3f})'

        elif item == 'item16':
            gt_text = strip_html(gt_val)
            pred_text = strip_html(pred_val)
            if len(gt_text) < 100 and len(pred_text) < 100:
                return 'noise', f'boundary item16 (both short: GT={len(gt_text)}, pred={len(pred_text)} chars)'
            if len(gt_val) > 1_000_000:
                return 'noise', f'boundary item16 (GT {len(gt_val):,} chars, source file mismatch)'
            return 'real', f'boundary item16 (F1={f1:.3f})'

        else:
            return 'real', f'boundary {item} (F1={f1:.3f})'


def main():
    gt_noise_docs = set()
    partial_noise = set()
    real_failure_docs = set()
    noise_log = []
    real_log = []
    total_evaluated = 0
    total_passing = 0

    for s in [1, 2, 3]:
        gt_dir = Path(f'data/ground_truth_{s}/ground_truth_{s}')
        pred_dir = Path(f'data/predictions_{s}')
        folder = Path(f'data/folder_{s}/folder_{s}')

        for gt_path in sorted(gt_dir.glob('*.json')):
            try:
                with open(gt_path) as f:
                    gt = json.load(f)
            except Exception:
                continue
            if not gt:
                continue
            acc = gt_path.stem
            pred_path = pred_dir / f'{acc}.json'
            if not pred_path.exists():
                continue
            with open(pred_path) as f:
                pred = json.load(f)

            gi = {k.split('#', 1)[1] if '#' in k else k: v for k, v in gt.items()}
            pi = {k.split('#', 1)[1] if '#' in k else k: v for k, v in pred.items()}

            missing = set(gi.keys()) - set(pi.keys())
            fps = set(pi.keys()) - set(gi.keys())
            boundary_fails = {}
            for item, tv in gi.items():
                pv = pi.get(item, '')
                if not tv:
                    continue
                if item in missing:
                    continue
                f1 = char_f1(pv, tv)
                if f1 < 0.9:
                    boundary_fails[item] = f1

            total_evaluated += 1
            if not missing and not fps and not boundary_fails:
                total_passing += 1
                continue

            doc_key = f'set{s}/{acc}'
            noise_issues = []
            real_issues = []

            for m in missing:
                cls, reason = classify_failure(m, 'missing', gi, pi, folder, acc)
                if cls == 'noise':
                    noise_issues.append(reason)
                else:
                    real_issues.append(reason)

            for fp in fps:
                cls, reason = classify_failure(fp, 'fp', gi, pi, folder, acc)
                if cls == 'noise':
                    noise_issues.append(reason)
                else:
                    real_issues.append(reason)

            for b_item in boundary_fails:
                cls, reason = classify_failure(b_item, 'boundary', gi, pi, folder, acc)
                if cls == 'noise':
                    noise_issues.append(reason)
                else:
                    real_issues.append(reason)

            if noise_issues and not real_issues:
                gt_noise_docs.add(doc_key)
                for r in noise_issues:
                    noise_log.append((doc_key, r))
            elif real_issues and not noise_issues:
                real_failure_docs.add(doc_key)
                for r in real_issues:
                    real_log.append((doc_key, r))
            else:
                partial_noise.add(doc_key)
                for r in noise_issues:
                    noise_log.append((doc_key, r))
                for r in real_issues:
                    real_log.append((doc_key, r))

    # --- Report ---
    print('=' * 70)
    print('GT NOISE ANALYSIS — ADJUSTED RETRIEVAL RATE')
    print('=' * 70)
    print()
    n_fail = len(gt_noise_docs) + len(partial_noise) + len(real_failure_docs)
    print(f'Total evaluated:             {total_evaluated}')
    print(f'Currently passing:           {total_passing}')
    print(f'Currently failing:           {n_fail}')
    print(f'  All failures are GT noise: {len(gt_noise_docs)}')
    print(f'  Mixed (noise + real):      {len(partial_noise)}')
    print(f'  All failures are real:     {len(real_failure_docs)}')
    print()
    print(f'--- RETRIEVAL RATES ---')
    raw = total_passing / total_evaluated
    print(f'Raw rate:                    {total_passing}/{total_evaluated} = {raw*100:.1f}%')

    adjusted = total_passing + len(gt_noise_docs)
    adj_rate = adjusted / total_evaluated
    print(f'Adjusted (noise=pass):       {adjusted}/{total_evaluated} = {adj_rate*100:.1f}%')

    clean_total = total_evaluated - len(gt_noise_docs) - len(partial_noise)
    clean_rate = total_passing / clean_total if clean_total > 0 else 0
    print(f'Clean GT only:               {total_passing}/{clean_total} = {clean_rate*100:.1f}%')

    # Breakdown of noise reasons
    print()
    print(f'--- GT NOISE BREAKDOWN ({len(gt_noise_docs)} all-noise docs) ---')
    noise_cats = Counter()
    for doc, reason in noise_log:
        if doc in gt_noise_docs:
            key = reason.split('(')[0].strip()
            noise_cats[key] += 1
    for cat, cnt in noise_cats.most_common():
        print(f'  {cat:<45} {cnt:>3}')

    print()
    print(f'--- ALL-NOISE DOCS (would pass with clean GT) ---')
    for doc, reason in sorted(noise_log):
        if doc in gt_noise_docs:
            print(f'  {doc:<35} {reason}')

    print()
    print(f'--- REAL FAILURE BREAKDOWN ({len(real_failure_docs)} real-only docs) ---')
    real_cats = Counter()
    for doc, reason in real_log:
        if doc in real_failure_docs:
            key = reason.split('(')[0].strip()
            real_cats[key] += 1
    for cat, cnt in real_cats.most_common():
        print(f'  {cat:<45} {cnt:>3}')


if __name__ == '__main__':
    main()
