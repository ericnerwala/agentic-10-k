"""
ML Anchor Item Classifier
==========================
Quick-win ML model: train a multi-class classifier that predicts which
10-K item an anchor belongs to, using features from HTML context.

The classifier's predicted probabilities replace regex confidence scores
in the DP sequence solver, improving anchor selection for ambiguous cases.

Train on Set 1, evaluate end-to-end F1 on Set 3.
"""

import json
import re
import html as html_module
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

from extract import (
    extract_10k_text, collect_toc_referenced_ids, find_all_id_elements,
    parse_toc_links, classify_anchors, extract_item_slices,
    normalize_text, _classify_tier1, _classify_tier2, _classify_anchor_id,
    ITEM_SEQ_INDEX, ITEM_SEQ_ORDER,
    _sequence_assign_dp,
)


def strip_html(h):
    t = re.sub(r'<[^>]+>', ' ', h)
    t = html_module.unescape(t)
    return re.sub(r'\s+', ' ', t.replace('\u00a0', ' ')).strip()


def char_f1(pred, truth):
    p, t = strip_html(pred), strip_html(truth)
    if not p and not t: return 1.0
    if not p or not t: return 0.0
    pc, tc = Counter(p), Counter(t)
    overlap = sum(min(pc[c], tc[c]) for c in pc)
    prec = overlap / len(p)
    rec = overlap / len(t)
    return 2 * prec * rec / (prec + rec) if prec + rec else 0


# ---------------------------------------------------------------
# Feature extraction (richer than binary re-scorer)
# ---------------------------------------------------------------
_BOLD_RE = re.compile(r'font-weight\s*:\s*(?:bold|[6-9]\d{2})', re.I)
_HEADING_RE = re.compile(r'<h[1-6][^>]*>', re.I)
_FONT_SIZE_RE = re.compile(r'font-size\s*:\s*(\d+)', re.I)

# Keywords associated with each item (for text similarity features)
ITEM_KEYWORDS = {
    'item1': ['business'],
    'item1a': ['risk', 'factors'],
    'item1b': ['unresolved', 'staff', 'comments'],
    'item2': ['properties'],
    'item3': ['legal', 'proceedings'],
    'item4': ['mine', 'safety'],
    'item5': ['market', 'registrant', 'equity', 'stock'],
    'item6': ['selected', 'financial', 'data', 'reserved'],
    'item7': ['management', 'discussion', 'analysis'],
    'item7a': ['quantitative', 'qualitative', 'market', 'risk'],
    'item8': ['financial', 'statements', 'supplementary'],
    'item9': ['changes', 'disagreements', 'accountants'],
    'item9a': ['controls', 'procedures'],
    'item9b': ['other', 'information'],
    'item10': ['directors', 'officers', 'governance'],
    'item11': ['executive', 'compensation'],
    'item12': ['security', 'ownership'],
    'item13': ['certain', 'relationships'],
    'item14': ['principal', 'accountant', 'fees'],
    'item15': ['exhibits', 'schedules'],
    'item16': ['summary', '10-k'],
    'signatures': ['signatures', 'pursuant'],
}


def extract_features(html_text, anchor_id, offset, tag_name, attr_name,
                     next_offset, doc_len, toc_mappings, anchor_to_items):
    """Extract feature vector for anchor classification."""
    max_fwd = min(2000, (next_offset - offset) if next_offset else 2000)
    fwd_html = html_text[offset:offset + max_fwd]
    fwd_text = normalize_text(re.sub(r'<[^>]+>', ' ', fwd_html))
    back_html = html_text[max(0, offset - 300):offset]

    feats = {}

    # Position
    feats['rel_pos'] = offset / max(doc_len, 1)

    # Tag type (one-hot)
    for t in ['a', 'div', 'p', 'span', 'td', 'tr']:
        feats[f'tag_{t}'] = int(tag_name == t)
    feats['attr_id'] = int(attr_name == 'id')

    # Anchor ID features
    feats['id_len'] = len(anchor_id)
    feats['id_has_item'] = int(bool(re.search(r'item', anchor_id, re.I)))

    # Tier 0 (ID-based) classification
    id_class = _classify_anchor_id(anchor_id)
    for item in ITEM_SEQ_ORDER:
        feats[f'id_is_{item}'] = int(id_class == item)

    # TOC mapping
    feats['in_toc'] = int(bool(toc_mappings) and anchor_id in toc_mappings.values())

    # Text features: first 150 chars
    text_150 = fwd_text[:150]
    t1 = _classify_tier1(text_150)
    t2 = _classify_tier2(text_150)
    for item in ITEM_SEQ_ORDER:
        feats[f't1_{item}'] = int(t1 == item)
        feats[f't2_{item}'] = int(t2 == item)

    # Keyword presence in first 300 chars of forward text
    fwd_300 = fwd_text[:300].lower()
    for item, keywords in ITEM_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in fwd_300) / len(keywords)
        feats[f'kw_{item}'] = score

    # Styling
    ctx = html_text[max(0, offset - 200):offset + 200]
    feats['bold'] = int(bool(_BOLD_RE.search(ctx)))
    feats['heading'] = int(bool(_HEADING_RE.search(ctx)))
    sizes = _FONT_SIZE_RE.findall(ctx)
    feats['font_size'] = max((int(s) for s in sizes), default=0)

    # Forward text length
    feats['fwd_len'] = len(fwd_text)

    # Part headers
    feats['has_part'] = int(bool(re.search(r'\bpart\s+[iv]+\b', fwd_text[:200], re.I)))

    return feats


# ---------------------------------------------------------------
# Build training data
# ---------------------------------------------------------------
def build_classification_data(input_dir, truth_dir):
    """Build (X, y) for multi-class anchor item classification."""
    input_dir, truth_dir = Path(input_dir), Path(truth_dir)
    X_rows, y_labels, metas = [], [], []

    item_to_idx = {item: i for i, item in enumerate(ITEM_SEQ_ORDER)}
    item_to_idx['__none__'] = len(ITEM_SEQ_ORDER)

    for txt_path in sorted(input_dir.glob('*.txt')):
        acc = txt_path.stem
        truth_path = truth_dir / f'{acc}.json'
        if not truth_path.exists():
            continue
        with open(truth_path, encoding='utf-8') as f:
            truth = json.load(f)
        if not truth:
            continue

        html_text = extract_10k_text(str(txt_path))
        if not html_text:
            continue
        ref_ids = collect_toc_referenced_ids(html_text)
        if not ref_ids:
            continue
        toc_map, anc_items = parse_toc_links(html_text, ref_ids)
        id_pos = find_all_id_elements(html_text, ref_ids)
        if not id_pos:
            continue

        # Derive labels: find GT boundary positions
        gt_boundaries = {}
        for key, gt_html in truth.items():
            item = key.split('#')[1] if '#' in key else key
            if not gt_html or len(gt_html) < 10:
                continue
            needle = gt_html[:200]
            pos = html_text.find(needle)
            if pos == -1:
                needle = gt_html[:50]
                pos = html_text.find(needle)
            if pos >= 0:
                gt_boundaries[item] = pos

        # Match each anchor to nearest GT boundary
        sorted_ancs = sorted(id_pos.items(), key=lambda x: x[1][0])
        for idx, (aid, (off, tname, aname)) in enumerate(sorted_ancs):
            next_off = sorted_ancs[idx + 1][1][0] if idx + 1 < len(sorted_ancs) else None

            feats = extract_features(
                html_text, aid, off, tname, aname,
                next_off, len(html_text), toc_map, anc_items
            )

            # Label: which item is this anchor's GT match?
            best_item = '__none__'
            best_dist = 500
            for item, gpos in gt_boundaries.items():
                d = abs(off - gpos)
                if d < best_dist:
                    best_dist = d
                    best_item = item

            X_rows.append(feats)
            y_labels.append(item_to_idx.get(best_item, item_to_idx['__none__']))
            metas.append((acc, aid, best_item))

    feat_names = sorted(X_rows[0].keys())
    X = np.array([[r[f] for f in feat_names] for r in X_rows])
    y = np.array(y_labels)
    return X, y, feat_names, metas, item_to_idx


# ---------------------------------------------------------------
# End-to-end evaluation with ML-boosted confidence
# ---------------------------------------------------------------
def evaluate_with_ml(clf, feat_names, input_dir, truth_dir, item_to_idx):
    """Run extraction with ML confidence blending and compute F1."""
    input_dir, truth_dir = Path(input_dir), Path(truth_dir)
    idx_to_item = {v: k for k, v in item_to_idx.items()}
    all_f1s = []

    for txt_path in sorted(input_dir.glob('*.txt')):
        acc = txt_path.stem
        truth_path = truth_dir / f'{acc}.json'
        if not truth_path.exists():
            continue
        with open(truth_path, encoding='utf-8') as f:
            truth = json.load(f)
        if not truth:
            continue

        html_text = extract_10k_text(str(txt_path))
        if not html_text:
            continue
        ref_ids = collect_toc_referenced_ids(html_text)
        if not ref_ids:
            continue
        toc_map, anc_items = parse_toc_links(html_text, ref_ids)
        id_pos = find_all_id_elements(html_text, ref_ids)
        if not id_pos:
            continue

        # Extract features for all anchors
        sorted_ancs = sorted(id_pos.items(), key=lambda x: x[1][0])
        anchor_features = []
        for idx, (aid, (off, tname, aname)) in enumerate(sorted_ancs):
            next_off = sorted_ancs[idx + 1][1][0] if idx + 1 < len(sorted_ancs) else None
            feats = extract_features(
                html_text, aid, off, tname, aname,
                next_off, len(html_text), toc_map, anc_items
            )
            anchor_features.append([feats[f] for f in feat_names])

        if not anchor_features:
            continue

        X_anc = np.array(anchor_features)
        probs = clf.predict_proba(X_anc)

        # Build candidates dict with ML-boosted confidence
        candidates = {}
        for idx, (aid, (off, tname, aname)) in enumerate(sorted_ancs):
            # For each possible item, compute blended confidence
            for item_idx, prob in enumerate(probs[idx]):
                item = idx_to_item.get(item_idx, '__none__')
                if item == '__none__' or prob < 0.05:
                    continue
                # Blend: ML probability * 10 as confidence
                confidence = int(prob * 10)
                if item not in candidates:
                    candidates[item] = []
                candidates[item].append((off, aid, confidence))

        # Run DP assignment
        assigned = _sequence_assign_dp(candidates)

        # Signatures fallback
        if not any(it == 'signatures' for _, it, _ in assigned):
            from extract import _find_signatures_fallback
            sig_off = _find_signatures_fallback(html_text, assigned)
            if sig_off is not None:
                assigned.append((sig_off, 'signatures', '__fb_sig__'))
                assigned.sort(key=lambda x: x[0])

        # Slice and evaluate
        slices = extract_item_slices(html_text, assigned)
        pred = {}
        for item, html_slice in slices.items():
            key = f'{acc}#{item}'
            if item == 'signatures':
                pred[key] = ''
            else:
                pred[key] = html_slice

        file_f1s = []
        for key, tv in truth.items():
            pv = pred.get(key, '')
            if tv:
                f1 = char_f1(pv, tv)
            else:
                f1 = 1.0 if not pv else 0.5
            file_f1s.append(f1)

        if file_f1s:
            all_f1s.append(sum(file_f1s) / len(file_f1s))

    return sum(all_f1s) / len(all_f1s) if all_f1s else 0


def main():
    base = Path(__file__).parent / 'data'

    print('=== Building training data from Set 1 ===')
    X_train, y_train, feat_names, meta_train, item_to_idx = build_classification_data(
        base / 'folder_1' / 'folder_1',
        base / 'ground_truth_1' / 'ground_truth_1',
    )
    n_classes = len(item_to_idx)
    print(f'  Samples: {len(y_train)}, Classes: {n_classes}')
    print(f'  Class distribution: {Counter(y_train).most_common(5)}...')

    print('\n=== Training GradientBoostingClassifier ===')
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.multiclass import OneVsRestClassifier

    # Random forest is faster for multi-class
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Training accuracy
    train_acc = clf.score(X_train, y_train)
    print(f'  Training accuracy: {train_acc:.4f}')

    # --- Baseline: rule-based F1 on Set 3 ---
    print('\n=== Baseline (rule-based) F1 on Set 3 ===')
    from evaluate import run_evaluation
    import io, contextlib

    # Run rule-based extraction on set 3
    from run_all import process_one
    pred_dir = base / 'predictions_3'
    pred_dir.mkdir(exist_ok=True)

    input_dir = base / 'folder_3' / 'folder_3'
    truth_dir = base / 'ground_truth_3' / 'ground_truth_3'

    # Already have predictions from earlier run, evaluate directly
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        baseline = run_evaluation(str(pred_dir), str(truth_dir))
    baseline_f1 = baseline['overall_f1']
    print(f'  Baseline F1: {baseline_f1:.4f} ({baseline_f1*100:.2f}%)')

    # --- ML-boosted F1 on Set 3 ---
    print('\n=== ML-boosted F1 on Set 3 ===')
    ml_f1 = evaluate_with_ml(clf, feat_names, input_dir, truth_dir, item_to_idx)
    print(f'  ML-boosted F1: {ml_f1:.4f} ({ml_f1*100:.2f}%)')
    print(f'  Delta: {(ml_f1 - baseline_f1)*100:+.2f} points')

    # Feature importance
    print('\n=== Top 15 features ===')
    importances = clf.feature_importances_
    for i in np.argsort(importances)[::-1][:15]:
        print(f'  {feat_names[i]:<25} {importances[i]:.4f}')


if __name__ == '__main__':
    main()
