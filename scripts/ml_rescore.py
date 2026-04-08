"""
ML Anchor Re-scorer: Quick-win proof of concept
================================================
Trains a gradient-boosted classifier to re-rank anchor candidates
when the rule-based pipeline assigns low confidence.

Approach:
  1. For each filing, extract features from ALL anchor candidates
     (not just the DP-selected ones).
  2. Derive training labels by matching ground truth slice boundaries
     to the nearest anchor position.
  3. Train a LightGBM / sklearn classifier to predict P(correct anchor | features).
  4. At inference, blend ML confidence with rule-based confidence in the DP.

This targets the ~25% of F1 loss from boundary misclassifications
(items 6, 7, 7a, 9b, 15) without touching item16 or signatures.
"""

import json
import re
import html as html_module
import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

from extract import (
    extract_10k_text, collect_toc_referenced_ids, find_all_id_elements,
    parse_toc_links, classify_anchors, extract_item_slices,
    normalize_text, _classify_tier1, _classify_tier2, _classify_anchor_id,
    _looks_like_toc_candidate, ITEM_SEQ_INDEX,
    _ANCHOR_ID_PATTERNS, ITEM_PATTERNS, TITLE_PATTERNS,
)


def strip_html(h):
    t = re.sub(r'<[^>]+>', ' ', h)
    t = html_module.unescape(t)
    return re.sub(r'\s+', ' ', t.replace('\u00a0', ' ')).strip()


# ---------------------------------------------------------------
# Feature extraction for each anchor candidate
# ---------------------------------------------------------------
_BOLD_RE = re.compile(r'font-weight\s*:\s*(?:bold|[6-9]\d{2})', re.I)
_HEADING_RE = re.compile(r'<h[1-6][^>]*>', re.I)
_FONT_SIZE_RE = re.compile(r'font-size\s*:\s*(\d+)', re.I)


def extract_anchor_features(html_text, anchor_id, offset, tag_name, attr_name,
                            next_offset, doc_len, toc_mappings, anchor_to_items):
    """Extract a feature vector for a single anchor candidate."""
    max_forward = min(2000, (next_offset - offset) if next_offset else 2000)

    # Forward and backward context
    lookahead_html = html_text[offset:offset + max_forward]
    lookahead_text = normalize_text(re.sub(r'<[^>]+>', ' ', lookahead_html))
    back_html = html_text[max(0, offset - 500):offset]
    back_text = normalize_text(re.sub(r'<[^>]+>', ' ', back_html))

    features = {}

    # --- Position features ---
    features['rel_position'] = offset / max(doc_len, 1)
    features['abs_offset'] = offset

    # --- Tag features ---
    features['tag_is_a'] = int(tag_name == 'a')
    features['tag_is_div'] = int(tag_name == 'div')
    features['tag_is_p'] = int(tag_name == 'p')
    features['tag_is_span'] = int(tag_name == 'span')
    features['tag_is_td'] = int(tag_name == 'td')
    features['attr_is_id'] = int(attr_name == 'id')

    # --- Anchor ID features ---
    features['id_len'] = len(anchor_id)
    features['id_has_item'] = int(bool(re.search(r'item', anchor_id, re.I)))
    features['id_has_toc'] = int(bool(re.search(r'toc', anchor_id, re.I)))
    features['id_has_sig'] = int(bool(re.search(r'sig', anchor_id, re.I)))

    # Tier 0 classification from ID
    id_class = _classify_anchor_id(anchor_id)
    features['tier0_match'] = int(id_class is not None)

    # --- TOC mapping features ---
    features['in_toc_mapping'] = int(
        bool(toc_mappings) and anchor_id in toc_mappings.values()
    )
    features['n_toc_items'] = len(anchor_to_items.get(anchor_id, [])) if anchor_to_items else 0

    # --- Text features ---
    features['forward_text_len'] = len(lookahead_text)
    features['back_text_len'] = len(back_text)

    # Count item mentions in forward text
    item_mentions = set()
    for m in re.finditer(r'item\s*\d+', lookahead_text[:500], re.I):
        name = _classify_tier1(m.group(0))
        if name:
            item_mentions.add(name)
    features['n_item_mentions_fwd'] = len(item_mentions)

    # Tier 1 and Tier 2 match in first 150 chars
    text_150 = lookahead_text[:150]
    features['tier1_match_150'] = int(_classify_tier1(text_150) is not None)
    features['tier2_match_150'] = int(_classify_tier2(text_150) is not None)
    features['tier1_match_full'] = int(_classify_tier1(lookahead_text) is not None)
    features['tier2_match_full'] = int(_classify_tier2(lookahead_text) is not None)

    # --- Styling features ---
    context_html = html_text[max(0, offset - 300):offset + 300]
    features['has_bold'] = int(bool(_BOLD_RE.search(context_html)))
    features['has_heading'] = int(bool(_HEADING_RE.search(context_html)))

    font_sizes = _FONT_SIZE_RE.findall(context_html)
    features['max_font_size'] = max((int(s) for s in font_sizes), default=0)

    # --- Content features ---
    features['starts_with_item'] = int(bool(re.match(r'item\s*\d', lookahead_text, re.I)))
    features['has_part_header'] = int(bool(re.search(r'\bpart\s+[iv]+\b', lookahead_text[:200], re.I)))

    return features


# ---------------------------------------------------------------
# Label derivation: match GT boundaries to anchor positions
# ---------------------------------------------------------------
def derive_labels(html_text, id_positions, truth_dict, accession):
    """
    For each anchor candidate, determine if it's the correct boundary
    for some GT item by finding the GT slice start position.
    Returns {anchor_id: item_name} for correct matches.
    """
    correct = {}

    for key, gt_html in truth_dict.items():
        item = key.split('#')[1] if '#' in key else key
        if not gt_html or len(gt_html) < 10:
            continue

        # Find where this GT slice starts in the document
        # Use first 200 chars of GT as search needle
        needle = gt_html[:200]
        pos = html_text.find(needle)
        if pos == -1:
            # Try shorter needle
            needle = gt_html[:50]
            pos = html_text.find(needle)
        if pos == -1:
            continue

        # Find nearest anchor to this position
        best_anchor = None
        best_dist = float('inf')
        for aid, (aoff, _, _) in id_positions.items():
            dist = abs(aoff - pos)
            if dist < best_dist:
                best_dist = dist
                best_anchor = aid
                best_off = aoff

        if best_anchor and best_dist < 500:
            correct[best_anchor] = item

    return correct


# ---------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------
def build_dataset(input_dir, truth_dir):
    """Build feature matrix X and label vector y from a set of filings."""
    X_rows = []
    y_labels = []
    meta = []  # (accession, anchor_id, assigned_item)

    input_dir = Path(input_dir)
    truth_dir = Path(truth_dir)

    for txt_path in sorted(input_dir.glob('*.txt')):
        accession = txt_path.stem
        truth_path = truth_dir / f'{accession}.json'
        if not truth_path.exists():
            continue

        with open(truth_path, encoding='utf-8') as f:
            truth_dict = json.load(f)
        if not truth_dict:
            continue

        # Run pipeline stages 1-3
        html_text = extract_10k_text(str(txt_path))
        if not html_text:
            continue

        referenced_ids = collect_toc_referenced_ids(html_text)
        if not referenced_ids:
            continue

        toc_mappings, anchor_to_items = parse_toc_links(html_text, referenced_ids)
        id_positions = find_all_id_elements(html_text, referenced_ids)
        if not id_positions:
            continue

        # Derive ground truth labels
        correct_anchors = derive_labels(html_text, id_positions, truth_dict, accession)

        # Sort anchors by position
        sorted_anchors = sorted(id_positions.items(), key=lambda x: x[1][0])
        doc_len = len(html_text)

        for idx, (anchor_id, (offset, tag_name, attr_name)) in enumerate(sorted_anchors):
            next_offset = sorted_anchors[idx + 1][1][0] if idx + 1 < len(sorted_anchors) else None

            feats = extract_anchor_features(
                html_text, anchor_id, offset, tag_name, attr_name,
                next_offset, doc_len, toc_mappings, anchor_to_items
            )

            # Label: 1 if this anchor is a correct GT boundary, 0 otherwise
            is_correct = anchor_id in correct_anchors
            assigned_item = correct_anchors.get(anchor_id, None)

            X_rows.append(feats)
            y_labels.append(int(is_correct))
            meta.append((accession, anchor_id, assigned_item))

    # Convert to arrays
    if not X_rows:
        return None, None, None, None

    feature_names = sorted(X_rows[0].keys())
    X = np.array([[row[f] for f in feature_names] for row in X_rows])
    y = np.array(y_labels)

    return X, y, feature_names, meta


# ---------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------
def main():
    base = Path(__file__).parent / 'data'

    print('Building training set from Set 1...')
    X_train, y_train, feature_names, meta_train = build_dataset(
        base / 'folder_1' / 'folder_1',
        base / 'ground_truth_1' / 'ground_truth_1'
    )
    print(f'  Samples: {len(y_train)}, Positive: {y_train.sum()}, '
          f'Negative: {(1-y_train).sum()}, Ratio: {y_train.mean():.3f}')

    print('\nBuilding test set from Set 3...')
    X_test, y_test, _, meta_test = build_dataset(
        base / 'folder_3' / 'folder_3',
        base / 'ground_truth_3' / 'ground_truth_3'
    )
    print(f'  Samples: {len(y_test)}, Positive: {y_test.sum()}, '
          f'Negative: {(1-y_test).sum()}, Ratio: {y_test.mean():.3f}')

    # Try sklearn GradientBoosting (available without extra installs)
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import classification_report, roc_auc_score

    print('\nTraining GradientBoostingClassifier...')
    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print('\nTest set classification report:')
    print(classification_report(y_test, y_pred, target_names=['not_boundary', 'boundary']))
    print(f'ROC AUC: {roc_auc_score(y_test, y_prob):.4f}')

    # Feature importance
    print('\nTop 10 features by importance:')
    importances = clf.feature_importances_
    for idx in np.argsort(importances)[::-1][:10]:
        print(f'  {feature_names[idx]:<25} {importances[idx]:.4f}')

    # How many correct boundaries does ML find that rules miss?
    print('\n--- Potential improvement analysis ---')
    # For test set, find anchors where ML says "boundary" but rules didn't select them
    rule_selected = set()
    for acc, aid, item in meta_test:
        if item is not None:
            rule_selected.add((acc, aid))

    ml_finds_new = 0
    ml_correct_new = 0
    for i, (acc, aid, item) in enumerate(meta_test):
        if y_prob[i] > 0.5 and (acc, aid) not in rule_selected:
            ml_finds_new += 1
            if y_test[i] == 1:
                ml_correct_new += 1

    print(f'ML identifies {ml_finds_new} additional boundary candidates')
    print(f'  Of which {ml_correct_new} are actually correct GT boundaries')


if __name__ == '__main__':
    main()
