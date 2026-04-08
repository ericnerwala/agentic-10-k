import json, re, sys
from collections import Counter
from extract import process_file

def strip_html(h):
    t = re.sub(r'<[^>]+>', ' ', h)
    return re.sub(r'\s+', ' ', t).strip()

def char_f1(p, t):
    if not t and not p: return 1.0
    if not t or not p: return 0.0
    tc = Counter(t); pc = Counter(p)
    overlap = sum(min(tc[c], pc[c]) for c in pc)
    prec = overlap/len(p); rec = overlap/len(t)
    return 2*prec*rec/(prec+rec) if prec+rec else 0

txt_file = sys.argv[1] if len(sys.argv) > 1 else 'data/folder_1/folder_1/0000004281-20-000038.txt'
truth_file = sys.argv[2] if len(sys.argv) > 2 else 'data/ground_truth_1/ground_truth_1/0000004281-20-000038.json'

pred = process_file(txt_file)
with open(truth_file, encoding='utf-8') as f:
    truth = json.load(f)

print('Ground truth items:', sorted(k.split('#')[1] for k in truth.keys()))
print('Predicted items:   ', sorted(k.split('#')[1] for k in pred.keys()))
print()

f1s = []
for key in sorted(truth.keys()):
    item = key.split('#')[1]
    t_html = truth[key]
    p_html = pred.get(key, '')
    t_text = strip_html(t_html)
    p_text = strip_html(p_html)
    f1 = char_f1(p_text, t_text)
    f1s.append(f1)
    status = 'OK' if f1 >= 0.9 else ('PARTIAL' if f1 > 0 else 'MISS')
    print(f'  {item:<12} F1={f1:.3f} [{status}]  t_len={len(t_text):>8}  p_len={len(p_text):>8}')

print(f'\nMean F1: {sum(f1s)/len(f1s):.4f}')
