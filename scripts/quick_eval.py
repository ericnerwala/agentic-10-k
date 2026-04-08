"""Quick evaluation: extract + evaluate all sets, report doc retrieval rates."""
import json, sys, time
from pathlib import Path
from collections import defaultdict
from evaluate import evaluate_pair
from extract import process_file_extended

def run_set(set_num):
    folder = Path(f"data/folder_{set_num}/folder_{set_num}")
    gt_dir = Path(f"data/ground_truth_{set_num}/ground_truth_{set_num}")
    gt_files = sorted(gt_dir.glob("*.json"))
    txt_files = {f.stem: f for f in folder.glob("*.txt")}
    n_pass = n_total = 0
    for gt_path in gt_files:
        accession = gt_path.stem
        try:
            with open(gt_path, 'r', encoding='utf-8') as f:
                truth_dict = json.load(f)
        except: continue
        if not truth_dict: continue
        txt_path = txt_files.get(accession)
        if not txt_path: continue
        pred_dict = process_file_extended(str(txt_path))
        result = evaluate_pair(pred_dict, truth_dict, accession)
        n_total += 1
        if result['doc_retrieved']: n_pass += 1
    print(f"  Set {set_num}: {n_pass}/{n_total} = {n_pass/n_total*100:.1f}%")
    return n_pass, n_total

if __name__ == '__main__':
    t0 = time.time()
    total_r = total_e = 0
    for s in [1, 2, 3]:
        n, t = run_set(s)
        total_r += n; total_e += t
    print(f"OVERALL: {total_r}/{total_e} = {100*total_r/total_e:.1f}% ({time.time()-t0:.0f}s)")
