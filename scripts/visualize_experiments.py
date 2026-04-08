"""
Visualize baseline agentic pipeline performance on 391 10-K filings.

Figures:
  1. Performance overview (DR rate, F1 distribution)
  2. Failure analysis (which items fail, failure modes)
  3. Cost & latency distributions
"""

import json
import sys
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import numpy as np

# ---------------------------------------------------------------------------
# Skill helpers
# ---------------------------------------------------------------------------

SKILL_DIR = Path.home() / ".claude" / "skills" / "scientific-visualization"
sys.path.insert(0, str(SKILL_DIR / "scripts"))
sys.path.insert(0, str(SKILL_DIR / "assets"))

from style_presets import apply_publication_style
from color_palettes import OKABE_ITO_LIST
from figure_export import save_publication_figure

apply_publication_style("default")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "experiments" / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

C_BLUE    = OKABE_ITO_LIST[4]  # #0072B2
C_ORANGE  = OKABE_ITO_LIST[0]  # #E69F00
C_GREEN   = OKABE_ITO_LIST[2]  # #009E73
C_RED     = OKABE_ITO_LIST[5]  # #D55E00
C_PURPLE  = OKABE_ITO_LIST[6]  # #CC79A7
C_BLACK   = OKABE_ITO_LIST[7]  # #000000

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

with open(PROJECT_DIR / "experiments" / "baseline_deepseek_391" / "results.json") as f:
    data = json.load(f)

N = len(data)
f1s = np.array([r["f1"] for r in data])
turns = np.array([r["turns"] for r in data])
tokens = np.array([r["tokens"] for r in data])
latency = np.array([r["latency_ms"] for r in data]) / 1000  # seconds
dr = np.array([r["doc_retrieved"] for r in data])
finalized = np.array([r["finalized"] for r in data])
candidates = np.array([r["candidates"] for r in data])

dr_count = int(dr.sum())
dr_rate = dr_count / N

print(f"Loaded {N} filings")
print(f"DR: {dr_count}/{N} ({dr_rate:.1%})")
print(f"Mean F1: {f1s.mean():.3f}")


# =========================================================================
# Figure 1: Performance Overview (2 panels)
# =========================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8))

# A — F1 distribution histogram
ax1.hist(f1s * 100, bins=30, color=C_BLUE, edgecolor="white",
         linewidth=0.5, alpha=0.85)
ax1.axvline(x=90, color=C_RED, linestyle="--", linewidth=1,
            label="DR threshold (90%)")
ax1.axvline(x=f1s.mean() * 100, color=C_BLACK, linestyle="-",
            linewidth=1, label=f"Mean ({f1s.mean()*100:.1f}%)")
ax1.set_xlabel("Character F1 (%)")
ax1.set_ylabel("Number of filings")
ax1.legend(fontsize=6, loc="upper left")
ax1.text(-0.15, 1.08, "A", transform=ax1.transAxes,
         fontsize=10, fontweight="bold", va="top")

# B — F1 per filing sorted (waterfall view)
sorted_f1 = np.sort(f1s * 100)
x_pos = np.arange(N)
colors_bar = [C_GREEN if v >= 90 else C_RED for v in sorted_f1]
ax2.bar(x_pos, sorted_f1, width=1.0, color=colors_bar, edgecolor="none")
ax2.axhline(y=90, color=C_BLACK, linestyle="--", linewidth=0.6)
ax2.set_xlabel("Filings (sorted by F1)")
ax2.set_ylabel("Character F1 (%)")
ax2.set_ylim(0, 105)
ax2.text(N * 0.02, 92, f"{dr_count}/{N} pass ({dr_rate:.1%})",
         fontsize=7, color=C_BLACK)
ax2.text(-0.15, 1.08, "B", transform=ax2.transAxes,
         fontsize=10, fontweight="bold", va="top")

save_publication_figure(fig, OUTPUT_DIR / "01_performance_overview",
                        formats=["pdf", "png"], dpi=300)
plt.close(fig)


# =========================================================================
# Figure 2: Failure Analysis (3 panels)
# =========================================================================

# Collect failure data
failed_docs = [r for r in data if not r["doc_retrieved"]]
item_fail_count = Counter()
item_fail_type = Counter()  # (item, type) -> count

for r in data:
    for item, m in r.get("failed_items", {}).items():
        item_fail_count[item] += 1
        f1_val = m["f1"]
        pred_len = m["pred_len"]
        truth_len = m["truth_len"]
        if pred_len == 0:
            item_fail_type[(item, "missing")] += 1
        elif truth_len > 1_000_000:
            item_fail_type[(item, "GT noise")] += 1
        elif f1_val < 0.5:
            item_fail_type[(item, "severe")] += 1
        else:
            item_fail_type[(item, "near-miss")] += 1

    for fp_item in r.get("false_positives", []):
        item_fail_count[fp_item] += 1
        item_fail_type[(fp_item, "false positive")] += 1

# Top failure items
top_items = [item for item, _ in item_fail_count.most_common(12)]
fail_types = ["GT noise", "severe", "missing", "near-miss", "false positive"]
fail_colors = [C_PURPLE, C_RED, C_ORANGE, OKABE_ITO_LIST[1], "#BBBBBB"]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.2, 3.0),
                                      gridspec_kw={"width_ratios": [2.5, 1.2, 1.5]})

# A — Stacked bar: failure types per item
x_pos = np.arange(len(top_items))
bottoms = np.zeros(len(top_items))

for ftype, fcolor in zip(fail_types, fail_colors):
    counts = [item_fail_type.get((item, ftype), 0) for item in top_items]
    ax1.bar(x_pos, counts, bottom=bottoms, width=0.7,
            color=fcolor, edgecolor="white", linewidth=0.3, label=ftype)
    bottoms += counts

ax1.set_xticks(x_pos)
ax1.set_xticklabels(top_items, rotation=45, ha="right", fontsize=6)
ax1.set_ylabel("Filings affected")
ax1.set_xlabel("Item")
ax1.legend(fontsize=5.5, loc="upper right", ncol=2)
ax1.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax1.text(-0.12, 1.08, "A", transform=ax1.transAxes,
         fontsize=10, fontweight="bold", va="top")

# B — Pie: DR pass vs fail breakdown
not_finalized = sum(1 for r in failed_docs if not r["finalized"])
finalized_fail = len(failed_docs) - not_finalized
sizes = [dr_count, finalized_fail, not_finalized]
labels_pie = [f"Pass\n({dr_count})", f"Fail-FIN\n({finalized_fail})",
              f"Fail-STALL\n({not_finalized})"]
pie_colors = [C_GREEN, C_ORANGE, C_RED]

wedges, texts = ax2.pie(sizes, colors=pie_colors, startangle=90,
                         wedgeprops=dict(edgecolor="white", linewidth=1.5))
ax2.legend(labels_pie, fontsize=5.5, loc="center",
           bbox_to_anchor=(0.5, -0.15), ncol=3)
ax2.text(-0.05, 1.08, "B", transform=ax2.transAxes,
         fontsize=10, fontweight="bold", va="top")

# C — Scatter: candidates vs F1 (structural signal strength)
ax3.scatter(candidates[dr], f1s[dr] * 100, s=10, alpha=0.5,
            color=C_GREEN, edgecolors="none", label="DR=YES", zorder=2)
ax3.scatter(candidates[~dr], f1s[~dr] * 100, s=14, alpha=0.7,
            color=C_RED, marker="x", linewidths=0.8,
            label="DR=NO", zorder=3)
ax3.set_xlabel("Structural candidates")
ax3.set_ylabel("Character F1 (%)")
ax3.legend(fontsize=6, loc="lower right")
ax3.text(-0.15, 1.08, "C", transform=ax3.transAxes,
         fontsize=10, fontweight="bold", va="top")

save_publication_figure(fig, OUTPUT_DIR / "02_failure_analysis",
                        formats=["pdf", "png"], dpi=300)
plt.close(fig)


# =========================================================================
# Figure 3: Cost, Latency, and Efficiency (3 panels)
# =========================================================================

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.2, 2.8))

# A — Tokens distribution (cost proxy)
ax1.hist(tokens / 1000, bins=40, color=C_BLUE, edgecolor="white",
         linewidth=0.3, alpha=0.85)
ax1.axvline(x=np.median(tokens) / 1000, color=C_BLACK, linestyle="-",
            linewidth=1, label=f"Median ({np.median(tokens)/1000:.0f}K)")
ax1.set_xlabel("Tokens per filing (K)")
ax1.set_ylabel("Number of filings")
ax1.legend(fontsize=6)
ax1.text(-0.15, 1.08, "A", transform=ax1.transAxes,
         fontsize=10, fontweight="bold", va="top")

# B — Latency distribution
ax2.hist(latency, bins=40, color=C_ORANGE, edgecolor="white",
         linewidth=0.3, alpha=0.85)
ax2.axvline(x=np.median(latency), color=C_BLACK, linestyle="-",
            linewidth=1, label=f"Median ({np.median(latency):.0f}s)")
ax2.set_xlabel("Latency per filing (seconds)")
ax2.set_ylabel("Number of filings")
ax2.legend(fontsize=6)
ax2.text(-0.15, 1.08, "B", transform=ax2.transAxes,
         fontsize=10, fontweight="bold", va="top")

# C — Turns vs F1, colored by DR
ax3.scatter(turns[dr], f1s[dr] * 100, s=10, alpha=0.5,
            color=C_GREEN, edgecolors="none", label="DR=YES", zorder=2)
ax3.scatter(turns[~dr], f1s[~dr] * 100, s=14, alpha=0.7,
            color=C_RED, marker="x", linewidths=0.8,
            label="DR=NO", zorder=3)
ax3.set_xlabel("Agent turns")
ax3.set_ylabel("Character F1 (%)")
ax3.legend(fontsize=6, loc="lower left")
ax3.text(-0.15, 1.08, "C", transform=ax3.transAxes,
         fontsize=10, fontweight="bold", va="top")

save_publication_figure(fig, OUTPUT_DIR / "03_cost_latency_efficiency",
                        formats=["pdf", "png"], dpi=300)
plt.close(fig)


print(f"\nAll figures saved to {OUTPUT_DIR}/")
