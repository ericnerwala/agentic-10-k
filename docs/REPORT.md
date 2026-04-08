# Agentic 10-K Itemization Pipeline — Technical Report

**Date:** 2026-04-08
**Model:** deepseek/deepseek-v3.2 (via OpenRouter)
**Dataset:** 391 SEC 10-K filings with ground truth annotations

---

## Executive Summary

An agentic LLM pipeline that extracts ~23 standard item sections from SEC 10-K HTML filings using a think-act-observe loop. The agent calls tools against a precomputed structural index to discover, assign, validate, and refine item boundaries.

### Key Results (391 filings)

| Metric | Value |
|--------|-------|
| Raw Document Retrieval | 265/391 (67.8%) |
| Adjusted Document Retrieval | 289/391 (73.9%) |
| Mean Character F1 | 96.6% |
| Errors/Crashes | 0 |
| Avg Turns per Filing | 15.6 |
| Avg Tokens per Filing | 158,379 |
| Total Tokens | 61.9M |

Document Retrieval is all-or-nothing: all GT items must be present, no false positives, and every item must have char F1 >= 0.90. The "adjusted" rate excludes 24 filings that fail solely due to ground truth annotation noise (signatures GT of 14M-86M chars, item16 GT of multi-million chars).

---

## Architecture

### Overview

```
Filing (.txt) --> Structural Index --> Agent Loop --> Assignments --> HTML Slicer --> Predictions
                    (~200ms)         (10-30 turns)                     (instant)
```

### Components

#### 1. Structural Index (`pipeline/agent/index.py`)

Precomputes all structural signals from the filing HTML in ~200ms, with no LLM calls:
- **TOC links**: Parses `<a href="#...">` links and classifies them by item name
- **Anchor positions**: Locates all `id=` and `name=` attributes in the HTML
- **Item candidates**: Scores each anchor as a potential item boundary (confidence 0-10)
- **Part boundaries**: Detects Part I/II/III/IV heading positions
- **Part III incorporation**: Checks for "incorporated by reference" language

**Candidate scoring tiers:**

| Confidence | Source | Description |
|-----------|--------|-------------|
| 9 | anchor_id | Anchor ID matches regex (e.g., `id="item1a"`) |
| 8 | toc_text | TOC link text classified to this item |
| 6 | tier1 (near) | "Item X" regex in first 150 chars after anchor |
| 5 | tier1 (far) | "Item X" regex in full 2000-char lookahead |
| 3 | tier2 | Descriptive title keywords (e.g., "Risk Factors") |
| 1 | title_keyword | Keywords in broader 1000-char context |

**Part-region scoring:** After initial scoring, candidates are adjusted based on whether they fall in the expected Part region of the document. Candidates in the correct Part get +1 confidence; candidates in the wrong Part get -2. This prevents the agent from selecting candidates that are positionally implausible (e.g., a Part II item in the Part IV region). High-confidence matches (>=9) are not adjusted.

#### 2. Agent Loop (`pipeline/agent/loop.py`)

Think-act-observe-repeat loop inspired by Claude Code's architecture:
- Single-threaded, flat message history
- Model-agnostic via OpenRouter (AsyncOpenAI client)
- Temperature 0, max_tokens 4096 per turn
- Maximum 30 turns per filing
- Stall detection: after 3 consecutive turns without improvement, nudges agent to move on
- Message compaction: every 10 turns, if message history exceeds 100K chars, older messages are summarized
- Supports native function calling and JSON-mode fallback

#### 3. Tool Registry (`pipeline/agent/tools.py`)

17 tools organized by function:

**Discovery (5 tools):**
- `get_filing_overview` — Document structure summary
- `get_toc_links` — All TOC entries with classifications
- `get_item_candidates` — All candidates for a specific item
- `get_all_top_candidates` — Top 3 candidates per item in one call
- `get_part_boundaries` — Part I/II/III/IV positions

**Classification (2 tools):**
- `classify_text` — Run regex classification on arbitrary text
- `detect_incorporation` — Check Part III incorporation by reference

**Assignment (4 tools):**
- `assign_item` — Assign one item to an anchor
- `unassign_item` — Remove an assignment
- `batch_assign` — Assign multiple items at once
- `get_current_assignments` — View all current assignments

**Validation (2 tools):**
- `validate_assignments` — Check canonical order, monotonic positions, expected items
- `check_span_sizes` — Flag suspiciously large or small spans

**Refinement (2 tools):**
- `refine_boundary` — Search nearby anchors around a position
- `scan_for_heading` — Search for "Item X" heading text between two positions

**Output (1 tool):**
- `finalize` — Gate: only succeeds if validation passes

#### 4. System Prompt (`pipeline/agent/prompts.py`)

Minimal, directive prompt (~2450 chars):
1. Bulk discovery with `get_all_top_candidates`
2. Batch assignment of high-confidence items (confidence >= 7)
3. Validate and fix loop
4. Finalize when clean

Key rules: items must appear in ascending order, Part III items are often incorporated by reference, signatures should only be assigned with strong evidence, conservative assignments preferred over false positives.

#### 5. Assignment State (`pipeline/agent/state.py`)

Immutable data types with append-only event log:
- `ItemAssignment` — frozen dataclass (item_name, anchor_id, char_position, reasoning)
- `AssignmentState` — manages assignments with full event history
- Supports `None` char_position for incorporated-by-reference items

#### 6. Validation (`pipeline/agent/validation.py`)

Checks run on every `validate_assignments` call:
1. No duplicate item names
2. Positions within valid range
3. Canonical item order (ITEM_SEQ_ORDER)
4. Monotonically increasing positions
5. Expected items present (item1, item1a, item7, item8, item15)

The finalize gate requires zero validation issues.

#### 7. HTML Slicer (`pipeline/agent/runner.py`)

Converts assignments to GT-compatible output:
- Slices HTML between consecutive boundary positions
- Deduplicates by item name (keeps first)
- Enforces ascending positions
- Emits empty strings for incorporated-by-reference items

---

## Agent Behavior Pattern

Typical successful filing (9-12 turns):

```
Turn 1: get_filing_overview → orient
Turn 2: get_all_top_candidates → see all candidates
Turn 3: batch_assign (15-22 items) → bulk assignment
Turn 4: validate_assignments → check work
Turn 5-8: Fix issues (get_item_candidates, read_text_at, assign_item)
Turn 9: validate_assignments → clean
Turn 10: finalize → done
```

Typical hard filing (25-30 turns):
- Multiple rounds of assign/validate/fix
- Agent investigates ambiguous items with read_text_at
- Stall detection triggers nudge messages
- May hit max turns without finalizing (BEST-EFFORT output)

---

## Performance Analysis

### By Difficulty

| Difficulty | Count | DR Rate | Avg Turns | Avg Tokens |
|-----------|-------|---------|-----------|------------|
| Easy (>10 bold items) | ~60% | ~80% | 10-12 | 60-80K |
| Medium (3-10 bold) | ~25% | ~65% | 15-20 | 120-180K |
| Hard (<3 bold) | ~15% | ~40% | 25-30 | 250-350K |

### Failure Breakdown (126 failures out of 391)

| Category | Count | Description |
|----------|-------|-------------|
| GT noise only | 24 | Signatures/item16 GT annotation errors |
| Mixed (noise + real) | 14 | GT noise plus genuine pipeline failures |
| Real failures | 88 | Genuine pipeline errors |
| Crashes/bugs | 0 | All fixed |

### Top Failure Items

**Boundary failures (wrong position, F1 < 0.9):**

| Item | Severe (F1<0.5) | Near-miss (0.5-0.9) | Missing | FP | Total |
|------|-----------------|---------------------|---------|-----|-------|
| item9b | 10 | 6 | 12 | 0 | 28 |
| item7a | 10 | 0 | 11 | 2 | 23 |
| item6 | 10 | 0 | 7 | 3 | 20 |
| item16 | 13 | 0 | 6 | 0 | 19 |
| item4 | 5 | 5 | 5 | 3 | 18 |
| item7 | 5 | 0 | 5 | 4 | 14 |
| item14 | 0 | 0 | 10 | 3 | 13 |
| item5 | 4 | 6 | 3 | 7 | 20 |
| item1 | 0 | 8 | 0 | 3 | 11 |

### Root Cause Analysis

**1. item9b (28 occurrences) — #1 failure item**
A very short section ("Other Information") between item9a and item10. The agent misses it entirely (12x) or places the boundary wrong (16x). Root cause: low-confidence candidates and the section is often just 1-2 paragraphs.

**2. item7a (23 occurrences) — shared anchor problem**
"Quantitative and Qualitative Disclosures About Market Risk" immediately follows item7 (MD&A). In many filings, item7 and item7a share the same TOC anchor, making them indistinguishable to the index. The agent defaults to the shared position, which is item7's start, not item7a's.

**3. item6/item7 boundary confusion (20 occurrences)**
"Selected Financial Data" (item6) is short, "MD&A" (item7) is long. When the boundary is placed wrong, item6 absorbs item7's content or vice versa. Adjacent items with very different sizes are vulnerable to off-by-one anchor selection.

**4. item16 (19 occurrences)**
"Form 10-K Summary" — most filings just say "None" or "Not applicable." GT annotation is inconsistent: some GT includes the placeholder text, some omits item16 entirely, some includes everything to EOF. Unfixable without GT normalization.

**5. False positives (scattered)**
Agent assigns items that don't exist in GT. Most common on unusual filings (combined 10-K/proxy, abbreviated filings). The agent's conservative assignment prompt helps but doesn't eliminate FPs.

---

## Iteration History

Six iterations were tested during development. Each taught a specific lesson about what works and what doesn't in agentic pipelines.

### Iteration 1 (baseline) — 50 filings
- Raw DR: 37/50 (74%)
- F1: 93.9%
- Errors: 2 (NoneType crash on None char_position sort)
- Simple prompt, basic tool set, single-threaded

### Iteration 2 (richer prompt + tools) — 50 filings
- Raw DR: 37/50 (74%) — same
- F1: 97.5% — better
- Changes: "Common Pitfalls" prompt section, span warnings in batch_assign, top-3 candidates in get_all_top_candidates, refine_boundary tool
- Learning: Richer prompts improved F1 (boundary precision) but caused agent overthinking on easy filings. Two improvements cancelled by two regressions. Token usage doubled.

### Iteration 3 (softened validation) — 20 filings
- Raw DR: 15/20 (75%) — worse than iteration 1
- Changes: Span checks as warnings instead of blockers in validation
- Learning: Changing validation semantics confused the agent. It stopped treating span issues seriously.

### Iteration 4 (pre-populated assignments) — 50 filings
- Raw DR: 35/50 (70%) — worse
- Changes: Pre-populated assignments from a deterministic solver, agent validates and fixes
- Learning: Agent second-guessed correct pre-populated assignments. The agent performs better discovering from scratch than verifying someone else's work. The validate-and-fix paradigm is worse than discover-and-assign.

### Iteration 5 (targeted LLM calls) — 50 filings
- Raw DR: 34/50 (68%) — worse
- Changes: No agent loop. Deterministic solver for clean filings, single-shot LLM calls for ambiguous items only.
- Learning: LLM was net-negative even on targeted calls — fixed 2 filings, broke 5. 322x fewer tokens but worse accuracy. The multi-turn agent loop is genuinely better than one-shot LLM calls for this task.

### Final (Part-region scoring) — 50 filings then 391 filings
- 50 filings: Raw DR 38/50 (76%) — best on 50
- 391 filings: Raw DR 265/391 (67.8%), Adjusted 289/391 (73.9%)
- Changes: Candidate confidence adjusted by Part-region position (+1 in correct Part, -2 in wrong Part). NoneType bug fixed.
- Learning: Improving candidate quality (the input to the agent) is more effective than improving the prompt or tools. This was the only change that improved DR.

### Iteration 6 (heading scan + stall cutoff) — 50 filings
- Raw DR: 36/50 (72%) — regression from final
- Changes: scan_for_heading tool, aggressive stall cutoff forcing finalize at 20 turns
- Learning: New tools can cause regressions when the agent uses them on filings that were already correct. Adding tools is not free — each tool adds decision complexity for the agent. Reverted.

### Key Lessons

1. **Simple prompts outperform complex ones.** A 2450-char prompt consistently beats longer versions. Every additional instruction is an opportunity for the agent to overthink.

2. **Improve inputs, not instructions.** Part-region scoring was the only change that improved DR. It changed what the agent sees, not what the agent is told to do.

3. **The agent's strength is exploration, not verification.** The agent excels at discovering items from scratch using tools. Attempts to pre-populate or constrain it made things worse.

4. **New tools have hidden costs.** More tools means more decisions for the agent. A tool that helps on 2 filings but confuses the agent on 3 others is net-negative.

5. **GT noise is a hard ceiling.** ~6% of filings fail due to annotation errors that no pipeline can fix. The adjusted rate is the true measure.

6. **Multi-turn agent loops beat single-shot LLM calls.** The iterative discover-assign-validate-fix cycle genuinely adds value over one-shot approaches. The agent uses validation feedback to self-correct.

---

## Future Improvement Directions

### High Impact (estimated +5-10% DR each)

1. **Better item9b/item7a detection in the index.** These two items account for 51 of 126 failures. Improving candidate quality specifically for these items (e.g., heading-scan fallback in the index builder, not as an agent tool) would have the largest single impact.

2. **Boundary swap detection in the index.** Detect and correct item6/7/7a/8 swaps as pre-corrected candidates so the agent gets them right on the first try.

3. **Smarter stall handling.** 11 filings max out at 30 turns with worsening results. A "best-so-far" checkpoint system could save the best assignment state and revert when things get worse.

### Medium Impact (estimated +2-5% DR each)

4. **Part III incorporation detection improvements.** 10 filings miss item14 entirely. Better detection of "incorporated by reference" language would help.

5. **False positive prevention.** 7 filings fail solely due to FPs. A post-processing step that removes items with very small spans (<0.1% of doc) could help.

6. **Model improvements.** Testing with stronger models (Claude, GPT-4) may improve the agent's reasoning on hard filings, though at higher cost.

---

## File Structure

```
agentic-10-k/
  REPORT.md                  # This report
  run_all.py                 # Full 391-filing runner (parallel, concurrency=8)
  run_all_resume.py          # Resume interrupted runs (skips completed filings)
  run_test50.py              # 50-filing test runner
  run_test100.py             # 100-filing test runner
  .env                       # OpenRouter API key (gitignored)
  pipeline/
    __init__.py
    config.py                # Constants, paths, item ordering
    agent/
      __init__.py
      index.py               # Structural index builder + Part-region scoring
      loop.py                # Agent loop (think-act-observe)
      prompts.py             # System prompt + task prompt
      tools.py               # 17 tools
      state.py               # Assignment state management
      validation.py          # Assignment validation
      runner.py              # Batch runner + HTML slicer
      hybrid.py              # Experimental hybrid runner (not used)
  results_full/
    results.json             # Results on 391 filings
  results_50/
    results.json             # Results on 50 filings
    report.md
  results_100/
    results.json             # Results on 100 filings
    report.md
```

---

## Running

```bash
# Set API key
export OPENROUTER_API_KEY="sk-or-..."
# Or use .env file (auto-loaded via python-dotenv)

# Test on 50 filings
python run_test50.py

# Full 391 filings
python run_all.py

# Resume interrupted full run
python run_all_resume.py
```

Requires: `openai`, `python-dotenv`, and data files in `../10-K/data/`.
