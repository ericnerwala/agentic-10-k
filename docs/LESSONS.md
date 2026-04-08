# Agentic Pipeline — What Doesn't Work (And Why)

Lessons from 6 iterations of an agentic 10-K itemization pipeline. Read this before making changes.

---

## The Task

Extract ~23 item sections from SEC 10-K HTML filings. The agent calls tools against a precomputed structural index to discover item boundaries, assign them, validate, and self-correct. Primary metric: Document Retrieval Rate (all-or-nothing per filing).

**Best result:** 265/391 (67.8% raw, 73.9% adjusted for GT noise) with deepseek/deepseek-v3.2.

---

## What Does NOT Work

### 1. Making the prompt longer or more detailed

**What we tried:** Added a "Common Pitfalls" section to the system prompt warning about false positives, boundary precision, small items (item9b/9c), item7a vs item7 confusion, large spans, and incorporated-by-reference detection.

**Result:** DR unchanged (74% -> 74%). F1 improved (+3.6%) but token usage doubled (50K -> 120K avg). Two filings fixed, two regressed. Net zero on the metric that matters.

**Why it fails:** The agent overthinks. More instructions create more decision branches. On easy filings (70% of cases), the agent was already getting them right — the extra instructions just added tokens and sometimes caused it to second-guess correct assignments. The failures are not reasoning failures — they're input quality failures.

**Rule:** Do not add prompt instructions to fix specific failure cases. The prompt should be minimal and directive. 2450 chars is the sweet spot.

---

### 2. Adding validation checks that block finalization

**What we tried:** Added `tiny_span` (<50 chars) and `huge_span` (>50% for non-item8) checks to `validate_assignments`. Made them block the `finalize` gate.

**Result:** DR dropped. Agent spent 30 turns trying to satisfy span checks on filings that were otherwise correct, eventually hitting max turns with worse assignments than it started with.

**What partially worked:** Making span checks warnings (not blockers) stopped the regression but didn't improve anything either.

**Why it fails:** The finalize gate is binary — pass or fail. Adding checks that sometimes fire on correct assignments traps the agent in an unwinnable loop. The agent can't distinguish "this span is suspicious but actually correct" from "this span is wrong and needs fixing."

**Rule:** Only add validation checks that are unambiguously errors (duplicates, order violations, missing expected items). Never block finalization on heuristic checks.

---

### 3. Pre-populating assignments from a deterministic solver

**What we tried:** Ran a DP (dynamic programming) sequence-constrained solver first (~200ms), pre-loaded its assignments into the agent's state, then told the agent to "validate and fix" instead of discovering from scratch.

**Result:** DR dropped from 74% to 70%. Agent second-guessed correct DP assignments, moved boundaries to worse positions, and spent more turns than discovering from scratch.

**Why it fails:** The agent's strength is exploration, not verification. When given someone else's work to check, it looks for problems even when there aren't any. The validate-and-fix paradigm fights the agent's natural behavior — it wants to discover, not audit. The DP solver's assignments are globally optimal under sequence constraints; the agent's local edits break that global optimality.

**Rule:** Let the agent discover from scratch. Don't pre-populate state. Don't tell it to verify existing work.

---

### 4. Targeted single-shot LLM calls instead of the agent loop

**What we tried:** Replaced the multi-turn agent loop with: (1) run DP solver, (2) identify ambiguous items, (3) make one LLM call per ambiguous item with candidates and context. No loop, no tools.

**Result:** DR dropped to 68%. LLM fixed 2 filings but broke 5. Used 322x fewer tokens but worse accuracy.

**Why it fails:** The LLM needs the iterative feedback loop. A single call with candidates and context is not enough — the agent needs to assign, validate, see what's wrong, investigate with read_text_at, try again. The think-act-observe cycle genuinely adds value over one-shot reasoning. Removing the loop removes the self-correction capability.

**Rule:** Keep the multi-turn agent loop. The iterative cycle is the core value of the agentic approach. Don't replace it with one-shot calls.

---

### 5. Adding new tools to fix specific failure items

**What we tried:** Added `scan_for_heading` tool to find "Item X" headings between two positions. Specifically targeted item6/7/7a boundary confusion and item9b detection. Also added aggressive stall cutoff (force finalize at 20 turns + 3 stalls).

**Result:** DR dropped from 76% to 72%. Three new regressions, one improvement.

**Why it fails:** The agent used `scan_for_heading` on filings that were already correct, found matches it shouldn't have acted on, and moved correct boundaries. Every new tool is a new decision the agent must make on every filing. A tool that helps on 2 hard filings but gets misused on 3 easy filings is net-negative.

**Rule:** Be extremely cautious adding tools. Each tool adds decision complexity across ALL filings, not just the ones you're targeting. Prefer improving the index (what the agent sees) over adding tools (what the agent can do).

---

### 6. Post-processing that changes agent output

**What we tried:** Added three post-processing steps to the output slicer:
- Signatures suppression (remove signatures if not in TOC)
- Placeholder item16 normalization (replace "None"/"N/A" item16 with empty string)
- Part III empty-key emission (add empty placeholders for incorporated items)

**Result:** Signatures suppression helped on 2-3 filings. Item16 normalization caused 3 regressions (it replaced item16 content that GT actually expected). Part III emission was neutral.

**Why item16 failed:** The normalization deleted short item16 slices (573 chars) where the GT actually contained the "None" placeholder text. The archive baseline only normalizes when item16 > 10KB, but we applied it unconditionally.

**Rule:** Post-processing must be very conservative. Only apply transformations with overwhelming evidence. When in doubt, keep the agent's output as-is. Test post-processing changes against the specific GT format.

---

## What DOES Work

### 1. Improving candidate quality in the structural index

**What we did:** Added Part-region scoring — adjusted candidate confidence based on whether the candidate falls in the expected Part region of the document (+1 in correct Part, -2 in wrong Part).

**Result:** DR improved from 74% to 76% on 50 filings. The only change across 6 iterations that improved DR.

**Why it works:** It changes what the agent sees, not what the agent is told. Better candidates mean the agent's default behavior (pick highest confidence) is more likely to be correct. The improvement happens before the agent even starts thinking.

**Principle:** The agent's performance is bounded by its inputs. Improve the structural index, not the prompt.

### 2. Simple, directive system prompt

The best-performing prompt is 2450 chars:
- Step 1: get_filing_overview
- Step 2: get_all_top_candidates
- Step 3: batch_assign high-confidence items
- Step 4: validate_assignments
- Step 5: fix issues
- Step 6: finalize

No "Common Pitfalls." No item-specific guidance. No examples. The agent figures out edge cases through the tool feedback loop.

### 3. The tool-calling loop itself

The think-act-observe cycle is genuinely valuable:
- The agent discovers items using `get_all_top_candidates`
- Validates with `validate_assignments`
- Sees specific issues ("item9b missing", "order violation between item6 and item7")
- Investigates with `read_text_at` and `get_item_candidates`
- Self-corrects and re-validates

This cycle catches errors that one-shot approaches miss. The validation tool acts as a ground truth proxy that guides the agent toward correct assignments.

### 4. Parallel filing processing

Running filings concurrently with `asyncio.gather` + `Semaphore(N)`:
- Concurrency=5: ~5x speedup
- Concurrency=8: ~6x speedup
- Each filing is independent — no shared state

### 5. NoneType bug fix in state.py

`get_assignments()` and `to_boundaries()` sort by `char_position`, but incorporated-by-reference items have `char_position=None`. Sorting `None < int` crashes Python 3. Fixed with `key=lambda a: (a.char_position if a.char_position is not None else -1)`.

This eliminated 3 crashes per 100 filings. Always handle None in sort keys.

---

## Remaining Failure Patterns (for future work)

These are the real failures to fix. All attempts so far to address them via prompt/tool changes have failed or regressed. The next approach should focus on improving the structural index.

| Pattern | Count | Root Cause |
|---------|-------|------------|
| item9b missed/wrong | 28 | Short section, low-confidence candidates |
| item7a missed/wrong | 23 | Shares TOC anchor with item7 |
| item6/item7 boundary swap | 20 | Adjacent items with very different sizes |
| item16 annotation noise | 19 | GT inconsistency, not fixable |
| item4 missed/wrong | 18 | Short section in Part I |
| item14 missing | 13 | Part III, often incorporated |
| False positives | ~20 | Agent assigns items GT doesn't have |
| Hard filings (F1 < 75%) | ~10 | Structurally unusual filings, agent spirals to 30 turns |

**The right lever for each:**
- item9b, item7a: Improve candidate detection in `index.py` (heading scan during index build, not as agent tool)
- item6/item7: Boundary swap detection in index builder
- item14: Better Part III incorporation detection
- False positives: Post-processing removal of tiny spans (< 0.1% of doc), but test carefully
- Hard filings: "Best-so-far" checkpoint in the agent loop — save best state, revert when things worsen

---

## Anti-Patterns Summary

| Temptation | Why It Fails | Do This Instead |
|-----------|-------------|-----------------|
| Add prompt instructions for failure X | Agent overthinks on all filings | Fix candidate scoring in index.py |
| Add validation check that blocks finalize | Agent gets trapped in unwinnable loops | Make it a warning, or fix the index |
| Pre-populate agent state | Agent second-guesses, breaks global optimality | Let agent discover from scratch |
| Replace loop with one-shot LLM | Loses self-correction capability | Keep the multi-turn loop |
| Add a new tool for failure X | Agent misuses it on correct filings | Improve the index instead |
| Post-process agent output aggressively | Deletes correct output, GT format surprises | Be conservative, test against GT |
| Make the prompt longer | Doubles tokens, zero DR improvement | Keep it under 2500 chars |
| Extend max turns beyond 30 | Agent spirals, makes things worse | Implement best-so-far checkpoint |
