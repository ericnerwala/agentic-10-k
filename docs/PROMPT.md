# Agentic 10-K Itemization

## Task

Build an LLM-orchestrated pipeline in a Google Colab notebook that extracts the ~20 standardized item sections (Item 1 through Item 16 + Signatures) from SEC Form 10-K filings.

**Input**: Raw SEC EDGAR `.txt` submission files containing embedded HTML. The dataset is 3 zip files (~225 MB total) with 501 filings and corresponding ground truth JSON labels. After excluding empty annotations, 392 filings are evaluated.

**Output**: JSON dict mapping `{accession}#{item_name}` keys to verbatim HTML slice values from the source filing.

## Dataset Structure

```
folder_1/*.txt, folder_2/*.txt, folder_3/*.txt     — raw SEC submission files
ground_truth_1/*.json, ground_truth_2/*.json, ground_truth_3/*.json  — label files
```

Each ground truth JSON maps keys like `0001613103-20-000021#item7a` to the exact HTML substring for that item section from the source filing. The HTML slice starts at the item's anchor element and ends just before the next item's anchor element.

## Standard 10-K Items

item1 (Business), item1a (Risk Factors), item1b (Unresolved Staff Comments), item2 (Properties), item3 (Legal Proceedings), item4 (Mine Safety), item5 (Market), item6 (Selected Financial Data/Reserved), item7 (MD&A), item7a (Market Risk), item8 (Financial Statements), item9 (Accountant Disagreements), item9a (Controls), item9b (Other Info), item9c (Foreign Jurisdiction), item10-14 (Directors/Comp/Ownership/Relationships/Accountant Fees), item15 (Exhibits), item16 (10-K Summary), signatures.

## What to Build

A Colab notebook that:

1. **Mounts Google Drive** and unzips the 10-K data
2. **Parses each filing** to extract the 10-K HTML document from the SEC submission wrapper
3. **Uses an LLM (via API)** to identify and classify item section boundaries — the LLM reads TOC anchors and surrounding text context, then classifies each anchor as an item type or "none" and resolves ordering conflicts
4. **Slices the HTML** at the identified boundaries to produce the output JSON
5. **Evaluates** using character-level F1 and document-level retrieval rate (a doc is "retrieved" only when every GT item is present, no false positives exist, and all item F1 scores >= 0.9)

The notebook should be self-contained — no dependency on any existing codebase files. Write all parsing, extraction, evaluation, and LLM orchestration code from scratch.

## Evaluation Metrics

- **Character-level F1**: Strip HTML tags, decode entities, compute bag-of-character precision/recall/F1 per item, average per filing, then macro-average across filings
- **Document-level retrieval rate**: Fraction of filings where ALL items are present, no false positives, and every item F1 >= 0.9

## Baseline to Beat

A rule-based pipeline (regex + dynamic programming, no LLM) achieves:
- Mean character-level F1: **97.5%**
- Document-level retrieval rate: **77.0%** (302/392)

## Constraints

- Self-contained Colab notebook, starts from raw zip files only
- LLM calls at temperature 0 with structured JSON output
- Batch all anchors per filing into a single LLM call for cost efficiency
- Total LLM cost < $20 for full 392-filing evaluation run
