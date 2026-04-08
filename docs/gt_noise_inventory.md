# Ground Truth Noise Inventory

Comprehensive audit of annotation inconsistencies in the 391-filing evaluation corpus.
These failures are attributable to ground truth (GT) errors, not pipeline errors.

## Summary

| Category | Item-Failures | Docs (noise-only) |
|----------|--------------|-------------------|
| Boundary signatures — source file mismatch | 21 | 13 |
| Boundary item16 — source file mismatch | 9 | 7 |
| Boundary item16 — short placeholder mismatch | 1 | 0 |
| FP item16 — GT inconsistently omits | 7 | 4 |
| FP signatures — GT inconsistently omits | 4 | 1 |
| **Total** | **42** | **24** |

Mixed documents (GT noise + real errors): **15**

## Adjusted Retrieval Rates

| Metric | Computation | Rate |
|--------|------------|------|
| Raw retrieval rate | 265 / 391 | 67.8% |
| **Noise-adjusted** (noise docs = pass) | 289 / 391 | **73.9%** |
| Clean GT only (exclude noise docs) | 265 / 352 | 75.3% |

## Noise Classification Criteria

### Boundary signatures — source file mismatch

GT value exceeds 1M characters (12–86 MB), indicating GT was generated from a different source file version.

### Boundary item16 — source file mismatch

GT value exceeds 1M characters; same pattern as signatures.

### Boundary item16 — short placeholder mismatch

Both GT and prediction are short placeholders (<500 chars) with minor HTML differences.

### FP item16 — GT inconsistently omits

Pipeline correctly extracts item16 (present in TOC), but GT annotation omits the key.

### FP signatures — GT inconsistently omits

Filing contains a clear SIGNATURES heading, but GT does not include the key.

## Noise-Only Documents (24)

These documents fail DR **solely** due to GT annotation errors. The pipeline extracts all items correctly.

| # | Accession | F1 | Noise Category | Item | GT Length |
|---|-----------|-----|----------------|------|----------|
| 1 | `0000002969-20-000049` | 95.5% | sig mismatch | signatures | 22,606,735 |
| 2 | `0000004904-20-000007` | 100.0% | FP item16 | item16 | — |
| 3 | `0000006951-20-000048` | 95.5% | item16 mismatch | item16 | 2,578,674 |
| 4 | `0000018230-20-000056` | 94.4% | item16 mismatch | item16 | 45,790,969 |
| 5 | `0000018926-20-000009` | 100.0% | FP item16 | item16 | — |
| 6 | `0000021076-20-000016` | 95.1% | sig mismatch | signatures | 19,169,037 |
| 7 | `0000029905-20-000011` | 100.0% | FP item16 | item16 | — |
| 8 | `0000093751-20-000684` | 95.5% | sig mismatch | signatures | 42,885,336 |
| 9 | `0000107815-20-000089` | 95.5% | sig mismatch | signatures | 31,420,149 |
| 10 | `0000713676-20-000042` | 95.5% | sig mismatch | signatures | 52,992,386 |
| 11 | `0000719739-20-000016` | 95.5% | sig mismatch | signatures | 39,095,971 |
| 12 | `0000788784-20-000004` | 95.2% | sig mismatch | signatures | 85,972,755 |
| 13 | `0000920522-20-000026` | 95.5% | item16 mismatch | item16 | 5,280,652 |
| 14 | `0001031296-20-000008` | 95.2% | item16 mismatch | item16 | 26,721,291 |
| 15 | `0001038357-20-000009` | 95.5% | sig mismatch | signatures | 14,369,163 |
| 16 | `0001065088-20-000006` | 90.9% | item16 mismatch | item16 | 2,194,454 |
| 16 | `0001065088-20-000006` | 90.9% | sig mismatch | signatures | 16,299,031 |
| 17 | `0001133421-20-000006` | 95.5% | sig mismatch | signatures | 17,184,723 |
| 18 | `0001467858-20-000028` | 95.3% | sig mismatch | signatures | 23,758,789 |
| 19 | `0001506307-20-000022` | 95.2% | sig mismatch | signatures | 32,998,265 |
| 20 | `0001558370-20-000581` | 95.2% | item16 mismatch | item16 | 35,078,988 |
| 21 | `0001564590-20-004475` | 100.0% | FP item16 | item16 | — |
| 22 | `0001564590-20-004992` | 100.0% | FP sig | signatures | — |
| 23 | `0001564590-20-005365` | 95.5% | sig mismatch | signatures | 54,519,596 |
| 24 | `0001628280-20-003693` | 95.2% | item16 mismatch | item16 | 51,676,123 |

## Mixed Documents (15)

These documents have **both** GT noise and real pipeline errors. They are NOT counted as passing in the adjusted rate.

| # | Accession | F1 | Noise Items | Real Errors |
|---|-----------|-----|-------------|-------------|
| 1 | `0000026172-20-000009` | 94.8% | signatures(boundary) | item9b(F1=0.88) |
| 2 | `0000027904-20-000004` | 85.0% | signatures(boundary) | item7(F1=0.01), item7a(F1=0.05), item8(F1=0.64) |
| 3 | `0000049196-20-000010` | 89.6% | item16(boundary) | item1b(F1=0.64), item5(F1=0.89), item6(F1=0.29) |
| 4 | `0000702165-20-000011` | 92.4% | signatures(boundary) | item9b(F1=0.52) |
| 5 | `0000721371-20-000089` | 78.1% | signatures(boundary) | crossReference(F1=0.00), item5(F1=0.18), item9a(FP) |
| 6 | `0000815097-20-000003` | 90.0% | item16(boundary) | item14(F1=0.00) |
| 7 | `0000827052-20-000026` | 22.8% | item16(boundary) | item1(F1=0.00), item14(F1=0.01), item15(F1=0.00), item1a(F1=0.00), item1b(F1=0.00), item2(F1=0.00), item3(F1=0.00), item4(F1=0.00), item5(F1=0.00), item6(F1=0.00), item7(F1=0.00), item7a(F1=0.00), item8(F1=0.00), item9(F1=0.00), item9a(F1=0.00), item9b(F1=0.00) |
| 8 | `0000895421-20-000265` | 67.3% | signatures(boundary) | item1b(F1=0.00), item2(F1=0.00), item3(F1=0.00), item4(F1=0.00), item7(F1=0.42), item7a(F1=0.00), item9a(F1=0.89), item9b(F1=0.54) |
| 9 | `0000899689-20-000007` | 94.8% | signatures(boundary) | item9b(F1=0.88) |
| 10 | `0001110803-20-000018` | 88.9% | signatures(fp) | item1(F1=0.86), item6(F1=0.50), item7a(F1=0.09), item8(FP), item5(FP), item2(FP) |
| 11 | `0001193125-20-043853` | 90.5% | signatures(boundary) | item8(F1=0.00) |
| 12 | `0001193125-20-048303` | 72.4% | item16(fp) | item4(F1=0.04), item9(F1=0.03), item1(FP), item9b(FP), item7a(FP), item5(FP), item13(FP), item9a(FP), item7(FP), item10(FP), item8(FP), item14(FP), item6(FP), item12(FP), item11(FP), item1b(FP) |
| 13 | `0001466258-20-000064` | 87.4% | signatures(boundary) | item15(F1=0.24), item16(F1=0.00) |
| 14 | `0001564590-20-005784` | 100.0% | item16(fp), signatures(fp) | item15(FP) |
| 15 | `0001739940-20-000006` | 98.9% | item16(fp), signatures(fp) | item1(F1=0.78) |
