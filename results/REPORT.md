# Retrieval arm sweep — results

Generated from `retrieval_raw.csv` by `runner/report.py`. **12 faults, 11 arms, 4 observation levels, 528 cells.**

Ground truth is the fault label: the function an edit must touch to fix the bug. `recall_fn@k` asks whether that exact function appears in the top k. CIs are percentile bootstrap over the fault population.


## 1. Observation level dominates retrieval method

Pooled over all ranking arms. This is the largest effect in the study.

| Level | what the query contains | MRR (fn) | 95% CI | recall_fn@1 | recall_fn@3 |
|---|---|---|---|---|---|
| **L0** | pass/fail only | 0.220 | [0.192, 0.251] | 0.03 | 0.20 |
| **L1** | + which tests failed | 0.552 | [0.500, 0.606] | 0.29 | 0.78 |
| **L2** | + assertion text | 0.608 | [0.551, 0.666] | 0.39 | 0.74 |
| **L3** | + full traceback | 0.718 | [0.660, 0.776] | 0.57 | 0.83 |

## 2a. Ranking arms at L2 (+ assertion text)

| Arm | MRR (fn) | 95% CI | recall_fn@1 | recall_fn@3 | recall_file@3 | median ms |
|---|---|---|---|---|---|---|
| `A7_cohere_embed` | 0.757 | [0.590, 0.917] | 0.58 | 0.92 | 1.00 | 1.8 |
| `A1_lexical` | 0.708 | [0.542, 0.875] | 0.50 | 0.83 | 1.00 | 1.5 |
| `A9_bm25_then_cohere_rerank` | 0.687 | [0.507, 0.861] | 0.50 | 0.92 | 1.00 | 7.1 |
| `A5_embed` | 0.669 | [0.478, 0.854] | 0.50 | 0.75 | 1.00 | 329.2 |
| `A8_cohere_rerank` | 0.653 | [0.472, 0.833] | 0.42 | 0.83 | 1.00 | 5676.6 |
| `A3_tfidf` | 0.649 | [0.446, 0.850] | 0.50 | 0.67 | 1.00 | 9.2 |
| `A6b_hybrid_lexonly` | 0.624 | [0.443, 0.808] | 0.42 | 0.75 | 1.00 | 5.6 |
| `A4_lsa` | 0.604 | [0.403, 0.806] | 0.42 | 0.67 | 1.00 | 7.8 |
| `A2_bm25` | 0.569 | [0.431, 0.729] | 0.25 | 0.83 | 1.00 | 3.3 |
| `A6_hybrid_rrf` | 0.569 | [0.431, 0.729] | 0.25 | 0.83 | 1.00 | 23.6 |
| `A0_none` | 0.198 | [0.142, 0.269] | 0.00 | 0.17 | 1.00 | 0.0 |

## 2b. Ranking arms at L3 (+ full traceback)

| Arm | MRR (fn) | 95% CI | recall_fn@1 | recall_fn@3 | recall_file@3 | median ms |
|---|---|---|---|---|---|---|
| `A9_bm25_then_cohere_rerank` | 0.850 | [0.675, 1.000] | 0.75 | 0.92 | 1.00 | 13.9 |
| `A8_cohere_rerank` | 0.833 | [0.653, 1.000] | 0.75 | 0.92 | 1.00 | 5618.9 |
| `A7_cohere_embed` | 0.819 | [0.667, 0.958] | 0.67 | 1.00 | 1.00 | 1.5 |
| `A1_lexical` | 0.794 | [0.617, 0.958] | 0.67 | 0.92 | 1.00 | 1.8 |
| `A2_bm25` | 0.771 | [0.604, 0.917] | 0.58 | 0.92 | 1.00 | 7.3 |
| `A6b_hybrid_lexonly` | 0.753 | [0.575, 0.917] | 0.58 | 0.92 | 1.00 | 10.4 |
| `A3_tfidf` | 0.726 | [0.528, 0.903] | 0.58 | 0.83 | 1.00 | 9.0 |
| `A4_lsa` | 0.726 | [0.528, 0.903] | 0.58 | 0.83 | 1.00 | 7.4 |
| `A6_hybrid_rrf` | 0.715 | [0.549, 0.875] | 0.50 | 0.92 | 1.00 | 30.8 |
| `A5_embed` | 0.711 | [0.510, 0.896] | 0.58 | 0.75 | 1.00 | 340.6 |
| `A0_none` | 0.198 | [0.142, 0.269] | 0.00 | 0.17 | 1.00 | 0.0 |

## 3. Arm x level grid — recall_fn@3

| Arm | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| `A0_none` | 0.17 | 0.17 | 0.17 | 0.17 |
| `A1_lexical` | 0.17 | 0.92 | 0.83 | 0.92 |
| `A2_bm25` | 0.17 | 0.92 | 0.83 | 0.92 |
| `A3_tfidf` | 0.17 | 0.92 | 0.67 | 0.83 |
| `A4_lsa` | 0.17 | 0.92 | 0.67 | 0.83 |
| `A5_embed` | 0.25 | 0.50 | 0.75 | 0.75 |
| `A6_hybrid_rrf` | 0.25 | 0.67 | 0.83 | 0.92 |
| `A6b_hybrid_lexonly` | 0.17 | 0.92 | 0.75 | 0.92 |
| `A7_cohere_embed` | 0.33 | 0.83 | 0.92 | 1.00 |
| `A8_cohere_rerank` | 0.25 | 0.92 | 0.83 | 0.92 |
| `A9_bm25_then_cohere_rerank` | 0.17 | 0.92 | 0.92 | 0.92 |

## 4. By fault category (best arm per level, L2)

| Category | n faults | mean MRR (fn) @ L2 |
|---|---|---|
| logic_error | 4 | 0.619 |
| missing_import | 1 | 0.257 |
| off_by_one | 5 | 0.691 |
| wrong_api | 2 | 0.553 |

## 5. By blast radius — how many tests the fault breaks

A fault breaking one test is a harder localisation problem than one breaking seven.

| Tests broken | n faults | mean MRR (fn) @ L2 |
|---|---|---|
| 1 | 4 | 0.770 |
| 2 | 3 | 0.568 |
| 3 | 2 | 0.527 |
| 4 | 2 | 0.587 |
| 7 | 1 | 0.283 |
