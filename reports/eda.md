# Exploratory data analysis

Regenerate with `python scripts/run_eda.py`. Every figure below comes from that script.

## 1. Files

| file | rows | size (MB) | empty address |
| --- | ---: | ---: | ---: |
| train_source1 | 2,206,821 | 210.1 | 0 (0.0%) |
| train_source2 | 5,034,616 | 489.3 | 168,967 (3.36%) |
| train_source3 | 5,285,603 | 503.7 | 175,916 (3.33%) |
| train_ground_truth | 2,206,821 | 127.0 | - |
| test_source1 | 1,732,544 | 175.0 | 0 (0.0%) |
| test_source2 | 4,887,273 | 509.5 | 129,408 (2.65%) |
| test_source3 | 5,082,316 | 506.0 | 136,098 (2.68%) |

### Country mix

| file | France | India | US |
| --- | ---: | ---: | ---: |
| train_source1 | 0 | 883,188 | 1,323,633 |
| train_source2 | 0 | 2,017,799 | 3,016,817 |
| train_source3 | 0 | 2,115,547 | 3,170,056 |
| test_source1 | 259,452 | 809,986 | 663,106 |
| test_source2 | 703,378 | 2,312,565 | 1,871,330 |
| test_source3 | 731,615 | 2,405,000 | 1,945,701 |

## 2. Ground-truth structure

- Source 1 entities: **2,206,821**
- Singletons: **123,247 (5.58%)** — each worth a full 1.0 if predicted empty
- Total links: **7,638,365**, mean **3.461** per entity
- Distinct S2/S3 IDs appearing in the truth: **7,638,365**

### Exclusivity

IDs claimed by more than one Source 1 entity: **0** of 7,638,365.

> **Exclusivity holds.** Every Source 2/3 record belongs to at most one Source 1 entity, so matching is a constrained assignment: candidates can be made to compete and only the best claim kept. This is the backbone of the decision rule.

### Country agreement

Country mismatches across true links: **0** of 7,638,365 — country is a safe hard blocking key.

### Distractors

| source | rows | matched | distractors |
| --- | ---: | ---: | ---: |
| S2 | 5,034,616 | 3,693,619 | 1,340,997 (26.64%) |
| S3 | 5,285,603 | 3,944,746 | 1,340,857 (25.37%) |

### Matches per entity

| matches | entities |
| ---: | ---: |
| 0 | 123,247 |
| 1 | 119,157 |
| 2 | 375,212 |
| 3 | 530,841 |
| 4 | 484,115 |
| 5 | 321,957 |
| 6 | 164,868 |
| 7 | 63,968 |
| 8 | 18,680 |
| 9 | 4,205 |
| 10 | 534 |
| 11 | 37 |

## 3. Source 2 vs Source 3 inside one group

Sampled **20,000** non-singleton groups; 17,075 contain records from both sources.

Best cross-source similarity within a group (name token-set ratio, averaged with address Jaccard where both addresses exist):

| p5 | p25 | p50 | p75 | p95 | mean |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.4974 | 0.6667 | 0.75 | 0.8333 | 0.9583 | 0.7429 |

- Groups containing a near-identical (>= 0.9) S2/S3 pair: **11.3%**
- Pairs identical after normalisation: **479**

Same-source siblings (two records from one source under one entity) — this is the intra-source noise level, name token-set ratio:

| source | p5 | p25 | p50 | p75 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| S2 | 0.3684 | 0.7407 | 0.973 | 1.0 | 1.0 |
| S3 | 0.3111 | 0.7407 | 1.0 | 1.0 | 1.0 |

## 4. True pairs vs hard negatives

**21,979** true pairs against **30,000** hard negatives (same country, sharing at least one token, not a true match), drawn from a pool of 421,979 records.

Median similarity, and the gap between them:

| feature | true p50 | hard-neg p50 | separation |
| --- | ---: | ---: | ---: |
| name_jaccard | 1.0 | 0.0 | **+1.0000** |
| name_jaccard_romanized | 1.0 | 0.0 | **+1.0000** |
| addr_numeric_jaccard | 1.0 | 0.0 | **+1.0000** |
| name_token_set_ratio | 1.0 | 0.383 | **+0.6170** |
| addr_jaccard_romanized | 0.6923 | 0.1667 | **+0.5256** |
| addr_jaccard | 0.6667 | 0.1667 | **+0.5000** |
| addr_empty_either | 0.0 | 0.0 | **+0.0000** |

