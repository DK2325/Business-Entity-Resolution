# Candidate generation

Built by `scripts/render_blocking_report.py` from the runs logged in
`artifacts/blocking_probe.jsonl`. Each run indexes every Source 1 record of one
country and queries every Source 2/3 record of that country, so pair counts and
runtimes carry over to the test set directly.

## Runs

| split | country | view | side | k | link recall | entities fully covered | pairs | pairs/S1 | sec | peak RSS |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | India | composite | s2s3 | 10 | 0.8838 | 0.7230 | 40,872,491 | 46.28 | 1530 | 1.80 GB |
| train | India | composite | s2s3 | 10 | 0.8838 | 0.7230 | 40,882,501 | 46.29 | 520 | 1.57 GB |
| train | India | composite+ngram | s2s3 | 10 | 0.9301 | 0.8315 | 79,310,813 | 89.80 | 625 | 1.67 GB |
| train | US | composite+ngram | s2s3 | 10 | 0.9653 | 0.9019 | 118,195,953 | 89.30 | 859 | 1.87 GB |

## Recall as k varies

**train / India / composite / side=s2s3**

| k | link recall | entities fully covered |
| ---: | ---: | ---: |
| 1 | 0.7629 | 0.5173 |
| 2 | 0.8127 | 0.5909 |
| 3 | 0.8346 | 0.6281 |
| 5 | 0.8577 | 0.6713 |
| 10 | 0.8838 | 0.7230 |

**train / India / composite / side=s2s3**

| k | link recall | entities fully covered |
| ---: | ---: | ---: |
| 1 | 0.7630 | 0.5173 |
| 2 | 0.8126 | 0.5906 |
| 3 | 0.8345 | 0.6280 |
| 5 | 0.8576 | 0.6713 |
| 10 | 0.8838 | 0.7230 |

**train / India / composite+ngram / side=s2s3**

| k | link recall | entities fully covered |
| ---: | ---: | ---: |
| 1 | 0.8333 | 0.6504 |
| 2 | 0.8769 | 0.7255 |
| 3 | 0.8951 | 0.7595 |
| 5 | 0.9126 | 0.7948 |
| 10 | 0.9301 | 0.8315 |

**train / US / composite+ngram / side=s2s3**

| k | link recall | entities fully covered |
| ---: | ---: | ---: |
| 1 | 0.9031 | 0.7632 |
| 2 | 0.9325 | 0.8258 |
| 3 | 0.9446 | 0.8533 |
| 5 | 0.9555 | 0.8788 |
| 10 | 0.9653 | 0.9019 |

## Where the misses come from

**train / India / composite / side=s2s3** — 21,353 missed links of 183,814

| reason | links | share of misses |
| --- | ---: | ---: |
| cross_script | 9,089 | 42.6% |
| other | 8,030 | 37.6% |
| empty_address | 4,234 | 19.8% |

**train / India / composite / side=s2s3** — 21,354 missed links of 183,814

| reason | links | share of misses |
| --- | ---: | ---: |
| cross_script | 9,089 | 42.6% |
| other | 8,031 | 37.6% |
| empty_address | 4,234 | 19.8% |

**train / India / composite+ngram / side=s2s3** — 12,854 missed links of 183,814

| reason | links | share of misses |
| --- | ---: | ---: |
| cross_script | 7,319 | 56.9% |
| other | 3,777 | 29.4% |
| empty_address | 1,758 | 13.7% |

**train / US / composite+ngram / side=s2s3** — 6,353 missed links of 183,014

| reason | links | share of misses |
| --- | ---: | ---: |
| other | 4,204 | 66.2% |
| empty_address | 1,930 | 30.4% |
| cross_script | 219 | 3.4% |

## Extrapolation to the test set

Test set: **1,732,544** Source 1 entities and **9,969,589** Source 2/3 records across three countries.

Source 2/3-side retrieval produces one shortlist of `k` per Source 2/3 record, so the total pair count is bounded by `|S2| + |S3|` times `k` no matter how the entities are distributed:

| k | bounded test pairs |
| ---: | ---: |
| 1 | 9,969,589 |
| 2 | 19,939,178 |
| 3 | 29,908,767 |
| 5 | 49,847,945 |
| 10 | 99,695,890 |

Measured pairs per query record, and the implied test-set totals:

| country | view | k | pairs/query | implied test pairs |
| --- | --- | ---: | ---: | ---: |
| India | composite | 10 | 9.89 | 98,584,037 |
| India | composite | 10 | 9.89 | 98,608,181 |
| India | composite+ngram | 10 | 19.19 | 191,296,883 |
| US | composite+ngram | 10 | 19.10 | 190,462,140 |

