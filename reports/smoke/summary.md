# bitnet-embed-smoke-report

## Overview

- plan: `smoke_stages`
- stage_count: `3`
- best_stage_by_loss: `stage2_initial_retrieval`

## Latency

- `p50_latency`: `6.495900015579537e-05`
- `p95_latency`: `0.00019841700122924522`
- `startup_p50`: `1.5839978004805744e-06`
- `startup_p95`: `1.5839978004805744e-06`
- `throughput`: `554.1444713652988`

## STS

- `sts_spearman`: `-0.3999999999999999`

## Memory

- `peak_cuda_mb`: `0.0`
- `peak_rss_mb`: `182.859375`
- `startup_cuda_mb`: `0.0`
- `startup_rss_mb`: `182.859375`

## ANN

- `ann_mrr@1`: `0.0`
- `ann_mrr@5`: `0.29166666666666663`
- `ann_ndcg@1`: `0.0`
- `ann_ndcg@5`: `0.4653382855837924`
- `ann_recall@1`: `0.0`
- `ann_recall@5`: `1.0`

## Package

- `package_name`: `bitnet-smoke-hf`
- `format`: `bitnet-embed-hf-style`
- metrics:
  - `sts_spearman`: `0.19999999999999996`
