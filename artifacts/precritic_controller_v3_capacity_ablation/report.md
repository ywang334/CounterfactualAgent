# Controller v3 Capacity Ablation Pilot

Development-only capacity comparison. No model, seed, threshold, budget, or operating point is selected.

| Variant | Parameters | OOF accuracy | OOF corrected/degraded | Validation accuracy | Validation corrected/degraded |
|---|---:|---:|---:|---:|---:|
| V3-Tiny | 79,111 | 0.715 | 1/0 | 0.610 | 0/0 |
| V3-MaAS | 129,095 | 0.715 | 4/3 | 0.620 | 1/0 |
| V3-454K | 454,791 | 0.718 | 4/0 | 0.610 | 0/0 |

All max-net thresholds and budget curves are diagnostic only.
Existing V3-454K values were read from frozen historical artifacts and were not retrained.

## Boundaries

- Feature cache reused; embedding forward calls added: 0.
- No LLM/backend/API calls or data collection.
- Final Test examples not read; sealed manifest only.
- No cost head and no budget-logic changes.
