# Prompt Policy Stability Audit

Pure offline audit: no model backend was initialized, no model was called, and no controller was trained.

The 50 samples are prompt-development data, not a final test set. No prompt policy is automatically selected or modified.

## Policy comparison

| Policy | Solver acc. | Full acc. | Corrected | Degraded | Corrected/N | Degraded/N | Corrected/Solver errors | Degraded/Solver correct | Benefit-risk | Avg tokens | Avg calls | Avg latency (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minimal_v1 | 0.7400 | 0.5400 | 4 | 14 | 0.0800 | 0.2800 | 0.3077 | 0.3784 | 0.2857 | 1449.78 | 3.00 | 5.8463 |
| structured_v2 | 0.7400 | 0.6600 | 3 | 7 | 0.0600 | 0.1400 | 0.2308 | 0.1892 | 0.4286 | 1948.86 | 3.00 | 6.4053 |

## Tolerant transition matrices

### minimal_v1

- correct→correct: 23
- correct→wrong: 14
- wrong→correct: 4
- wrong→wrong: 9

### structured_v2

- correct→correct: 30
- correct→wrong: 7
- wrong→correct: 3
- wrong→wrong: 10

## structured_v2 Critic protocol

| Classification | Count | IDs |
|---|---:|---|
| canonical_keep | 34 | 11656, 6310, 3417, 11625, 3197, 14198, 5181, 6581, 11618, 13351, 6137, 2767, 7776, 12776, 10583, 11674, 11484, 14532, 2288, 10680, 1391, 6643, 7040, 14352, 1222, 7564, 10985, 12476, 2900, 8685, 11554, 1522, 2941, 6773 |
| contradictory_keep | 0 | None |
| incomplete_revise | 4 | 7261, 2586, 8844, 1062 |
| noop_revise | 2 | 7475, 3541 |
| actionable_revise | 10 | 2745, 8898, 5027, 13622, 2221, 11202, 1444, 7450, 486, 4219 |
| malformed | 0 | None |

- Historical critic_parse_failure: 24
- contract_inconsistency: 6
- Canonical KEEP recovered from historical parse failures: 20

## Critic error detection

- TP=6, FP=10, TN=27, FN=7
- Precision=0.3750
- Recall=0.4615
- F1=0.4138
- Specificity=0.7297

## Corrected/degraded label stability

### corrected

- Intersection: 7450
- Union: 11554, 2288, 486, 5027, 7040, 7450
- minimal_v1 only: 11554, 2288, 7040
- structured_v2 only: 486, 5027
- Jaccard: 0.1667

### degraded

- Intersection: 11202, 13622, 1444, 4219, 8898
- Union: 10583, 11202, 1222, 13622, 14352, 1444, 14532, 2221, 2745, 4219, 6137, 6643, 6773, 7776, 8685, 8898
- minimal_v1 only: 10583, 1222, 14352, 14532, 6137, 6643, 6773, 7776, 8685
- structured_v2 only: 2221, 2745
- Jaccard: 0.3125

## Minimum-cost posthoc oracles

**Both oracles use gold after generation and are deployable=false.**

| Policy | Oracle acc. | Full usage | Avg tokens | Avg calls | Avg latency (s) |
|---|---:|---:|---:|---:|---:|
| minimal_v1 | 0.8200 | 0.0800 | 470.02 | 1.16 | 2.1143 |
| structured_v2 | 0.8000 | 0.0600 | 493.00 | 1.12 | 2.1386 |

## Stop condition

No policy was selected, frozen, or modified. No further inference or prompt adjustment was performed.
