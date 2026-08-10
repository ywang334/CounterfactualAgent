# LogiQA 2.0 Prompt Policy Validation

**policy_selection_validation=true; final_test=false; mock_only=false.**

This held-out validation compares two fixed FULL continuation policies. It does not automatically select or modify either prompt.

Samples: 100 (seed=20260811)
Actual calls: 500

## Accuracy, transition, and recorded full-workflow cost

| Strategy | Strict acc. | Strict failures | Tolerant acc. | Tolerant failures | Corrected | Degraded | Corrected/errors | Degraded/correct | Benefit-risk | Avg tokens | Avg calls | Avg latency (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Solver Only | 0.5700 | 9 | 0.6100 | 0 | 0 | 0 | 0.0000 | 0.0000 | undefined | 397.73 | 1.00 | 1.9182 |
| Full minimal_v1 | 0.5200 | 9 | 0.5800 | 0 | 15 | 18 | 0.3846 | 0.2951 | 0.8333 | 1499.69 | 3.00 | 6.3766 |
| Full structured_v2 | 0.6400 | 0 | 0.6400 | 0 | 8 | 5 | 0.2051 | 0.0820 | 1.6000 | 1975.17 | 3.00 | 6.4580 |

## Tolerant transitions

### minimal_v1

- correct→correct: 43
- correct→wrong: 18
- wrong→correct: 15
- wrong→wrong: 24

### structured_v2

- correct→correct: 56
- correct→wrong: 5
- wrong→correct: 8
- wrong→wrong: 31

## structured_v2 Critic error detection

### raw_revise_intent

- TP=17, FP=9, TN=52, FN=22
- Precision=0.6538; Recall=0.4359; F1=0.5231; Specificity=0.8525

### actionable_revise

- TP=11, FP=5, TN=56, FN=28
- Precision=0.6875; Recall=0.2821; F1=0.4000; Specificity=0.9180

## Paired policy comparison

- Exact McNemar discordant pairs: 36
- minimal_v1 correct / structured_v2 wrong: 15
- minimal_v1 wrong / structured_v2 correct: 21
- Two-sided p-value: 0.405032

### corrected ID overlap

- Intersection: 12482, 3103, 5219, 9905
- Union: 10421, 10967, 11188, 12220, 1231, 12482, 2386, 3103, 5219, 6001, 7623, 7694, 8495, 8961, 9060, 9165, 9274, 9880, 9905
- Jaccard: 0.2105

### degraded ID overlap

- Intersection: 14257
- Union: 10871, 11699, 12788, 12969, 13068, 13249, 13413, 13913, 13971, 1413, 14257, 14452, 2248, 3639, 3860, 3908, 5544, 6694, 8242, 8861, 8883, 9935
- Jaccard: 0.0455

## Minimum-cost posthoc oracles

- minimal_v1: accuracy=0.7600, Full usage=0.1500, avg tokens=573.45, avg calls=1.30, avg latency=2.5599s; posthoc_oracle=true, deployable=false.
- structured_v2: accuracy=0.6900, Full usage=0.0800, avg tokens=550.65, avg calls=1.16, avg latency=2.3864s; posthoc_oracle=true, deployable=false.

All token usage is backend-reported. Missing usage aborts the run; no usage is estimated.
No policy was automatically selected.
