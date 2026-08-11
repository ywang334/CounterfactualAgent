# Controller v3 Generalization Audit

This is a pure-offline stability diagnostic. It is not an independent test and selects no new threshold, seed, budget, or architecture.

## Regime comparison

| Regime | solver-error PR-AUC | fix PR-AUC | harm PR-AUC | helpful PR-AUC | harmful PR-AUC | factorized macro-F1 | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Training full model (in-sample diagnostic) | 0.6203 | 1.0000 | 0.8575 | 0.7158 | 0.7634 | 0.6595 | 0.0743 |
| Training OOF | 0.3305 | 0.2437 | 0.1469 | 0.1026 | 0.0889 | 0.2905 | 0.2646 |
| Validation 100 | 0.5347 | 0.2237 | 0.1837 | 0.0984 | 0.0997 | 0.2719 | 0.2914 |

Training-full values are explicitly in-sample diagnostics and must not be read as generalization performance.

## Source shift

- Collection 200: net=2, call rate=0.015.
- Collection 800: net=2, call rate=0.011.
- Label-distribution total variation (200 vs 800): 0.0638.

## Seed and threshold stability

- Pairwise gate-score Pearson mean: 0.4131.
- Pairwise gate-score Spearman mean: 0.4958.
- Threshold range: 0.803589 to 0.967596.
- Bootstrap STOP fallback rate: 0.007.
- Bootstrap OOB net-gain 95% interval: [-7.00, 2.00].

## Diagnostic conclusion

The audit separates in-sample/OOF gaps, cross-source differences, and seed/threshold instability, but these observational results do not support a unique causal attribution. Training-full metrics are diagnostic only; OOF and Validation remain the relevant generalization views.

No causal attribution is made from these diagnostics, and no new operating point is selected.

## Boundaries

- No LLM/backend/API calls; no embedding forward.
- No training, backward pass, or optimizer initialization.
- Final Test examples were not read; only the sealed manifest was verified.
- Existing models, thresholds, prompts, parsers, cache, and historical outputs were unchanged.
