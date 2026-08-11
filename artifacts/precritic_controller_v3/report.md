# Pre-Critic Controller v3 Offline Training

This is development validation only. Final Test 500 was not read or evaluated, and no deployment operating point was selected.

## Fixed architecture and training

- Device: `{'type': 'cuda', 'device': 'cuda:0', 'cuda_available': True, 'gpu_name': 'Tesla V100-SXM2-32GB'}`
- Trainable parameters: 454,791
- Learned absolute position embeddings: 32 positions
- Optimizer: AdamW, lr=3e-4, weight_decay=1e-4, batch=32, epochs=50, gradient_clip=1.0
- Loss: error + fix + harm + 0.25 * auxiliary transition CE
- Gate score uses only factorized heads; the auxiliary head is diagnostic.
- Cost head is absent; hard-budget and fixed cost-fallback semantics are unchanged.

## Primary-seed OOF

- Corrected/degraded/net: 4/0/4
- Critic call rate: 0.0120
- OOF aggregate token and latency costs remain unavailable whenever selected samples come from Collection 200; no estimates are used.

## Validation 100 policy comparison

| Policy | Accuracy | Corrected | Degraded | Net | Critic rate | Mean total tokens | Mean calls | Mean latency (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| STOP | 0.6100 | 0 | 0 | 0 | 0.0000 | 397.73 | 1.0000 | 1.9182 |
| ALWAYS_CRITIC_ONLY | 0.6400 | 8 | 5 | 3 | 1.0000 | 1182.92 | 2.0000 | 4.7269 |
| CONTROLLER_V1_PRIMARY | 0.6300 | 2 | 0 | 2 | 0.0800 | 465.75 | 1.0800 | 2.2444 |
| CONTROLLER_V2_PRIMARY | 0.6300 | 3 | 1 | 2 | 0.1700 | 543.39 | 1.1700 | 2.5868 |
| CONTROLLER_V3_PRIMARY | 0.6100 | 0 | 0 | 0 | 0.0100 | 405.00 | 1.0100 | 1.9447 |

## Validation diagnostics

- Solver-error PR-AUC/F1: 0.5347/0.5263
- Critic-fix PR-AUC/F1: 0.2237/0.1333
- Critic-harm PR-AUC/F1: 0.1837/0.1538
- Helpful/harmful PR-AUC: 0.0984/0.0997
- Factorized/auxiliary macro-F1: 0.2719/0.2746
- Factorized/auxiliary disagreement: 10/100

No claim that v3 is superior to v1 or v2 is made from this development validation.

## Boundaries

- model_calls=0; llm_calls=0; backend_initialized=false
- final_test_evaluated=false; final_test_examples_read=false
- prompt_modified=false; parser_modified=false; v1_v2_modified=false
- rollout_collected=false; cost_head=false; deployable=false
