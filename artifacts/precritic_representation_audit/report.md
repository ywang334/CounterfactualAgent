# Pre-Critic Input Representation Audit

Pure offline tokenizer analysis. No embedding forward, Controller training, LLM/backend call, prompt/parser change, or Final Test example read occurred.

## Direct answers

1. 978/1100 combined samples (88.91%) exceed the locally configured MiniLM limit of 256 tokens. Training: 887/1000; Validation: 91/100.
2. Solver output retained/partial/dropped: 365/579/156. Parse status retained/partial/dropped: 149/109/842.
3. Corrected truncation is descriptively more severe than other labels: True. Degraded truncation is more severe: True. These comparisons are descriptive, not causal.
4. Field-independent encoding recommended: True. Late fields lose tokens under concatenation; independent field encoding would prevent earlier fields from consuming their token budget.

## Tokenizer

- Model: sentence-transformers/all-MiniLM-L6-v2
- SentenceTransformer max_seq_length: 256
- Tokenizer: BertTokenizerFast
- local_files_only: True
- embedding_forward_calls: 0

## Split truncation

| Split | N | Exceeded | Rate | Mean tokens | P50 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Training 1000 | 1000 | 887 | 88.70% | 333.47 | 323.5 | 421.1 | 453.0 | 521.1 | 782 |
| Validation 100 | 100 | 91 | 91.00% | 339.26 | 330.5 | 438.7 | 490.1 | 524.2 | 639 |
| Combined | 1100 | 978 | 88.91% | 333.99 | 324.0 | 424.0 | 454.1 | 523.0 | 782 |

## Field retention, combined

| Field | Full | Partial | Dropped | Mean retention | Individually exceeds |
|---|---:|---:|---:|---:|---:|
| passage | 1100 | 0 | 0 | 1.0000 | 0 (0.00%) |
| question | 1099 | 0 | 1 | 0.9991 | 0 (0.00%) |
| option_A | 1096 | 2 | 2 | 0.9976 | 0 (0.00%) |
| option_B | 1088 | 7 | 5 | 0.9926 | 0 (0.00%) |
| option_C | 1056 | 31 | 13 | 0.9758 | 0 (0.00%) |
| option_D | 985 | 68 | 47 | 0.9266 | 0 (0.00%) |
| solver_raw_output | 365 | 579 | 156 | 0.5967 | 7 (0.64%) |
| parse_status | 149 | 109 | 842 | 0.1794 | 0 (0.00%) |

## Transition-label truncation

| Split/label | N | Exceed rate | Mean tokens | Solver loss | Solver retained | Parse loss | Parse retained |
|---|---:|---:|---:|---:|---:|---:|---:|
| training_1000/correct_to_correct | 635 | 87.09% | 326.23 | 410 | 0.6212 | 536 | 0.1993 |
| training_1000/correct_to_wrong | 79 | 93.67% | 344.16 | 53 | 0.5363 | 70 | 0.1726 |
| training_1000/wrong_to_correct | 64 | 93.75% | 344.19 | 45 | 0.5682 | 56 | 0.1570 |
| training_1000/wrong_to_wrong | 222 | 90.09% | 347.27 | 163 | 0.5570 | 198 | 0.1437 |
| validation_100/correct_to_correct | 56 | 89.29% | 324.54 | 33 | 0.6561 | 50 | 0.1843 |
| validation_100/correct_to_wrong | 5 | 80.00% | 323.80 | 4 | 0.6837 | 4 | 0.2000 |
| validation_100/wrong_to_correct | 8 | 100.00% | 405.75 | 6 | 0.3810 | 8 | 0.0227 |
| validation_100/wrong_to_wrong | 31 | 93.55% | 351.19 | 21 | 0.5246 | 29 | 0.1202 |

## Boundaries

- offline_audit=true
- controller_v3_implemented=false
- controller_trained=false
- model_calls=0
- embedding_forward_calls=0
- final_test_evaluated=false
- deployable=false
- critical inputs and v1/v2 artifacts unchanged=true
