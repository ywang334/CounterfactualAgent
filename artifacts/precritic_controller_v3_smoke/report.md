# Pre-Critic Controller v3 Feature-Pipeline Smoke

This is an offline architecture and feature-pipeline validation only. It is not training or policy evaluation.

## Architecture

- Field order: `['cls', 'passage', 'question', 'option_A', 'option_B', 'option_C', 'option_D', 'solver_chunk_0..N', 'state']` plus one token per Solver chunk and a structured state token.
- Projection: 384 -> 128.
- Transformer: 2 layers, 4 heads, FFN=512, dropout=0.1.
- Heads: solver_error, critic_fix, critic_harm, and four-class transition auxiliary head.
- Cost head: absent. Existing hard-budget and fixed cost-fallback semantics are unchanged.

## Smoke result

- Samples: 8.
- Batch text embedding shape: `[8, 11, 384]`.
- Solver chunks per sample: `[3, 2, 2, 2, 2, 2, 2, 1]`.
- Controller parameters: 450,695 trainable / 450,695 total.
- MiniLM parameters: 22,713,216 frozen / 22,713,216 total.
- All factorized and four-class probabilities passed finite/range/sum checks.

## Boundaries

- local_files_only=true; mock_only=false
- llm_calls=0; model_calls=0; backend_initialized=false
- controller_trained=false; training_steps=0; backward_calls=0
- oof_generated=false; threshold_selected=false; policy_evaluated=false
- final_test_evaluated=false; final_test_examples_read=false
- v1_v2_modified=false; prompt_modified=false; parser_modified=false
- deployable=false
