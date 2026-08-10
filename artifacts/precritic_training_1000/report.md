# Formal Pre-Critic training protocol

Pure offline preparation: model_calls=0; controller_trained=false.

Training samples: 1000
Unique content SHA256: 1000
Unique source question IDs: 997 (not an identity key)

## Labels

- correct_to_correct: 635
- correct_to_wrong: 79
- wrong_to_correct: 64
- wrong_to_wrong: 222

## Critic cost targets

- available, service-reported: 800
- unavailable and not estimated: 200

## Content isolation

- collection_200_vs_collection_800: 0
- training_vs_pilot: 0
- training_vs_validation: 0
- final_test_vs_training: 0
- final_test_vs_pilot: 0
- final_test_vs_validation: 0
- final_test_internal_duplicates: 0

## Sealed final test

- samples: 500
- split_sha256: 3063ec62c332860dfdf6fc9a9cc195ec7e3cef661209b1180868d55bb7ab7bd3
- final_test=true; sealed=true; never_evaluated=true; model_calls=0
- It is prohibited for prompt, feature, threshold, or hyperparameter selection.
