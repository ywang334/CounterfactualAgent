from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hierarchical_control.cli import main
from hierarchical_control.precritic_controller_v3 import (
    ANSWER_CATEGORIES,
    EMBEDDING_DIM,
    EXPECTED_TRAINABLE_PARAMETER_MAX,
    EXPECTED_TRAINABLE_PARAMETER_MIN,
    LocalMiniLMFieldEncoder,
    PreCriticControllerV3,
    STRUCTURED_STATE_FEATURES,
    build_v3_feature_batch,
    chunk_solver_output,
    controller_parameter_counts,
    structured_state_vector,
)


class TinyFieldEncoder:
    name = "tiny-injected"
    dimension = EMBEDDING_DIM
    local_files_only = True
    mock_only = True

    def __init__(self, max_seq_length: int = 5) -> None:
        self.max_seq_length = max_seq_length
        self.embedding_forward_calls = 0
        self.encoded_batches: list[list[str]] = []
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def content_token_ids(self, text: str) -> list[int]:
        result = []
        for token in text.split():
            if token not in self._token_to_id:
                token_id = len(self._token_to_id) + 10
                self._token_to_id[token] = token_id
                self._id_to_token[token_id] = token
            result.append(self._token_to_id[token])
        return result

    def decode_content_ids(self, token_ids):
        return " ".join(self._id_to_token[token_id] for token_id in token_ids)

    def sequence_token_length(self, text: str) -> int:
        return len(self.content_token_ids(text)) + 2

    def special_token_count(self) -> int:
        return 2

    def encode(self, texts):
        values = list(texts)
        self.encoded_batches.append(values)
        self.embedding_forward_calls += 1
        rows = torch.zeros((len(values), self.dimension), dtype=torch.float32)
        for index, text in enumerate(values):
            rows[index, 0] = float(len(self.content_token_ids(text)))
            rows[index, 1] = float(index + 1)
        return rows


def _model_input(solver_words: int = 2):
    return {
        "problem": {
            "passage": "passage text",
            "question": "question text",
            "options": {
                "A": "alpha",
                "B": "beta",
                "C": "gamma",
                "D": "delta",
            },
        },
        "solver": {
            "raw_output": " ".join(f"solver{index}" for index in range(solver_words)),
            "parse_status": {
                "strict_answer": "B",
                "strict_parse_failure": False,
                "tolerant_answer": "B",
                "tolerant_parse_failure": False,
                "tolerant_match_count": 1,
                "tolerant_conflict": False,
            },
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
                "calls": 1,
                "latency_seconds": 0.5,
            },
        },
    }


def test_field_order_solver_chunks_padding_and_batch_shapes():
    encoder = TinyFieldEncoder(max_seq_length=5)
    batch = build_v3_feature_batch(
        [_model_input(2), _model_input(7)], encoder
    )
    assert batch.field_order[0] == (
        "cls",
        "passage",
        "question",
        "option_A",
        "option_B",
        "option_C",
        "option_D",
        "solver_chunk_0",
        "state",
    )
    assert batch.field_order[1][-4:] == (
        "solver_chunk_0",
        "solver_chunk_1",
        "solver_chunk_2",
        "state",
    )
    assert batch.solver_chunk_counts == (1, 3)
    assert batch.text_embeddings.shape == (2, 11, EMBEDDING_DIM)
    assert batch.type_ids.shape == batch.padding_mask.shape == (2, 11)
    assert batch.structured_state.shape == (2, len(STRUCTURED_STATE_FEATURES))
    assert batch.state_positions.tolist() == [8, 10]
    assert batch.padding_mask[0].tolist() == [False] * 9 + [True, True]
    assert batch.padding_mask[1].tolist() == [False] * 11
    # All fields for the full batch are encoded together and in source order.
    assert len(encoder.encoded_batches) == 1
    assert encoder.encoded_batches[0][:6] == [
        "passage text",
        "question text",
        "alpha",
        "beta",
        "gamma",
        "delta",
    ]


def test_solver_chunking_covers_tokenizer_ids_without_truncation():
    encoder = TinyFieldEncoder(max_seq_length=6)  # four content IDs per chunk
    chunks = chunk_solver_output("one two three four five six seven eight nine", encoder)
    assert chunks.source_content_tokens == 9
    assert chunks.max_content_tokens_per_chunk == 4
    assert chunks.chunk_content_tokens == (4, 4, 1)
    assert sum(chunks.chunk_content_tokens) == chunks.source_content_tokens
    assert all(
        encoder.sequence_token_length(text) <= encoder.max_seq_length
        for text in chunks.texts
    )


def test_structured_state_uses_eight_frozen_features_parse_and_answer():
    vector, parse_status, answer = structured_state_vector(_model_input())
    assert vector.shape == (18,)
    assert len(STRUCTURED_STATE_FEATURES) == 8 + 5 + len(ANSWER_CATEGORIES)
    assert parse_status == "both_parsed_agree"
    assert answer == "B"
    assert vector[-len(ANSWER_CATEGORIES) :].tolist() == [0, 1, 0, 0, 0]
    leaked = _model_input()
    leaked["gold"] = "B"
    with pytest.raises(ValueError, match="whitelist"):
        structured_state_vector(leaked)


def test_forward_shapes_probabilities_padding_mask_and_parameter_count():
    encoder = TinyFieldEncoder(max_seq_length=5)
    batch = build_v3_feature_batch([_model_input(2), _model_input(7)], encoder)
    torch.manual_seed(7)
    model = PreCriticControllerV3().eval()
    with torch.inference_mode():
        output = model(batch)
    for name in (
        "solver_error_logits",
        "critic_fix_logits",
        "critic_harm_logits",
        "p_solver_error",
        "p_critic_fix_given_error",
        "p_critic_harm_given_correct",
    ):
        assert output[name].shape == (2,)
    assert output["transition_logits"].shape == (2, 4)
    for name in (
        "factorized_transition_probabilities",
        "transition_aux_probabilities",
    ):
        probabilities = output[name]
        assert probabilities.shape == (2, 4)
        assert torch.isfinite(probabilities).all()
        assert torch.all((probabilities >= 0) & (probabilities <= 1))
        assert torch.allclose(probabilities.sum(-1), torch.ones(2), atol=1e-6)
    for name in (
        "p_solver_error",
        "p_critic_fix_given_error",
        "p_critic_harm_given_correct",
    ):
        assert torch.all((output[name] >= 0) & (output[name] <= 1))

    changed_padding = batch.text_embeddings.clone()
    changed_padding[batch.padding_mask] = 1_000_000.0
    changed = type(batch)(
        **{**batch.__dict__, "text_embeddings": changed_padding}
    )
    with torch.inference_mode():
        changed_output = model(changed)
    assert torch.allclose(
        output["transition_logits"], changed_output["transition_logits"], atol=1e-6
    )

    counts = controller_parameter_counts(model)
    assert counts == {"total": 450695, "trainable": 450695, "frozen": 0}
    assert EXPECTED_TRAINABLE_PARAMETER_MIN <= counts["trainable"]
    assert counts["trainable"] <= EXPECTED_TRAINABLE_PARAMETER_MAX
    assert not hasattr(model, "cost_head")


class _FakeTokenizer:
    def __call__(self, text, **kwargs):
        count = len(text.split())
        ids = list(range(count))
        if kwargs.get("add_special_tokens", True):
            ids = [101, *ids, 102]
        return {"input_ids": ids}

    def decode(self, token_ids, **kwargs):
        return " ".join(f"token{value}" for value in token_ids)

    def num_special_tokens_to_add(self, pair=False):
        return 2


class _FakeSentenceTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(3))
        self.tokenizer = _FakeTokenizer()
        self.max_seq_length = 16

    def get_sentence_embedding_dimension(self):
        return EMBEDDING_DIM

    def encode(self, texts, **kwargs):
        return torch.zeros((len(texts), EMBEDDING_DIM), dtype=torch.float32)


def test_local_minilm_loader_is_local_only_and_freezes_every_parameter():
    calls = []
    fake_model = _FakeSentenceTransformer()

    def factory(path, **kwargs):
        calls.append((path, kwargs))
        return fake_model

    encoder = LocalMiniLMFieldEncoder(
        model_factory=factory,
        tokenizer_bundle=SimpleNamespace(
            snapshot_path="/local/cache/minilm",
            max_seq_length=16,
        ),
    )
    assert calls == [
        (
            "/local/cache/minilm",
            {"device": "cpu", "local_files_only": True},
        )
    ]
    assert encoder.local_files_only is True
    assert encoder.max_seq_length == 16
    assert all(not parameter.requires_grad for parameter in fake_model.parameters())
    assert encoder.parameter_counts() == {"total": 3, "trainable": 0, "frozen": 3}


def test_v3_smoke_cli_never_initializes_backend_or_exposes_training_options(
    monkeypatch, tmp_path
):
    import hierarchical_control.cli as cli

    def forbidden(*args, **kwargs):
        raise AssertionError("Controller v3 smoke must not initialize a backend")

    received = {}

    def fake_smoke(**kwargs):
        received.update(kwargs)
        return {
            "precritic_controller_v3_smoke": True,
            "controller_trained": False,
            "model_calls": 0,
            "deployable": False,
        }

    monkeypatch.setattr(cli, "OpenAIBackend", forbidden)
    monkeypatch.setattr(cli, "MockBackend", forbidden)
    monkeypatch.setattr(cli, "run_precritic_controller_v3_smoke", fake_smoke)
    output = tmp_path / "v3"
    assert main(
        [
            "smoke-precritic-controller-v3",
            "--sample-count",
            "3",
            "--output-dir",
            str(output),
        ]
    ) == 0
    assert received["sample_count"] == 3
    assert received["output_dir"] == str(output)
    assert "backend" not in received

