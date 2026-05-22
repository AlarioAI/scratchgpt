import pytest

from benchmarks._shared import materialize_text_dataset, parse_arch_overrides


def test_parse_arch_overrides_coerces_bool_int_and_string() -> None:
    overrides = parse_arch_overrides(["tie_weights=true", "num_blocks=8", "ffn_activation=gelu"])

    assert overrides == {
        "tie_weights": True,
        "num_blocks": 8,
        "ffn_activation": "gelu",
    }


def test_materialize_text_dataset_filters_truncates_and_limits() -> None:
    ds = materialize_text_dataset(
        [
            {"text": "too short"},
            {"text": "  abcdefghij  "},
            {"text": "klmnopqrst"},
            {"text": "uvwxyz"},
        ],
        text_column="text",
        limit=2,
        min_chars=10,
        max_chars=4,
    )

    assert len(ds) == 2
    assert ds[0]["text"] == "abcd"
    assert ds[1]["text"] == "klmn"


def test_materialize_text_dataset_errors_on_empty_result() -> None:
    with pytest.raises(ValueError, match="No usable text examples"):
        materialize_text_dataset([{"body": "missing column"}], text_column="text", limit=1)
