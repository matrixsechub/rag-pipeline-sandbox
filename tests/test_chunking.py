import pytest

from src.chunking import fixed_chunk


def test_fixed_chunk_rejects_overlap_that_would_not_advance():
    with pytest.raises(ValueError, match="overlap must be smaller than chunk_size"):
        fixed_chunk("x" * 500, chunk_size=300, overlap=300)


def test_fixed_chunk_rejects_negative_step_inputs():
    with pytest.raises(ValueError, match="chunk_size must be greater than 0"):
        fixed_chunk("x", chunk_size=0, overlap=0)

    with pytest.raises(ValueError, match="overlap must be greater than or equal to 0"):
        fixed_chunk("x", chunk_size=300, overlap=-1)
