"""Deterministic unit tests for the no-LLM per-turn curation: cosine ranking + dedup."""
import struct
from types import SimpleNamespace

from app.planner.pool_rerank import (
    _cosine, _unpack_embedding, _dedup_take_top_k, DEDUP_COSINE_THRESHOLD,
)


def pack(v):
    return struct.pack(f"<{len(v)}f", *v)


def item(name, summary, emb=None):
    return SimpleNamespace(dependency_name=name, payload_summary=summary,
                           summary_embedding=pack(emb) if emb else None, kind="data")


# ---- cosine ----

def test_cosine_identical_is_one():
    assert abs(_cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) - 1.0) < 1e-9


def test_cosine_orthogonal_is_zero():
    assert abs(_cosine([1.0, 0.0], [0.0, 1.0])) < 1e-9


def test_cosine_opposite_is_minus_one():
    assert abs(_cosine([1.0, 0.0], [-1.0, 0.0]) + 1.0) < 1e-9


def test_cosine_guards_return_zero():
    assert _cosine([], [1.0]) == 0.0           # empty
    assert _cosine([1.0, 2.0], [1.0]) == 0.0   # length mismatch
    assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0  # zero norm


def test_unpack_embedding_roundtrip():
    v = [0.5, -1.25, 3.0, 0.0]
    assert _unpack_embedding(pack(v)) == v
    assert _unpack_embedding(b"") == []


# ---- dedup_take_top_k ----

def test_dedup_keeps_distinct_items_up_to_max():
    scored = [
        (0.9, item("policy", "POLICY A"), [1.0, 0.0]),
        (0.8, item("claims", "CLAIM X"), [0.0, 1.0]),
        (0.7, item("rates", "RATE Z"), [0.7, 0.7]),
    ]
    picks = _dedup_take_top_k(scored, max_picks=3)
    assert [p[1].dependency_name for p in picks] == ["policy", "claims", "rates"]


def test_dedup_drops_key_equal_duplicate():
    # same dependency_name + same summary prefix -> second is a duplicate
    a = item("policy", "POLICY #INS-882431 full coverage", [1.0, 0.0])
    b = item("policy", "POLICY #INS-882431 full coverage", [0.0, 1.0])  # orthogonal emb, but key-equal
    picks = _dedup_take_top_k([(0.9, a, [1.0, 0.0]), (0.85, b, [0.0, 1.0])], max_picks=3)
    assert len(picks) == 1
    assert picks[0][1] is a


def test_dedup_drops_embedding_near_duplicate():
    # different summaries (distinct keys) but embeddings above the cosine threshold
    a = item("rag", "claims summary one", [1.0, 0.0])
    b = item("rag", "claims summary two", [0.999, 0.001])  # cosine ~1 with a
    assert _cosine([1.0, 0.0], [0.999, 0.001]) >= DEDUP_COSINE_THRESHOLD
    picks = _dedup_take_top_k([(0.9, a, [1.0, 0.0]), (0.88, b, [0.999, 0.001])], max_picks=3)
    assert len(picks) == 1


def test_dedup_respects_max_picks():
    scored = [(0.9 - i * 0.1, item(f"d{i}", f"S{i}"), [float(i), 1.0]) for i in range(5)]
    assert len(_dedup_take_top_k(scored, max_picks=2)) == 2
