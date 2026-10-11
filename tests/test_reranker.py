"""Логика реранкинга без загрузки модели: scorer подменяется функцией."""
from reranker import rerank


def overlap_scorer(pairs):
    # сколько слов вопроса встречается в чанке
    return [len(set(q.lower().split()) & set(t.lower().split())) for q, t in pairs]


def test_rerank_orders_by_score_and_cuts_top_k():
    cands = [{"text": "погода в Москве"}, {"text": "как работает индекс в PostgreSQL"}, {"text": "индекс B-дерево PostgreSQL"}]
    out = rerank("индекс PostgreSQL B-дерево", cands, overlap_scorer, top_k=2)
    assert [c["text"] for c in out] == ["индекс B-дерево PostgreSQL", "как работает индекс в PostgreSQL"]
    assert out[0]["rerank_score"] >= out[1]["rerank_score"]


def test_rerank_keeps_payload_and_handles_empty():
    assert rerank("q", [], overlap_scorer, top_k=3) == []
    out = rerank("a", [{"text": "a", "source": "doc.pdf", "page": 2}], overlap_scorer, top_k=3)
    assert out[0]["source"] == "doc.pdf" and out[0]["page"] == 2
