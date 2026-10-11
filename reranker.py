"""Реранкинг кандидатов cross-encoder'ом.

Векторный поиск (bi-encoder) быстрый, но грубый: вопрос и чанк кодируются
независимо. Cross-encoder читает пару (вопрос, чанк) целиком и точнее
оценивает релевантность, но медленнее — поэтому схема двухэтапная:
достаём top-N из Qdrant, переупорядочиваем и оставляем top-k.
"""
from typing import Callable, Dict, List, Optional, Sequence

Scorer = Callable[[Sequence[tuple]], Sequence[float]]


def rerank(query: str, candidates: List[Dict], scorer: Scorer, top_k: int) -> List[Dict]:
    """Переупорядочить кандидатов (dict с ключом "text") по score от scorer."""
    if not candidates:
        return []
    scores = scorer([(query, c["text"]) for c in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)
    return [{**c, "rerank_score": float(s)} for c, s in ranked[:top_k]]


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", max_length: int = 512,
                 device: Optional[str] = None):
        from sentence_transformers import CrossEncoder  # тяжёлый импорт — только при включённом реранкере
        self.model_name = model_name
        self.model = CrossEncoder(model_name, max_length=max_length, device=device)

    def score(self, pairs: Sequence[tuple]) -> Sequence[float]:
        return self.model.predict(list(pairs), batch_size=16, show_progress_bar=False)

    def __call__(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        return rerank(query, candidates, self.score, top_k)
