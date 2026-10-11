"""Оценка retrieval: LangChain vs LlamaIndex, с реранкером и без.

Корпус — русский SberQuAD (validation): параграфы Википедии, собранные по 5
в синтетические документы, чтобы чанкинг реально влиял на результат. Вопрос
считается найденным на позиции i, если i-й чанк из того же документа и содержит
эталонный ответ. Эмбеддер одинаковый для обоих фреймворков
(multilingual-e5-base), Qdrant in-memory — сравниваются только чанкинг и
retrieval-обвязка, а затем вклад cross-encoder реранкера.

Запуск (из корня репо): python eval/retrieval_eval.py --docs 100 --questions 200
"""
import argparse, json, os, random, sys, time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("RERANKER_MODEL", "")  # реранкер создаём явно ниже

from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from reranker import CrossEncoderReranker

EMB = "intfloat/multilingual-e5-base"
RERANK = "BAAI/bge-reranker-v2-m3"
CANDIDATES = 20


def build_corpus(n_docs, n_questions, per_doc=5, seed=42):
    """SberQuAD без статей (title у всех один) — собираем синтетические документы
    из per_doc случайных параграфов, чтобы чанкинг резал документы по-разному."""
    ds = load_dataset("kuznetsoffandrey/sberquad", split="validation")
    by_ctx = defaultdict(list)
    for r in ds:
        if r["answers"]["text"]:
            by_ctx[r["context"]].append(r)
    ctxs = sorted(by_ctx)
    random.Random(seed).shuffle(ctxs)
    ctxs = ctxs[:n_docs * per_doc]
    docs, qs = {}, []
    for i in range(n_docs):
        name = f"doc{i:03d}"
        part = ctxs[i * per_doc:(i + 1) * per_doc]
        docs[name] = "\n\n".join(part)
        for c in part:
            for r in by_ctx[c]:
                qs.append({"q": r["question"], "answer": r["answers"]["text"][0], "title": name, "context": c})
    random.Random(seed).shuffle(qs)
    return docs, qs[:n_questions]


class LangChainRetriever:
    """Как в rag_pipeline.py: RecursiveCharacterTextSplitter(1000/200) -> Qdrant cosine."""
    def __init__(self, docs, emb):
        self.emb = emb
        sp = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        self.chunks = [{"text": c, "source": t} for t, d in docs.items() for c in sp.split_text(d)]
        self.client = QdrantClient(":memory:")
        vecs = emb.get_text_embedding_batch(["passage: " + c["text"] for c in self.chunks])
        self.client.create_collection("lc", vectors_config=VectorParams(size=len(vecs[0]), distance=Distance.COSINE))
        self.client.upsert("lc", [PointStruct(id=i, vector=v, payload=c) for i, (v, c) in enumerate(zip(vecs, self.chunks))])

    def search(self, q, k):
        v = self.emb.get_query_embedding("query: " + q)
        return [dict(h.payload, score=h.score) for h in self.client.query_points("lc", query=v, limit=k).points]


class LlamaIndexRetriever:
    """llamaindex_pipeline.LlamaIndexRAG с тем же эмбеддером (SentenceSplitter 256/50 токенов)."""
    def __init__(self, docs, emb):
        from llamaindex_pipeline import LlamaIndexRAG
        self.rag = LlamaIndexRAG(embed_model=emb, llm="eval-no-llm", chunk_size=256, chunk_overlap=50)
        self.n_chunks = self.rag.add_documents([Document(text=d, metadata={"source": t}, excluded_embed_metadata_keys=["source"])
                                                for t, d in docs.items()])

    def search(self, q, k):
        return self.rag.search_similar(q, limit=k)


def is_hit(chunk, item):
    src = chunk.get("source") or chunk.get("metadata", {}).get("source")
    return src == item["title"] and item["answer"] in chunk["text"]


def metrics(ranked_lists, qs):
    h1 = h3 = mrr = 0
    for ranked, it in zip(ranked_lists, qs):
        pos = next((i for i, c in enumerate(ranked) if is_hit(c, it)), None)
        if pos is not None:
            h1 += pos == 0; h3 += pos < 3; mrr += 1 / (pos + 1)
    n = len(qs)
    return {"hit@1": round(h1 / n, 3), "hit@3": round(h3 / n, 3), "mrr@10": round(mrr / n, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=100)
    ap.add_argument("--questions", type=int, default=200)
    ap.add_argument("--out", default="eval/results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    docs, qs = build_corpus(a.docs, a.questions)
    print(f"corpus: {len(docs)} docs, {sum(len(d) for d in docs.values())} chars, {len(qs)} questions", flush=True)
    # e5 требует префиксы query:/passage: — в LlamaIndex задаём их через instruction-параметры
    emb = HuggingFaceEmbedding(model_name=EMB, query_instruction="query: ", text_instruction="passage: ")
    raw_emb = HuggingFaceEmbedding(model_name=EMB)  # для LangChain-ветки префиксы ставим руками
    rr = CrossEncoderReranker(RERANK, max_length=384)
    out = {"embedder": EMB, "reranker": RERANK, "candidates": CANDIDATES, "docs": len(docs), "questions": len(qs), "variants": {}}
    dump = {}
    for name, R in [("langchain", lambda: LangChainRetriever(docs, raw_emb)), ("llamaindex", lambda: LlamaIndexRetriever(docs, emb))]:
        t0 = time.time(); r = R()
        n_chunks = len(r.chunks) if hasattr(r, "chunks") else r.n_chunks
        cands = [r.search(it["q"], CANDIDATES) for it in qs]
        t_dense = time.time() - t0
        out["variants"][name] = {"chunks": n_chunks, **metrics([c[:10] for c in cands], qs), "seconds": round(t_dense, 1)}
        print(name, out["variants"][name], flush=True)
        t0 = time.time()
        reranked = [rr(it["q"], c, 10) for it, c in zip(qs, cands)]
        out["variants"][name + "+rerank"] = {"chunks": n_chunks, **metrics(reranked, qs),
                                             "rerank_seconds": round(time.time() - t0, 1)}
        print(name + "+rerank", out["variants"][name + "+rerank"], flush=True)
        dump[name] = [[c["text"] for c in x[:3]] for x in cands]
        dump[name + "+rerank"] = [[c["text"] for c in x[:3]] for x in reranked]
        json.dump(out, open(f"{a.out}/retrieval.json", "w"), ensure_ascii=False, indent=2)
    json.dump({"questions": qs, "top3": dump}, open(f"{a.out}/contexts.json", "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
