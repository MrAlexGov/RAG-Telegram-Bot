"""Альтернативный RAG-бэкенд на LlamaIndex (RAG_BACKEND=llamaindex).

Тот же интерфейс, что у RAGPipeline (add_document / search_similar /
ask_question / get_collection_stats), тот же Qdrant, но загрузка, чанкинг
(SentenceSplitter — режет по границам предложений), индексация и retrieval
сделаны средствами LlamaIndex. Модели по умолчанию — те же, что в боте
(Ollama: nomic-embed-text + qwen2:7b); в eval/ эмбеддер подменяется на
HuggingFace, чтобы сравнение с LangChain-пайплайном было при равных моделях.
"""
import os
from typing import Any, Dict, List, Optional

from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from prompts import PROMPT
from config import (CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, LLM_MODEL, OLLAMA_BASE_URL,
                    RERANK_CANDIDATES, RERANKER_MODEL, TOP_K)




class LlamaIndexRAG:
    def __init__(self, embed_model=None, llm=None, collection_name: str = "rag_documents_li",
                 chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP, reranker=None):
        if embed_model is None:
            from llama_index.embeddings.ollama import OllamaEmbedding
            embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        if llm is None:
            from llama_index.llms.ollama import Ollama
            llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.3, request_timeout=300)
        self.embed_model, self.llm = embed_model, llm
        self.splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # SentenceSplitter считает размер в токенах, а не в символах — chunk_size подбирать отдельно
        self.client = QdrantClient(":memory:")
        store = QdrantVectorStore(client=self.client, collection_name=collection_name)
        self.index = VectorStoreIndex([], storage_context=StorageContext.from_defaults(vector_store=store),
                                      embed_model=embed_model)
        self.reranker = reranker
        if self.reranker is None and RERANKER_MODEL:
            from reranker import CrossEncoderReranker
            self.reranker = CrossEncoderReranker(RERANKER_MODEL)
        self._chunks = 0

    def add_documents(self, documents) -> int:
        """Индексировать готовые llama_index Document (используется в eval/)."""
        nodes = self.splitter.get_nodes_from_documents(documents)
        self.index.insert_nodes(nodes)
        self._chunks += len(nodes)
        return len(nodes)

    def add_document(self, file_path: str) -> Dict[str, Any]:
        try:
            docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
        except Exception as e:
            return {"error": f"Не удалось загрузить документ: {e}"}
        n = self.add_documents(docs)
        if not n:
            return {"error": "Не удалось извлечь текст из документа"}
        return {"chunks_count": n, "total_chunks": self._chunks, "document_name": os.path.basename(file_path)}

    def search_similar(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        hits = self.index.as_retriever(similarity_top_k=limit).retrieve(query)
        return [{"text": h.node.get_content(), "source": h.node.metadata.get("file_name", h.node.metadata.get("source", "")),
                 "page": h.node.metadata.get("page_label", 0), "score": h.score,
                 "metadata": h.node.metadata} for h in hits]

    def retrieve(self, question: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        if self.reranker:
            return self.reranker(question, self.search_similar(question, limit=RERANK_CANDIDATES), top_k)
        return self.search_similar(question, limit=top_k)

    def ask_question(self, question: str, user_id: str) -> Dict[str, Any]:
        chunks = self.retrieve(question)
        if not chunks:
            return {"answer": "В базе знаний не найдено информации по вашему вопросу.", "sources": []}
        context = "\n\n".join(f"Источник: {c['source']} (страница {c['page']})\nТекст: {c['text']}" for c in chunks)
        try:
            answer = self.llm.complete(PROMPT.format(context=context, question=question)).text
        except Exception as e:
            print(f"Error generating response: {e}")
            answer = "Извините, произошла ошибка при генерации ответа."
        return {"answer": answer, "sources": chunks}

    def get_collection_stats(self) -> Dict[str, Any]:
        return {"total_chunks": self._chunks, "total_documents": "N/A", "collection_size": self._chunks,
                "embedding_model": getattr(self.embed_model, "model_name", str(self.embed_model)),
                "llm_model": getattr(self.llm, "model", str(self.llm))}
