import os
import uuid
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredFileLoader
)
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import ollama
from sentence_transformers import SentenceTransformer
import numpy as np
from config import RERANKER_MODEL, RERANK_CANDIDATES, TOP_K

class RAGPipeline:
    def __init__(self):
        # Инициализация компонентов
        self.embedding_model = "nomic-embed-text"  # Модель для эмбеддингов через Ollama
        self.llm_model = "qwen2:7b"  # Основная модель для генерации

        # Инициализация векторной БД
        self.qdrant_client = QdrantClient(":memory:")  # Для демо - в памяти
        self.collection_name = "rag_documents"
        self._init_vector_db()

        # Инициализация текст сплиттера
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        self.reranker = None
        if RERANKER_MODEL:
            from reranker import CrossEncoderReranker
            self.reranker = CrossEncoderReranker(RERANKER_MODEL)


    def _init_vector_db(self):
        """Инициализация векторной базы данных"""
        try:
            # Создаем коллекцию если не существует
            collections = self.qdrant_client.get_collections().collections
            collection_names = [col.name for col in collections]

            if self.collection_name not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
                )
        except Exception as e:
            print(f"Error initializing vector DB: {e}")

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Получение эмбеддингов через Ollama"""
        try:
            # Получаем эмбеддинги для каждого текста
            embeddings = []
            for text in texts:
                response = ollama.embeddings(model=self.embedding_model, prompt=text)
                embeddings.append(response["embedding"])
            return embeddings
        except Exception as e:
            print(f"Ollama embedding failed: {e}")
            raise  # Вместо fallback, вызываем ошибку для тестирования

    def _load_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Загрузка и парсинг документа"""
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return []

        file_ext = os.path.splitext(file_path)[1].lower()
        print(f"Loading document with extension: {file_ext}")

        try:
            if file_ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_ext == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_ext == '.docx':
                loader = Docx2txtLoader(file_path)
            elif file_ext == '.doc':
                # Для старых .doc файлов пробуем несколько подходов
                try:
                    # Сначала попробуем через python-docx с конвертацией
                    from docx import Document
                    doc = Document(file_path)
                    text_content = []
                    for paragraph in doc.paragraphs:
                        if paragraph.text.strip():
                            text_content.append(paragraph.text)

                    # Создаем объект документа в формате langchain
                    from langchain.schema import Document as LangchainDocument
                    documents = [LangchainDocument(
                        page_content='\n'.join(text_content),
                        metadata={"source": file_path}
                    )]
                    return documents
                except Exception as e:
                    print(f"python-docx failed for .doc file, trying unstructured: {e}")
                    # Fallback на unstructured
                    loader = UnstructuredFileLoader(file_path)
            else:
                loader = UnstructuredFileLoader(file_path)

            documents = loader.load()
            print(f"Successfully loaded {len(documents)} documents from {file_path}")
            return documents
        except ImportError as e:
            print(f"Missing dependency for {file_ext} files: {e}")
            return []
        except Exception as e:
            print(f"Error loading document {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return []

    def add_document(self, file_path: str) -> Dict[str, Any]:
        """Добавление документа в векторную базу"""
        # Загружаем документ
        documents = self._load_document(file_path)
        if not documents:
            return {"error": "Не удалось загрузить документ"}

        # Разделяем на чанки
        chunks = []
        for doc in documents:
            if hasattr(doc, 'page_content') and doc.page_content.strip():
                doc_chunks = self.text_splitter.split_text(doc.page_content)
                for chunk in doc_chunks:
                    chunks.append({
                        "text": chunk,
                        "source": os.path.basename(file_path),
                        "page": getattr(doc, 'page_number', 0),
                        "metadata": doc.metadata
                    })

        if not chunks:
            return {"error": "Не удалось извлечь текст из документа"}

        print(f"Created {len(chunks)} chunks for document {file_path}")

        # Создаем эмбеддинги и сохраняем в Qdrant
        try:
            points = []
            for i, chunk in enumerate(chunks):
                embedding = self._get_embeddings([chunk["text"]])[0]

                point_id = str(uuid.uuid4())
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": chunk["text"],
                        "source": chunk["source"],
                        "page": chunk["page"],
                        "chunk_id": i
                    }
                )
                points.append(point)

            # Сохраняем в векторную БД
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            return {
                "chunks_count": len(chunks),
                "total_chunks": self.get_collection_stats()["total_chunks"],
                "document_name": os.path.basename(file_path)
            }
        except Exception as e:
            print(f"Error creating embeddings or saving to DB: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Ошибка при обработке эмбеддингов: {str(e)}"}

    def search_similar(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Поиск похожих чанков"""
        query_embedding = self._get_embeddings([query])[0]

        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit
        )

        results = []
        for result in search_results:
            results.append({
                "text": result.payload["text"],
                "source": result.payload["source"],
                "page": result.payload.get("page", 0),
                "score": result.score
            })

        return results

    def ask_question(self, question: str, user_id: str) -> Dict[str, Any]:
        """Генерация ответа на вопрос с использованием RAG"""
        # Поиск релевантных чанков
        if self.reranker:
            # двухэтапный поиск: широкий top-N из Qdrant -> cross-encoder -> top-k
            similar_chunks = self.reranker(question, self.search_similar(question, limit=RERANK_CANDIDATES), TOP_K)
        else:
            similar_chunks = self.search_similar(question, limit=TOP_K)

        if not similar_chunks:
            return {
                "answer": "В базе знаний не найдено информации по вашему вопросу.",
                "sources": []
            }

        # Формирование контекста
        context = "\n\n".join([
            f"Источник: {chunk['source']} (страница {chunk['page']})\nТекст: {chunk['text']}"
            for chunk in similar_chunks
        ])

        # Создание промпта для LLM
        prompt = f"""Используй предоставленный контекст, чтобы ответить на вопрос. Если ответа нет в контексте, скажи об этом.

Контекст:
{context}

Вопрос: {question}

Ответь подробно и информативно, цитируя источники:"""

        try:
            # Генерация ответа через Ollama
            response = ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                options={
                    'temperature': 0.3,
                    'top_k': 40,
                    'top_p': 0.9,
                }
            )

            return {
                "answer": response['response'],
                "sources": similar_chunks
            }

        except Exception as e:
            print(f"Error generating response: {e}")
            return {
                "answer": "Извините, произошла ошибка при генерации ответа.",
                "sources": similar_chunks
            }

    def get_collection_stats(self) -> Dict[str, Any]:
        """Получение статистики коллекции"""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            return {
                "total_chunks": collection_info.points_count,
                "total_documents": "N/A",  # Нужно отслеживать отдельно
                "collection_size": collection_info.vectors_count,
                "embedding_model": self.embedding_model,
                "llm_model": self.llm_model
            }
        except:
            return {
                "total_chunks": 0,
                "total_documents": 0,
                "collection_size": 0,
                "embedding_model": self.embedding_model,
                "llm_model": self.llm_model
            }