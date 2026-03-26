# retrieval/medical_retriever.py

import os
import pickle
from pathlib import Path
from typing import List

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from models.deepseek_model import DeepSeekModel
from models.llama_model import LlamaModel


class MedicalRetriever:
    """
    Production-ready RAG retriever:
    - Uses DeepSeek or LLaMA embeddings
    - Precomputes and stores embeddings
    - Uses FAISS for fast similarity search
    """

    def __init__(
        self,
        docs_path="datasets/medical_docs/",
        index_path="retrieval/faiss_index.bin",
        embeddings_path="retrieval/doc_embeddings.pkl",
        top_k=3,
        embedding_model="deepseek"
    ):
        self.docs_path = Path(docs_path)
        self.index_path = index_path
        self.embeddings_path = embeddings_path
        self.top_k = top_k
        self.embedding_model_name = embedding_model

        # Load models
        self.deepseek = DeepSeekModel()
        self.llama = LlamaModel()

        # Load documents
        self.documents = self.load_documents()

        if len(self.documents) == 0:
            raise ValueError("No documents found in datasets/medical_docs/")

        # Load or build index
        if os.path.exists(self.index_path) and os.path.exists(self.embeddings_path):
            self.load_index()
        else:
            self.build_index()

    def load_documents(self) -> List[str]:
        docs = []
        for file in self.docs_path.glob("*.txt"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    docs.append(f.read())
            except Exception as e:
                print(f"Error reading {file}: {e}")
        return docs

    def embed_text(self, text: str):
        """Use selected embedding model"""
        if self.embedding_model_name == "deepseek":
            return self.deepseek.embed_text(text)

        elif self.embedding_model_name == "llama":
            return self.llama.embed(text)

        else:
            raise ValueError("Unsupported embedding model")

    def build_index(self):
        """Compute embeddings and store FAISS index"""
        print("Building embeddings and FAISS index...")

        if faiss is None:
            raise ImportError("faiss not installed. Run: pip install faiss-cpu")

        embeddings = []

        for doc in self.documents:
            try:
                emb = self.embed_text(doc)
                embeddings.append(emb)
            except Exception as e:
                print("Embedding error:", e)

        if len(embeddings) == 0:
            raise ValueError("No embeddings were created.")

        embeddings = np.array(embeddings).astype("float32")

        # Save embeddings
        with open(self.embeddings_path, "wb") as f:
            pickle.dump(embeddings, f)

        # Build FAISS index
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        faiss.write_index(self.index, self.index_path)

        self.embeddings = embeddings

        print(f"FAISS index built with {self.index.ntotal} vectors")

    def load_index(self):
        """Load FAISS index and embeddings"""
        print("Loading FAISS index...")

        with open(self.embeddings_path, "rb") as f:
            self.embeddings = pickle.load(f)

        self.index = faiss.read_index(self.index_path)

        print(f"FAISS index loaded with {self.index.ntotal} vectors")
        print(f"Documents loaded: {len(self.documents)}")

    def retrieve(self, query: str) -> List[str]:
        """Retrieve top-K relevant documents safely"""

        if not query or not query.strip():
            return []

        try:
            query_emb = np.array([self.embed_text(query)]).astype("float32")

            distances, indices = self.index.search(query_emb, self.top_k)

            results = []

            for i, idx in enumerate(indices[0]):
                # Validate index
                if idx < 0 or idx >= len(self.documents):
                    continue

                # Optional: filter noisy matches
                if distances[0][i] > 100:  # adjust threshold if needed
                    continue

                results.append(self.documents[idx])

            return results

        except Exception as e:
            print("Retriever error:", e)
            return []


# 🔥 GLOBAL retriever (loaded once)
_retriever = None


def retrieve_medical_context(query, top_k=3):
    global _retriever

    if _retriever is None:
        _retriever = MedicalRetriever(top_k=top_k)

    docs = _retriever.retrieve(query)

    if not docs:
        return "No relevant medical context found."

    return "\n".join(docs)