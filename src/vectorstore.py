import os
import faiss
import numpy as np
import pickle
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from embeddings import EmbeddingPipeline

class FaissVectorSearch:
    def __init__(self, persist_dir = 'faiss_store'):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        

    def add_embeddings(self, chunks, embeddings):
        metadatas = [{'text': chunk.page_content for chunk in chunks}]
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO]: Added embeddings of shape {embeddings.shape} to the faiss vector store.")

    def save_store(self):
        faiss_path = os.path.join(self.persist_dir, 'faiss.index')
        meta_path = os.path.join(self.persist_dir, 'metadata.pkl')
        faiss.write_index(self.index, faiss_path)
        with open (meta_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO]: Saved faiss index and metadata to self.persist_dir to {self.persist_dir}.")

    def load_store (self):
        faiss_path = os.path.join(self.persist_dir, 'faiss.index')
        meta_path = os.path.join(self.persist_dir, 'metadata.pkl')
        self.index = faiss.read_index(faiss_path)
        with open (meta_path, 'rb') as f:
            self.metadata = pickle.load(f)
        print(f"[INFO]: Loaded faiss index and metadata to self.persist_dir from {self.persist_dir}.")

    def search (self, query_embedding, top_k = 5):
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx]if idx < len(self.metadata) else None
            results.append({"index": idx, "distance": dist, "metadata": meta})
        return results
    
    def query (self, query_text, top_k = 5):
        print("[INFO]: Querying vector store for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb, top_k=top_k)
    


