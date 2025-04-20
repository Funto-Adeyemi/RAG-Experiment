from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Any
from sentence_transformers import SentenceTransformer
from data_loader import process_all_pdfs
import numpy as np

class EmbeddingPipeline:
    def __init__(self, model= 'all-MiniLM-L6-v2', chunk_size = 1000, chunk_overlap = 200):
        self.model = SentenceTransformer(model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"Loading embedding model {model}")

    def chunk_documents (self, documents):
        splitter = RecursiveCharacterTextSplitter(chunk_size= self.chunk_size, chunk_overlap= self.chunk_overlap, length_function = len, separators=["\n\n", "\n", "", " "])
        chunks = splitter.split_documents(documents)
        print(f"[INFO]: {len(documents)} have been split into {len(chunks)}")
        return chunks
    
    def embed_chunks (self, chunks):
        # chunked_documents = self.chunk_documents(documents)
        embeddings = self.model.encode([chunk.page_content for chunk in chunks], show_progress_bar=True)
        print(f"[INFO]: {len(chunks)} chunks have been converted to {len(embeddings)} embeddings with shape {embeddings.shape}.")
        return embeddings
