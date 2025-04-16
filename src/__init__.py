from data_loader import process_all_pdfs
from embeddings import EmbeddingPipeline
from vectorstore import FaissVectorSearch

if __name__ == '__main__':
    docs = process_all_pdfs("data/pdf_files")
    embed_pipeline = EmbeddingPipeline()
    chunks = embed_pipeline.chunk_documents(docs)
    embeddings = embed_pipeline.embed_chunks(chunks)
    vectorsearch = FaissVectorSearch()
    vectorsearch.add_embeddings(chunks, embeddings)
    vectorsearch.save_store()
    vectorsearch.load_store()
    
