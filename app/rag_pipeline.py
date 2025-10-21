"""
RAG Pipeline for Insurance Policy Documents
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import hashlib
import numpy as np
from datetime import datetime

# Document processing
from pypdf import PdfReader

# Embeddings and Vector Store
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import pickle

# Azure AI Search (optional)
try:
    from azure.search.documents import SearchClient
    from azure.core.credentials import AzureKeyCredential
    AZURE_SEARCH_AVAILABLE = True
except ImportError:
    AZURE_SEARCH_AVAILABLE = False

@dataclass
class ProcessedChunk:
    """Processed document chunk with metadata"""
    chunk_id: str
    content: str
    embedding: Optional[np.ndarray]
    metadata: Dict[str, Any]
    source_page: int
    source_file: str
    
class RAGPipeline:
    """RAG pipeline for insurance documents"""
    
    def __init__(
        self, 
        index_path: str = "data/vector_index",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        chunk_size: int = 700,
        chunk_overlap: int = 100
    ):
        self.index_path = index_path
        os.makedirs(index_path, exist_ok=True)
        
        # Initialize models
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model)
        print("Loading reranking model...")
        self.rerank_model = CrossEncoder(rerank_model)
        
        # Text splitting parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Storage
        self.chunks: List[ProcessedChunk] = []
        self.index = None
        self.metadata_store = {}
        
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF with page information"""
        pages_content = []
        
        try:
            reader = PdfReader(pdf_path)
            
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text.strip():
                    pages_content.append({
                        'text': text,
                        'page': page_num,
                        'source': os.path.basename(pdf_path)
                    })
                    
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
            
        return pages_content
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        chunks = []
        
        # Simple text splitter
        words = text.split()
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word) + 1  # +1 for space
            
            if current_size + word_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_size = 0
                overlap_words = []
                for w in reversed(current_chunk):
                    overlap_size += len(w) + 1
                    if overlap_size >= self.chunk_overlap:
                        break
                    overlap_words.insert(0, w)
                
                current_chunk = overlap_words + [word]
                current_size = sum(len(w) + 1 for w in current_chunk)
            else:
                current_chunk.append(word)
                current_size += word_size
        
        # Add last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def process_documents(self, documents_path: str = "data/policy_documents") -> None:
        """Process all PDF documents in directory"""
        print(f"Processing documents from {documents_path}...")
        
        # Load metadata
        metadata_path = os.path.join(documents_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                doc_metadata = json.load(f)
        else:
            doc_metadata = {"documents": []}
        
        # Process each PDF
        all_chunks = []
        for doc_info in doc_metadata.get("documents", []):
            pdf_path = os.path.join(documents_path, doc_info["filename"])
            
            if not os.path.exists(pdf_path):
                print(f"Warning: {pdf_path} not found")
                continue
                
            print(f"Processing {doc_info['filename']}...")
            
            # Extract text from PDF
            pages = self.extract_text_from_pdf(pdf_path)
            
            # Process each page
            for page_data in pages:
                # Split into chunks
                chunks = self._split_text(page_data['text'])
                
                # Create ProcessedChunk objects
                for i, chunk_text in enumerate(chunks):
                    chunk_id = hashlib.md5(
                        f"{doc_info['filename']}_{page_data['page']}_{i}_{chunk_text[:50]}".encode()
                    ).hexdigest()
                    
                    chunk = ProcessedChunk(
                        chunk_id=chunk_id,
                        content=chunk_text,
                        embedding=None,  # Will be computed later
                        metadata={
                            'document_title': doc_info.get('title', ''),
                            'document_type': doc_info.get('type', ''),
                            'category': doc_info.get('category', ''),
                            'keywords': doc_info.get('keywords', []),
                            'chunk_index': i,
                            'total_chunks': len(chunks)
                        },
                        source_page=page_data['page'],
                        source_file=doc_info['filename']
                    )
                    
                    all_chunks.append(chunk)
        
        print(f"Created {len(all_chunks)} chunks from {len(doc_metadata.get('documents', []))} documents")
        
        # Generate embeddings
        print("Generating embeddings...")
        self._generate_embeddings(all_chunks)
        
        # Build index
        print("Building FAISS index...")
        self._build_faiss_index(all_chunks)
        
        # Store chunks
        self.chunks = all_chunks
        
        # Save to disk
        self._save_index()
        
        print("✓ Document processing complete")
        
    def _generate_embeddings(self, chunks: List[ProcessedChunk]) -> None:
        """Generate embeddings for chunks"""
        texts = [chunk.content for chunk in chunks]
        
        # Batch encode for efficiency
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Assign embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            
    def _build_faiss_index(self, chunks: List[ProcessedChunk]) -> None:
        """Build FAISS index from embeddings"""
        if not chunks:
            return
            
        # Stack embeddings
        embeddings = np.stack([chunk.embedding for chunk in chunks])
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity for normalized vectors
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        for i, chunk in enumerate(chunks):
            self.metadata_store[i] = {
                'chunk_id': chunk.chunk_id,
                'source_file': chunk.source_file,
                'source_page': chunk.source_page,
                'metadata': chunk.metadata
            }
            
    def search(
        self, 
        query: str, 
        k: int = 10,
        rerank: bool = True,
        rerank_top_k: int = 5,
        filter_category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for relevant chunks"""
        
        if not self.index:
            self._load_index()
            
        if not self.index:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search index
        scores, indices = self.index.search(query_embedding, k)
        
        # Collect results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                metadata = self.metadata_store.get(idx, {})
                
                # Apply category filter if specified
                if filter_category and metadata.get('metadata', {}).get('category') != filter_category:
                    continue
                    
                results.append({
                    'content': chunk.content,
                    'score': float(score),
                    'source_file': chunk.source_file,
                    'source_page': chunk.source_page,
                    'metadata': metadata,
                    'chunk_id': chunk.chunk_id
                })
        
        # Rerank if requested
        if rerank and len(results) > 0:
            results = self._rerank_results(query, results, rerank_top_k)
            
        return results
    
    def _rerank_results(
        self, 
        query: str, 
        results: List[Dict[str, Any]], 
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Rerank results using cross-encoder"""
        
        # Prepare pairs for reranking
        pairs = [[query, r['content']] for r in results]
        
        # Get reranking scores
        rerank_scores = self.rerank_model.predict(pairs)
        
        # Add rerank scores and sort
        for result, rerank_score in zip(results, rerank_scores):
            result['rerank_score'] = float(rerank_score)
            
        # Sort by rerank score
        results.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
        
        return results[:top_k]
    
    def answer_question(
        self, 
        query: str,
        context_window: int = 3,
        require_citation: bool = True
    ) -> Dict[str, Any]:
        """Answer question using retrieved context"""
        
        # Search for relevant chunks
        search_results = self.search(query, k=10, rerank=True, rerank_top_k=5)
        
        if not search_results:
            return {
                'answer': "I couldn't find relevant information in the policy documents.",
                'citations': [],
                'confidence': 0.0,
                'chunks_used': []
            }
        
        # Prepare context
        context_chunks = search_results[:context_window]
        context = "\n\n".join([chunk['content'] for chunk in context_chunks])
        
        # Prepare citations
        citations = []
        for chunk in context_chunks:
            citation = f"{chunk['source_file']} (Page {chunk['source_page']})"
            if citation not in citations:
                citations.append(citation)
        
        # Calculate confidence based on scores
        avg_score = np.mean([c.get('rerank_score', c.get('score', 0)) for c in context_chunks])
        confidence = min(1.0, avg_score)  # Normalize to 0-1
        
        return {
            'context': context,
            'citations': citations,
            'confidence': confidence,
            'chunks_used': context_chunks,
            'query': query
        }
    
    def _save_index(self) -> None:
        """Save index and chunks to disk"""
        # Save FAISS index
        if self.index:
            faiss.write_index(self.index, os.path.join(self.index_path, "faiss.index"))
        
        # Save chunks and metadata
        with open(os.path.join(self.index_path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
            
        with open(os.path.join(self.index_path, "metadata.json"), "w") as f:
            json.dump(self.metadata_store, f, indent=2)
            
        print(f"✓ Index saved to {self.index_path}")
        
    def _load_index(self) -> None:
        """Load index from disk"""
        index_file = os.path.join(self.index_path, "faiss.index")
        chunks_file = os.path.join(self.index_path, "chunks.pkl")
        metadata_file = os.path.join(self.index_path, "metadata.json")
        
        if os.path.exists(index_file):
            self.index = faiss.read_index(index_file)
            
        if os.path.exists(chunks_file):
            with open(chunks_file, "rb") as f:
                self.chunks = pickle.load(f)
                
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                self.metadata_store = json.load(f)
                # Convert string keys back to integers
                self.metadata_store = {int(k): v for k, v in self.metadata_store.items()}
                
            print(f"✓ Loaded index with {len(self.chunks)} chunks")
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get RAG pipeline statistics"""
        if not self.chunks:
            self._load_index()
            
        stats = {
            'total_chunks': len(self.chunks),
            'total_documents': len(set(c.source_file for c in self.chunks)),
            'index_size': self.index.ntotal if self.index else 0,
            'embedding_dimension': self.chunks[0].embedding.shape[0] if self.chunks else 0,
            'categories': list(set(
                c.metadata.get('category', 'unknown') 
                for c in self.chunks
            ))
        }
        
        return stats

# Singleton instance
_rag_pipeline = None

def get_rag_pipeline() -> RAGPipeline:
    """Get or create RAG pipeline instance"""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
        # Try to load existing index
        _rag_pipeline._load_index()
    return _rag_pipeline

if __name__ == "__main__":
    # Initialize pipeline
    pipeline = RAGPipeline()
    
    # Process documents
    pipeline.process_documents()
    
    # Test search
    test_queries = [
        "What is the surrender charge in year 2?",
        "What is the lock-in period for ULIP?",
        "What are the exclusions in health insurance?",
        "How to file a claim?",
        "What is the grace period for premium payment?"
    ]
    
    print("\n" + "="*60)
    print("Testing RAG Pipeline")
    print("="*60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        result = pipeline.answer_question(query)
        
        if result['citations']:
            print(f"Citations: {', '.join(result['citations'])}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Context preview: {result['context'][:200]}...")
        else:
            print("No relevant information found")
    
    # Print statistics
    print("\n" + "="*60)
    print("RAG Pipeline Statistics")
    print("="*60)
    stats = pipeline.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")