"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Embeddings Generator + Supabase Vector Storage                              ║
║                                                                              ║
║  - Generates embeddings using sentence-transformers (all-MiniLM-L6-v2)      ║
║  - Stores vectors in Supabase pgvector for similarity search                ║
║  - Supports batch processing for efficiency                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import logging
from typing import Optional
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
from dotenv import load_dotenv

from pdf_processor import DocumentChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddedChunk:
    """A document chunk with its embedding vector."""
    content: str
    embedding: list[float]
    source: str
    section: Optional[str] = None
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EmbeddingGenerator:
    """
    Generate embeddings using sentence-transformers.
    
    Default model: all-MiniLM-L6-v2
    - 384 dimensions
    - Fast and efficient
    - Good quality for semantic search
    - Runs locally (no API costs)
    
    Alternative models:
    - "all-mpnet-base-v2": Higher quality, 768 dims, slower
    - "multi-qa-MiniLM-L6-cos-v1": Optimized for Q&A
    - "paraphrase-multilingual-MiniLM-L12-v2": Multilingual support
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",  # "cuda" for GPU
        normalize: bool = True,
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Sentence transformer model to use
            device: Device for inference ("cpu" or "cuda")
            normalize: Whether to normalize embeddings to unit length
        """
        self.model_name = model_name
        self.normalize = normalize
        
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Model loaded. Embedding dimension: {self.dimension}")
    
    def embed(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        return embedding.tolist()
    
    def embed_batch(
        self, 
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Show progress bar
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
        )
        return embeddings.tolist()
    
    def embed_chunks(
        self,
        chunks: list[DocumentChunk],
        batch_size: int = 32,
    ) -> list[EmbeddedChunk]:
        """
        Embed a list of document chunks.
        
        Args:
            chunks: List of DocumentChunk objects
            batch_size: Batch size for processing
            
        Returns:
            List of EmbeddedChunk objects with embeddings
        """
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embed_batch(texts, batch_size=batch_size)
        
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            embedded_chunks.append(EmbeddedChunk(
                content=chunk.content,
                embedding=embedding,
                source=chunk.source,
                section=chunk.section,
                metadata=chunk.metadata,
            ))
        
        logger.info(f"Embedded {len(embedded_chunks)} chunks")
        return embedded_chunks


class SupabaseVectorStore:
    """
    Store and retrieve embeddings from Supabase pgvector.
    
    Required Supabase Setup:
    ═══════════════════════
    
    Run this SQL in your Supabase SQL Editor:
    
    ```sql
    -- Enable the vector extension
    create extension if not exists vector with schema extensions;
    
    -- Create the documents table
    create table documents (
      id bigint primary key generated always as identity,
      content text not null,
      source text,
      section text,
      metadata jsonb default '{}'::jsonb,
      embedding extensions.vector(384)
    );
    
    -- Create an HNSW index for fast similarity search
    create index on documents using hnsw (embedding vector_cosine_ops);
    
    -- Enable Row Level Security (optional but recommended)
    alter table documents enable row level security;
    
    -- Create a policy to allow all operations (adjust for production)
    create policy "Allow all" on documents for all using (true);
    
    -- Create a function for similarity search
    create or replace function match_documents (
      query_embedding extensions.vector(384),
      match_threshold float default 0.5,
      match_count int default 5
    )
    returns table (
      id bigint,
      content text,
      source text,
      section text,
      metadata jsonb,
      similarity float
    )
    language sql stable
    as $$
      select
        documents.id,
        documents.content,
        documents.source,
        documents.section,
        documents.metadata,
        1 - (documents.embedding <=> query_embedding) as similarity
      from documents
      where 1 - (documents.embedding <=> query_embedding) > match_threshold
      order by documents.embedding <=> query_embedding
      limit match_count;
    $$;
    ```
    """
    
    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
        table_name: str = "documents",
    ):
        """
        Initialize connection to Supabase.
        
        Args:
            url: Supabase project URL (or set SUPABASE_URL env var)
            key: Supabase service key (or set SUPABASE_KEY env var)
            table_name: Name of the table to store documents
        """
        load_dotenv()
        
        self.url = url or os.getenv("SUPABASE_URL")
        self.key = key or os.getenv("SUPABASE_KEY")
        self.table_name = table_name
        
        if not self.url or not self.key:
            raise ValueError(
                "Supabase credentials not found!\n"
                "Set SUPABASE_URL and SUPABASE_KEY environment variables,\n"
                "or pass them directly to the constructor."
            )
        
        self.client: Client = create_client(self.url, self.key)
        logger.info(f"Connected to Supabase: {self.url[:50]}...")
    
    def insert(self, chunk: EmbeddedChunk) -> dict:
        """
        Insert a single embedded chunk into the database.
        
        Args:
            chunk: EmbeddedChunk with content and embedding
            
        Returns:
            Inserted record data
        """
        data = {
            "content": chunk.content,
            "source": chunk.source,
            "section": chunk.section,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding,
        }
        
        result = self.client.table(self.table_name).insert(data).execute()
        return result.data[0] if result.data else {}
    
    def insert_batch(
        self, 
        chunks: list[EmbeddedChunk],
        batch_size: int = 100,
    ) -> int:
        """
        Insert multiple embedded chunks efficiently.
        
        Args:
            chunks: List of EmbeddedChunk objects
            batch_size: Number of records per batch insert
            
        Returns:
            Number of records inserted
        """
        total_inserted = 0
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            data = [
                {
                    "content": chunk.content,
                    "source": chunk.source,
                    "section": chunk.section,
                    "metadata": chunk.metadata,
                    "embedding": chunk.embedding,
                }
                for chunk in batch
            ]
            
            result = self.client.table(self.table_name).insert(data).execute()
            total_inserted += len(result.data) if result.data else 0
            
            logger.info(f"Inserted batch {i//batch_size + 1}: {len(batch)} records")
        
        logger.info(f"Total inserted: {total_inserted} records")
        return total_inserted
    
    def search(
        self,
        query_embedding: list[float],
        match_threshold: float = 0.5,
        match_count: int = 5,
    ) -> list[dict]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query_embedding: Embedding vector to search for
            match_threshold: Minimum similarity threshold (0-1)
            match_count: Maximum number of results
            
        Returns:
            List of matching documents with similarity scores
        """
        result = self.client.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_threshold": match_threshold,
                "match_count": match_count,
            }
        ).execute()
        
        return result.data if result.data else []
    
    def delete_by_source(self, source: str) -> int:
        """
        Delete all documents from a specific source.
        
        Args:
            source: Source identifier to delete
            
        Returns:
            Number of records deleted
        """
        result = self.client.table(self.table_name).delete().eq("source", source).execute()
        count = len(result.data) if result.data else 0
        logger.info(f"Deleted {count} records from source: {source}")
        return count
    
    def count(self) -> int:
        """Get total number of documents in the store."""
        result = self.client.table(self.table_name).select("id", count="exact").execute()
        return result.count if result.count else 0
    
    def clear(self) -> int:
        """
        Delete all documents from the store.
        
        Returns:
            Number of records deleted
        """
        # Supabase doesn't support DELETE without WHERE, so we use a workaround
        result = self.client.table(self.table_name).delete().neq("id", -1).execute()
        count = len(result.data) if result.data else 0
        logger.info(f"Cleared {count} records from {self.table_name}")
        return count


class EmbeddingPipeline:
    """
    Complete pipeline for processing documents to embeddings in Supabase.
    
    Usage:
        pipeline = EmbeddingPipeline()
        pipeline.process_and_store(chunks)
        results = pipeline.search("your query")
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
    ):
        """
        Initialize the embedding pipeline.
        
        Args:
            embedding_model: Sentence transformer model name
            supabase_url: Supabase project URL
            supabase_key: Supabase service key
        """
        self.embedder = EmbeddingGenerator(model_name=embedding_model)
        self.store = SupabaseVectorStore(url=supabase_url, key=supabase_key)
        
        # Verify dimensions match
        expected_dim = 384  # Default for all-MiniLM-L6-v2
        if self.embedder.dimension != expected_dim:
            logger.warning(
                f"Embedding dimension ({self.embedder.dimension}) differs from "
                f"expected ({expected_dim}). Update your Supabase schema!"
            )
    
    def process_and_store(
        self,
        chunks: list[DocumentChunk],
        batch_size: int = 32,
    ) -> int:
        """
        Embed chunks and store them in Supabase.
        
        Args:
            chunks: List of document chunks to process
            batch_size: Batch size for embedding and insertion
            
        Returns:
            Number of records inserted
        """
        logger.info(f"Processing {len(chunks)} chunks")
        
        # Generate embeddings
        embedded_chunks = self.embedder.embed_chunks(chunks, batch_size=batch_size)
        
        # Store in Supabase
        inserted = self.store.insert_batch(embedded_chunks, batch_size=batch_size)
        
        return inserted
    
    def search(
        self,
        query: str,
        match_count: int = 5,
        match_threshold: float = 0.5,
    ) -> list[dict]:
        """
        Search for documents similar to a query.
        
        Args:
            query: Search query text
            match_count: Number of results to return
            match_threshold: Minimum similarity threshold
            
        Returns:
            List of matching documents with similarity scores
        """
        # Embed the query
        query_embedding = self.embedder.embed(query)
        
        # Search in Supabase
        results = self.store.search(
            query_embedding=query_embedding,
            match_threshold=match_threshold,
            match_count=match_count,
        )
        
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Interface
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """CLI for testing embeddings."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate and store embeddings")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("-n", "--count", type=int, default=5, help="Number of results")
    search_parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Similarity threshold")
    
    # Count command
    subparsers.add_parser("count", help="Count documents in store")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test embedding generation")
    test_parser.add_argument("text", help="Text to embed")
    
    args = parser.parse_args()
    
    if args.command == "test":
        embedder = EmbeddingGenerator()
        embedding = embedder.embed(args.text)
        print(f"Text: {args.text}")
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First 10 values: {embedding[:10]}")
    
    elif args.command == "search":
        pipeline = EmbeddingPipeline()
        results = pipeline.search(args.query, match_count=args.count, match_threshold=args.threshold)
        
        print(f"\nSearch results for: '{args.query}'")
        print("=" * 60)
        
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] Similarity: {result['similarity']:.4f}")
            print(f"    Source: {result['source']}")
            print(f"    Section: {result.get('section', 'N/A')}")
            print(f"    Content: {result['content'][:200]}...")
    
    elif args.command == "count":
        store = SupabaseVectorStore()
        count = store.count()
        print(f"Documents in store: {count}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

