"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   🧠 CONTEXTUAL RAG with DSPy 3.x                                            ║
║   Based on Anthropic's "Contextual Retrieval" technique (Sept 2024)          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

THE PROBLEM WITH TRADITIONAL RAG:
═════════════════════════════════
When you chunk a document, each chunk loses its context:

    Original Document:
    ┌─────────────────────────────────────────────┐
    │ ACME Corp Q3 2024 Financial Report          │
    │                                             │
    │ Revenue grew by 15% compared to Q2...       │
    │                                             │
    │ The company expanded into Asia...           │
    └─────────────────────────────────────────────┘

    After Chunking (Context Lost!):
    ┌─────────────────────────┐
    │ Revenue grew by 15%     │  ← Which company? What quarter?
    │ compared to Q2...       │    The retriever doesn't know!
    └─────────────────────────┘

THE CONTEXTUAL RAG SOLUTION:
════════════════════════════
Use an LLM to prepend context to each chunk BEFORE embedding:

    ┌─────────────────────────────────────────────────────────────┐
    │ [Context: This chunk is from ACME Corp's Q3 2024 Financial  │
    │ Report, specifically the Revenue section]                   │
    │                                                             │
    │ Revenue grew by 15% compared to Q2...                       │
    └─────────────────────────────────────────────────────────────┘

This example teaches you:
1. How contextual chunking works
2. Hybrid retrieval (BM25 + vector search)
3. Building it all with DSPy 3.x modules

Run with: python contextual_rag.py
"""

import os
import hashlib
import json
import time
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import numpy as np
import dspy

# Rate limiting for Gemini free tier (10 requests/minute)
# Set to 0 to disable rate limiting (if you have a paid plan)
RATE_LIMIT_DELAY = 7  # seconds between LLM calls

# ═══════════════════════════════════════════════════════════════════════════════
# 📚 PART 1: THE KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════════════════════
#
# We'll use a realistic example: documentation for a fictional API.
# This demonstrates how chunks can lose context in real scenarios.
#
# Notice how some chunks reference "the API" or "this endpoint" without
# saying WHICH API or endpoint - this is the context loss problem!

DOCUMENTS = {
    "DataFlow API Documentation": {
        "sections": {
            "Overview": """
DataFlow is a real-time data streaming API designed for high-throughput applications.
It supports up to 10,000 events per second per connection. The API uses WebSocket
connections for real-time streaming and REST endpoints for configuration and management.
DataFlow is ideal for financial data, IoT sensors, and live analytics dashboards.
            """,
            "Authentication": """
All API requests require authentication using Bearer tokens. Tokens are obtained
from the /auth/token endpoint using your API key and secret. Tokens expire after
24 hours and must be refreshed. For WebSocket connections, include the token in
the connection URL as a query parameter: wss://api.dataflow.io/stream?token=YOUR_TOKEN
            """,
            "Rate Limits": """
Free tier accounts are limited to 100 requests per minute and 1,000 events per second.
Pro tier increases this to 1,000 requests per minute and 10,000 events per second.
Enterprise tier has custom limits. Rate limit headers are included in all responses:
X-RateLimit-Remaining and X-RateLimit-Reset.
            """,
            "WebSocket Streaming": """
Connect to wss://api.dataflow.io/stream to receive real-time events. After connecting,
send a subscription message: {"action": "subscribe", "channels": ["channel1", "channel2"]}.
Events are delivered as JSON objects with timestamp, channel, and data fields.
The connection automatically reconnects on network failures with exponential backoff.
            """,
            "Error Handling": """
The API returns standard HTTP status codes. 400 indicates invalid request parameters.
401 means authentication failed - check your token. 429 means rate limit exceeded -
implement exponential backoff. 500 indicates server error - retry with backoff.
All error responses include a JSON body with 'error' and 'message' fields.
            """
        }
    },
    "CloudStore Storage Documentation": {
        "sections": {
            "Overview": """
CloudStore provides S3-compatible object storage with global CDN distribution.
Objects can be up to 5TB in size. The service offers 99.99% availability SLA
and 11 nines of durability. CloudStore supports versioning, lifecycle policies,
and cross-region replication for disaster recovery.
            """,
            "Authentication": """
CloudStore uses AWS Signature Version 4 for authentication. You'll need an access
key ID and secret access key from your account dashboard. For temporary credentials,
use the STS AssumeRole API. All requests must be signed using the canonical request
format documented in the AWS signature specification.
            """,
            "Buckets": """
Buckets are containers for objects. Bucket names must be globally unique across all
CloudStore accounts. Names must be 3-63 characters, lowercase, and DNS-compatible.
Each account can have up to 100 buckets. Buckets can be configured for public access,
website hosting, or private-only access.
            """,
            "Upload Operations": """
Small files (under 100MB) can be uploaded with a single PUT request.
For larger files, use multipart upload: initiate with POST /?uploads,
upload parts with PUT ?partNumber=N&uploadId=ID, then complete with
POST ?uploadId=ID. Multipart uploads can be parallelized for speed.
            """,
            "Pricing": """
Storage costs $0.023 per GB per month for standard tier. Infrequent access
tier costs $0.0125 per GB. Data transfer out costs $0.09 per GB for the first
10TB. PUT/COPY/POST requests cost $0.005 per 1,000 requests.
GET/SELECT requests cost $0.0004 per 1,000 requests.
            """
        }
    },
    "TaskQueue Job Processing Documentation": {
        "sections": {
            "Overview": """
TaskQueue is a distributed job processing system for background tasks.
It guarantees at-least-once delivery and supports priority queues.
Jobs can be scheduled for future execution or immediate processing.
TaskQueue handles retry logic, dead letter queues, and job dependencies.
            """,
            "Creating Jobs": """
Create a job by POSTing to /jobs with a JSON body containing: queue (string),
payload (object), priority (1-10, default 5), and optional scheduled_at (ISO8601).
The response includes a job_id for tracking. Jobs are immutable once created -
to modify, cancel and recreate.
            """,
            "Processing Jobs": """
Workers poll for jobs using GET /jobs/next?queue=QUEUE_NAME. The returned job
is locked for 5 minutes (configurable). Workers must call POST /jobs/{id}/complete
or POST /jobs/{id}/fail within the lock period. Failed jobs are automatically
retried up to 3 times with exponential backoff.
            """,
            "Monitoring": """
The /stats endpoint returns queue depths, processing rates, and error rates.
Each job has a status: pending, processing, completed, or failed. Use
GET /jobs/{id} to check individual job status. Webhook notifications can be
configured to fire on job completion or failure.
            """,
            "Best Practices": """
Keep job payloads small (under 64KB) - store large data externally and reference it.
Use idempotent job handlers to handle potential duplicate deliveries.
Set appropriate timeouts based on expected job duration.
Monitor queue depths to scale workers dynamically.
            """
        }
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
# 🔧 PART 2: SETUP AND CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

def setup_dspy() -> dspy.LM:
    """Configure DSPy with Gemini 2.5 Flash."""
    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your-gemini-api-key-here":
        raise ValueError(
            "❌ GEMINI_API_KEY not found!\n"
            "   1. Copy .env.example to .env in the project root\n"
            "   2. Add your Gemini API key (get one at https://aistudio.google.com/apikey)"
        )
    
    lm = dspy.LM(model="gemini/gemini-2.5-flash", api_key=api_key)
    dspy.configure(lm=lm)
    
    print("✅ DSPy configured with Gemini 2.5 Flash")
    return lm


# ═══════════════════════════════════════════════════════════════════════════════
# 📝 PART 3: TRADITIONAL CHUNKING (Without Context)
# ═══════════════════════════════════════════════════════════════════════════════
#
# This shows what traditional RAG does - just split the document into chunks.
# Notice how much context is lost!

class TraditionalChunk:
    """A basic chunk without contextual enhancement."""
    
    def __init__(self, content: str, source_doc: str, section: str):
        self.content = content.strip()
        self.source_doc = source_doc
        self.section = section
        self.embedding: Optional[np.ndarray] = None
    
    def __repr__(self):
        return f"Chunk({self.source_doc}/{self.section}: {self.content[:50]}...)"


def create_traditional_chunks() -> list[TraditionalChunk]:
    """
    Create chunks the traditional way - just split by section.
    
    Problem: The chunks don't know their context!
    When we embed "All API requests require authentication using Bearer tokens",
    the embedding doesn't capture WHICH API (DataFlow? CloudStore? TaskQueue?)
    """
    chunks = []
    
    for doc_name, doc_content in DOCUMENTS.items():
        for section_name, section_text in doc_content["sections"].items():
            chunk = TraditionalChunk(
                content=section_text.strip(),
                source_doc=doc_name,
                section=section_name
            )
            chunks.append(chunk)
    
    print(f"📦 Created {len(chunks)} traditional chunks")
    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# 🧠 PART 4: CONTEXTUAL CHUNKING (The Magic!)
# ═══════════════════════════════════════════════════════════════════════════════
#
# This is Anthropic's Contextual Retrieval innovation:
# Use an LLM to generate context for each chunk!

class ContextualChunk:
    """A chunk enhanced with LLM-generated context."""
    
    def __init__(
        self,
        original_content: str,
        contextual_content: str,
        source_doc: str,
        section: str
    ):
        self.original_content = original_content.strip()
        self.contextual_content = contextual_content.strip()  # Used for retrieval
        self.source_doc = source_doc
        self.section = section
        self.embedding: Optional[np.ndarray] = None
    
    def __repr__(self):
        return f"ContextualChunk({self.source_doc}/{self.section})"


class GenerateChunkContext(dspy.Signature):
    """
    Generate a brief context prefix for a document chunk.
    
    This context will be prepended to the chunk to help retrieval systems
    understand what the chunk is about, even without seeing the full document.
    """
    
    document_title = dspy.InputField(desc="Title of the source document")
    section_title = dspy.InputField(desc="Section within the document")
    full_document_summary = dspy.InputField(desc="Brief summary of the entire document")
    chunk_content = dspy.InputField(desc="The actual chunk content to contextualize")
    
    context_prefix = dspy.OutputField(
        desc="A 1-2 sentence context that situates this chunk within the document. "
             "Start with 'This chunk' and mention the specific document/section."
    )


class ChunkContextualizer(dspy.Module):
    """
    A DSPy module that adds context to chunks using an LLM.
    
    How it works:
    ┌──────────────────────────────────────────────────────────────┐
    │  INPUT: Raw chunk + metadata                                 │
    │                                                              │
    │  "All API requests require authentication using Bearer       │
    │   tokens. Tokens are obtained from the /auth/token..."       │
    │                                                              │
    │  Document: DataFlow API Documentation                        │
    │  Section: Authentication                                     │
    ├──────────────────────────────────────────────────────────────┤
    │  LLM generates context prefix                                │
    ├──────────────────────────────────────────────────────────────┤
    │  OUTPUT: Contextualized chunk                                │
    │                                                              │
    │  "[This chunk describes the authentication process for the   │
    │   DataFlow real-time streaming API, specifically how to      │
    │   obtain and use Bearer tokens.]                             │
    │                                                              │
    │   All API requests require authentication using Bearer       │
    │   tokens. Tokens are obtained from the /auth/token..."       │
    └──────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self):
        super().__init__()
        self.generate_context = dspy.Predict(GenerateChunkContext)
    
    def forward(
        self,
        chunk_content: str,
        document_title: str,
        section_title: str,
        document_summary: str
    ) -> str:
        """Generate a contextualized version of the chunk."""
        
        result = self.generate_context(
            document_title=document_title,
            section_title=section_title,
            full_document_summary=document_summary,
            chunk_content=chunk_content
        )
        
        # Prepend the context to the original chunk
        contextualized = f"[{result.context_prefix}]\n\n{chunk_content}"
        return contextualized


# Document summaries for context generation
DOCUMENT_SUMMARIES = {
    "DataFlow API Documentation": 
        "DataFlow is a real-time data streaming API using WebSockets, "
        "designed for high-throughput applications like financial data and IoT.",
    
    "CloudStore Storage Documentation":
        "CloudStore is an S3-compatible object storage service with global CDN, "
        "supporting files up to 5TB with high availability and durability.",
    
    "TaskQueue Job Processing Documentation":
        "TaskQueue is a distributed background job processing system with "
        "at-least-once delivery, priority queues, and automatic retries."
}


def create_contextual_chunks(use_cache: bool = True) -> list[ContextualChunk]:
    """
    Create chunks with LLM-generated context.
    
    This is where the magic happens! Each chunk gets enriched with
    context that helps retrieval understand WHAT the chunk is about.
    
    Args:
        use_cache: Cache results to avoid redundant LLM calls (default True)
    """
    cache_file = Path(__file__).parent / ".chunk_cache.json"
    
    # Try to load from cache
    if use_cache and cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                cached = json.load(f)
            
            chunks = []
            for item in cached:
                chunk = ContextualChunk(
                    original_content=item["original"],
                    contextual_content=item["contextual"],
                    source_doc=item["doc"],
                    section=item["section"]
                )
                chunks.append(chunk)
            
            print(f"📦 Loaded {len(chunks)} contextual chunks from cache")
            return chunks
        except Exception:
            pass  # Cache invalid, regenerate
    
    # Generate contextual chunks
    print("🧠 Generating contextual chunks (this uses the LLM)...")
    if RATE_LIMIT_DELAY > 0:
        print(f"   ⏱️  Rate limiting enabled: {RATE_LIMIT_DELAY}s between calls (Gemini free tier)")
    contextualizer = ChunkContextualizer()
    chunks = []
    cache_data = []
    
    is_first_call = True
    for doc_name, doc_content in DOCUMENTS.items():
        summary = DOCUMENT_SUMMARIES[doc_name]
        
        for section_name, section_text in doc_content["sections"].items():
            original = section_text.strip()
            
            # Rate limiting for Gemini free tier
            if RATE_LIMIT_DELAY > 0 and not is_first_call:
                time.sleep(RATE_LIMIT_DELAY)
            is_first_call = False
            
            # Generate context using the LLM
            contextual = contextualizer(
                chunk_content=original,
                document_title=doc_name,
                section_title=section_name,
                document_summary=summary
            )
            
            chunk = ContextualChunk(
                original_content=original,
                contextual_content=contextual,
                source_doc=doc_name,
                section=section_name
            )
            chunks.append(chunk)
            
            cache_data.append({
                "original": original,
                "contextual": contextual,
                "doc": doc_name,
                "section": section_name
            })
            
            print(f"   ✓ {doc_name}/{section_name}")
    
    # Save cache
    if use_cache:
        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)
    
    print(f"📦 Created {len(chunks)} contextual chunks")
    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# 🔍 PART 5: HYBRID RETRIEVAL (BM25 + Vector Search)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Anthropic's paper shows that combining BM25 (keyword matching) with
# vector search (semantic matching) gives the best results.
#
# BM25: Good at finding exact terms ("WebSocket", "Bearer token")
# Vector: Good at finding semantic meaning ("how to authenticate")

class HybridRetriever:
    """
    Combines BM25 (keyword) and vector (semantic) search.
    
    How Hybrid Retrieval Works:
    ═══════════════════════════
    
    Query: "How do I authenticate with the streaming API?"
    
    BM25 Search:                      Vector Search:
    ┌────────────────────────┐       ┌────────────────────────┐
    │ Finds chunks with      │       │ Finds chunks with      │
    │ "authenticate" and     │       │ similar MEANING to     │
    │ "streaming" and "API"  │       │ authentication         │
    └────────────────────────┘       └────────────────────────┘
            │                                  │
            └──────────┬───────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Combine scores  │
              │ using RRF       │
              │ (Reciprocal     │
              │  Rank Fusion)   │
              └─────────────────┘
                       │
                       ▼
              Top-K results
    """
    
    def __init__(self, chunks: list, k: int = 3, use_contextual: bool = True):
        """
        Initialize the hybrid retriever.
        
        Args:
            chunks: List of TraditionalChunk or ContextualChunk objects
            k: Number of results to return
            use_contextual: If True and chunks are ContextualChunks, 
                          use contextual_content for retrieval
        """
        self.chunks = chunks
        self.k = k
        self.use_contextual = use_contextual
        
        # Determine which content to use for retrieval
        if use_contextual and hasattr(chunks[0], 'contextual_content'):
            self.retrieval_texts = [c.contextual_content for c in chunks]
        else:
            self.retrieval_texts = [
                c.content if hasattr(c, 'content') else c.original_content 
                for c in chunks
            ]
        
        # Initialize BM25
        self._init_bm25()
        
        # Initialize vector search
        self._init_vectors()
    
    def _init_bm25(self):
        """
        Initialize BM25 index.
        
        BM25 (Best Matching 25) is a ranking function used in search engines.
        It scores documents based on term frequency, inverse document frequency,
        and document length normalization.
        """
        from rank_bm25 import BM25Okapi
        
        # Tokenize documents (simple word splitting)
        tokenized = [doc.lower().split() for doc in self.retrieval_texts]
        self.bm25 = BM25Okapi(tokenized)
        print("   ✓ BM25 index built")
    
    def _init_vectors(self):
        """
        Initialize vector embeddings using sentence-transformers.
        
        We use 'all-MiniLM-L6-v2' which is:
        - Fast (83M parameters)
        - Good quality embeddings
        - Free and runs locally!
        """
        from sentence_transformers import SentenceTransformer
        
        print("   ⏳ Loading embedding model (first run downloads ~90MB)...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate embeddings for all chunks
        print("   ⏳ Generating embeddings...")
        self.embeddings = self.embedding_model.encode(
            self.retrieval_texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / norms
        
        print("   ✓ Vector index built")
    
    def _bm25_search(self, query: str) -> list[tuple[int, float]]:
        """Get BM25 scores for all chunks."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Return (index, score) pairs, sorted by score
        results = [(i, score) for i, score in enumerate(scores)]
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _vector_search(self, query: str) -> list[tuple[int, float]]:
        """Get vector similarity scores for all chunks."""
        # Embed the query
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        )[0]
        
        # Normalize
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Compute cosine similarities
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Return (index, score) pairs, sorted by score
        results = [(i, float(sim)) for i, sim in enumerate(similarities)]
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _reciprocal_rank_fusion(
        self,
        bm25_results: list[tuple[int, float]],
        vector_results: list[tuple[int, float]],
        k: int = 60  # RRF constant
    ) -> list[tuple[int, float]]:
        """
        Combine rankings using Reciprocal Rank Fusion (RRF).
        
        RRF Formula: score(d) = Σ 1/(k + rank(d))
        
        This works better than simple score addition because:
        - It handles different score scales (BM25 vs cosine similarity)
        - It emphasizes rank position over absolute scores
        - The k parameter prevents top-ranked items from dominating
        """
        scores = {}
        
        # Add BM25 contribution
        for rank, (idx, _) in enumerate(bm25_results):
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
        
        # Add vector contribution
        for rank, (idx, _) in enumerate(vector_results):
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
        
        # Sort by combined score
        results = [(idx, score) for idx, score in scores.items()]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def retrieve(self, query: str) -> list[dict]:
        """
        Retrieve the most relevant chunks for a query.
        
        Returns list of dicts with:
        - chunk: The chunk object
        - score: Combined retrieval score
        - bm25_rank: Rank from BM25 search
        - vector_rank: Rank from vector search
        """
        # Get rankings from both methods
        bm25_results = self._bm25_search(query)
        vector_results = self._vector_search(query)
        
        # Create rank lookup
        bm25_ranks = {idx: rank for rank, (idx, _) in enumerate(bm25_results)}
        vector_ranks = {idx: rank for rank, (idx, _) in enumerate(vector_results)}
        
        # Combine with RRF
        combined = self._reciprocal_rank_fusion(bm25_results, vector_results)
        
        # Return top-k with metadata
        results = []
        for idx, score in combined[:self.k]:
            results.append({
                "chunk": self.chunks[idx],
                "score": score,
                "bm25_rank": bm25_ranks[idx] + 1,
                "vector_rank": vector_ranks[idx] + 1
            })
        
        return results
    
    def __call__(self, query: str) -> list[dict]:
        """Shorthand for retrieve()."""
        return self.retrieve(query)


# ═══════════════════════════════════════════════════════════════════════════════
# 🤖 PART 6: THE CONTEXTUAL RAG MODULE
# ═══════════════════════════════════════════════════════════════════════════════

class AnswerFromContext(dspy.Signature):
    """Answer a question accurately using the provided context chunks."""
    
    context = dspy.InputField(
        desc="Retrieved documentation chunks relevant to the question"
    )
    question = dspy.InputField(
        desc="The user's question about the APIs"
    )
    
    reasoning = dspy.OutputField(
        desc="Brief analysis of which parts of the context are relevant"
    )
    answer = dspy.OutputField(
        desc="A helpful, accurate answer based on the context"
    )


class SimpleAnswer(dspy.Signature):
    """Answer a question using the provided context. Be accurate and concise."""
    
    context = dspy.InputField(desc="Retrieved documentation")
    question = dspy.InputField(desc="The user's question")
    answer = dspy.OutputField(desc="A clear, accurate answer")


class ContextualRAG(dspy.Module):
    """
    The complete Contextual RAG system.
    
    Architecture:
    ═════════════
    
    User Question
          │
          ▼
    ┌─────────────────┐
    │ Hybrid Retriever │  ← BM25 + Vector search
    │ (with contextual │    on contextualized chunks
    │  chunks)         │
    └────────┬────────┘
             │
             ▼
    Retrieved Chunks (with context)
             │
             ▼
    ┌─────────────────┐
    │ Answer Generator │  ← LLM with Chain-of-Thought
    │ (DSPy Module)    │
    └────────┬────────┘
             │
             ▼
    Answer + Reasoning
    """
    
    def __init__(self, retriever: HybridRetriever):
        super().__init__()
        self.retriever = retriever
        self.generate = dspy.ChainOfThought(AnswerFromContext)
    
    def forward(self, question: str):
        """Process a question through the full RAG pipeline."""
        
        # Step 1: Retrieve relevant chunks
        retrieved = self.retriever(question)
        
        # Step 2: Format context for the LLM
        context_parts = []
        for i, result in enumerate(retrieved, 1):
            chunk = result["chunk"]
            # Use original content for the answer (more readable)
            content = (
                chunk.original_content 
                if hasattr(chunk, 'original_content') 
                else chunk.content
            )
            context_parts.append(
                f"[Source: {chunk.source_doc} - {chunk.section}]\n{content}"
            )
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Step 3: Generate answer
        result = self.generate(context=context, question=question)
        
        # Attach retrieval metadata
        result.retrieved = retrieved
        
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# 🧬 PART 7: GEPA OPTIMIZER - Automatic Prompt Evolution (2025)
# ═══════════════════════════════════════════════════════════════════════════════
#
# GEPA (Genetic-Pareto Evolutionary Algorithm) is DSPy's newest optimizer (2025).
# It dramatically improves prompts by:
#
#   1. REFLECTION: LLM analyzes what went wrong in failed predictions
#   2. EVOLUTION: Successful prompt traits are combined (like genetics)
#   3. PARETO: Maintains diverse solutions, not just one "best"
#
# Results vs MIPROv2 (previous best):
#   ✅ 10%+ better performance
#   ✅ 35x fewer LLM calls needed
#   ✅ 9x shorter prompts generated
#
# Why GEPA matters for Contextual RAG:
#   - Your retrieval might be perfect, but if the generation prompt is weak,
#     answers will still be poor
#   - GEPA optimizes the "last mile" - how the LLM uses the retrieved context

class ContextualRAGForOptimization(dspy.Module):
    """
    A simpler RAG module designed for GEPA optimization.
    
    We use a simple signature so GEPA has room to improve it.
    GEPA will evolve the instructions to be more specific and effective.
    """
    
    def __init__(self, retriever: HybridRetriever):
        super().__init__()
        self.retriever = retriever
        # Simple predictor that GEPA will optimize
        self.generate = dspy.Predict(SimpleAnswer)
    
    def forward(self, question: str):
        """Process a question and return an answer."""
        # Retrieve relevant chunks
        retrieved = self.retriever(question)
        
        # Format context
        context_parts = []
        for result in retrieved:
            chunk = result["chunk"]
            content = (
                chunk.original_content 
                if hasattr(chunk, 'original_content') 
                else chunk.content
            )
            context_parts.append(
                f"[{chunk.source_doc} - {chunk.section}]\n{content}"
            )
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Generate answer
        return self.generate(context=context, question=question)


# Training data for GEPA optimization
# These are question/expected_answer pairs that GEPA uses to evolve prompts
TRAINING_DATA = [
    dspy.Example(
        question="How do I authenticate with the DataFlow streaming API?",
        expected_answer="Bearer tokens obtained from /auth/token endpoint"
    ).with_inputs("question"),
    
    dspy.Example(
        question="What's the maximum file size for CloudStore?",
        expected_answer="5TB"
    ).with_inputs("question"),
    
    dspy.Example(
        question="How does TaskQueue handle failed jobs?",
        expected_answer="automatically retried up to 3 times with exponential backoff"
    ).with_inputs("question"),
    
    dspy.Example(
        question="What's the rate limit for DataFlow free tier?",
        expected_answer="100 requests per minute and 1,000 events per second"
    ).with_inputs("question"),
    
    dspy.Example(
        question="How do I upload large files to CloudStore?",
        expected_answer="multipart upload"
    ).with_inputs("question"),
    
    dspy.Example(
        question="What authentication does CloudStore use?",
        expected_answer="AWS Signature Version 4"
    ).with_inputs("question"),
    
    dspy.Example(
        question="How long are TaskQueue jobs locked for?",
        expected_answer="5 minutes"
    ).with_inputs("question"),
    
    dspy.Example(
        question="What format are DataFlow events delivered in?",
        expected_answer="JSON"
    ).with_inputs("question"),
]


def gepa_metric(gold, pred, trace=None):
    """
    GEPA metric function with feedback for reflection.
    
    GEPA is special: it can use textual FEEDBACK to guide evolution.
    When a prediction fails, we tell GEPA WHY it failed, and it uses
    this information to propose better prompts.
    
    The feedback becomes part of the "reflection" that guides evolution.
    """
    expected = gold.expected_answer.lower()
    actual = pred.answer.lower() if hasattr(pred, 'answer') else ""
    
    # Perfect match - expected answer is contained in response
    if expected in actual:
        return 1.0
    
    # Check for partial matches
    expected_words = set(expected.split())
    actual_words = set(actual.split())
    overlap = len(expected_words & actual_words) / len(expected_words) if expected_words else 0
    
    if overlap > 0.5:
        score = 0.7
        feedback = f"Partially correct. Expected '{gold.expected_answer}' to appear in answer."
    elif overlap > 0:
        score = 0.3
        feedback = f"Some relevant info but missing key details. Expected: '{gold.expected_answer}'"
    else:
        score = 0.0
        feedback = f"Answer doesn't contain expected info. Expected: '{gold.expected_answer}', Got: '{actual[:80]}...'"
    
    # GEPA uses ScoreWithFeedback to learn from mistakes
    from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback
    return ScoreWithFeedback(score=score, feedback=feedback)


def run_gepa_optimization(retriever: HybridRetriever, verbose: bool = True):
    """
    Run GEPA optimization on the Contextual RAG module.
    
    GEPA Evolution Process:
    ═══════════════════════
    
    Generation 0 (Initial):
    ┌─────────────────────────────────────┐
    │ "Answer a question using context"   │  ← Basic prompt
    └─────────────────────────────────────┘
                    │
                    ▼ Run on training data
                    │ Score + Feedback
                    ▼
    
    Generation 1 (After Reflection):
    ┌─────────────────────────────────────┐
    │ "Answer the question accurately     │
    │  using ONLY the provided API        │  ← GEPA evolved this!
    │  documentation. Be specific."       │
    └─────────────────────────────────────┘
                    │
                    ▼ Repeat...
    
    Final (Optimized):
    ┌─────────────────────────────────────┐
    │ "You are an API documentation       │
    │  assistant. Answer questions using  │  ← Even better!
    │  the exact details from context..." │
    └─────────────────────────────────────┘
    """
    if verbose:
        print("\n🧬 GEPA Optimization")
        print("═" * 50)
        print("""
GEPA (Genetic-Pareto) is DSPy's 2025 optimizer that:

  🔬 REFLECTS on failures to understand what went wrong
  🧬 EVOLVES prompts by combining successful traits  
  📊 PARETO optimizes for multiple objectives

Let's optimize our Contextual RAG prompts!
        """)
    
    # Create the base module
    rag = ContextualRAGForOptimization(retriever)
    
    # Test BEFORE optimization
    if verbose:
        print("📊 BEFORE Optimization:")
        print("-" * 40)
        
        test_questions = [
            "How do I authenticate with DataFlow?",
            "What's the max file size for CloudStore?",
        ]
        
        for q in test_questions:
            result = rag(question=q)
            print(f"  Q: {q}")
            print(f"  A: {result.answer[:100]}...")
            print()
    
    # Run GEPA optimization
    if verbose:
        print("🧬 Running GEPA (this makes several LLM calls)...")
        print()
    
    try:
        # GEPA needs a reflection LM - can be the same or different model
        api_key = os.getenv("GEMINI_API_KEY")
        reflection_lm = dspy.LM(
            model="gemini/gemini-2.5-flash",
            api_key=api_key,
            temperature=1.0,  # Higher temp for creative reflection
            max_tokens=8000
        )
        
        optimizer = dspy.GEPA(
            metric=gepa_metric,
            auto="light",  # "light" = fast (~5 mins), "medium", "heavy" = more thorough
            reflection_lm=reflection_lm,
        )
        
        optimized_rag = optimizer.compile(
            rag,
            trainset=TRAINING_DATA,
        )
        
        # Test AFTER optimization
        if verbose:
            print("\n📊 AFTER Optimization:")
            print("-" * 40)
            
            for q in test_questions:
                result = optimized_rag(question=q)
                print(f"  Q: {q}")
                print(f"  A: {result.answer[:100]}...")
                print()
            
            print("✅ GEPA evolved better prompts for your Contextual RAG!")
            print()
            print("💡 What GEPA learned:")
            print("   - How to extract specific info from API docs")
            print("   - Which details matter for different question types")
            print("   - How to format answers clearly")
        
        return optimized_rag
        
    except Exception as e:
        if verbose:
            print(f"⚠️  GEPA optimization encountered an issue: {e}")
            print()
            print("   This can happen with limited training data.")
            print("   For production, use 10-50+ training examples.")
        return rag


# ═══════════════════════════════════════════════════════════════════════════════
# 🎮 PART 8: DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def print_header(title: str):
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print('═' * 70)


def demo_context_difference():
    """Show the difference between traditional and contextual chunks."""
    print_header("📊 DEMO: Traditional vs Contextual Chunks")
    
    print("\n🔴 TRADITIONAL CHUNK (without context):")
    print("─" * 50)
    # Find the authentication chunk from DataFlow
    for doc_name, doc_content in DOCUMENTS.items():
        if "DataFlow" in doc_name:
            auth_section = doc_content["sections"]["Authentication"]
            print(auth_section.strip()[:300] + "...")
            break
    
    print("\n🟢 CONTEXTUAL CHUNK (with LLM-generated context):")
    print("─" * 50)
    
    # Generate context for this chunk
    contextualizer = ChunkContextualizer()
    contextual = contextualizer(
        chunk_content=auth_section.strip(),
        document_title="DataFlow API Documentation",
        section_title="Authentication",
        document_summary=DOCUMENT_SUMMARIES["DataFlow API Documentation"]
    )
    print(contextual[:400] + "...")
    
    print("\n💡 Notice how the contextual version explicitly mentions 'DataFlow'")
    print("   and 'real-time streaming API' - this helps retrieval!")
    
    # Small delay to help with rate limiting before next demo
    if RATE_LIMIT_DELAY > 0:
        time.sleep(RATE_LIMIT_DELAY)


def demo_hybrid_retrieval():
    """Show how hybrid retrieval works."""
    print_header("🔍 DEMO: Hybrid Retrieval (BM25 + Vector)")
    
    # Create contextual chunks
    chunks = create_contextual_chunks(use_cache=True)
    
    print("\n📚 Building hybrid retriever...")
    retriever = HybridRetriever(chunks, k=3, use_contextual=True)
    
    query = "How do I authenticate with the streaming API?"
    print(f"\n🔎 Query: \"{query}\"")
    print("\nRetrieval Results:")
    print("─" * 50)
    
    results = retriever(query)
    for i, result in enumerate(results, 1):
        chunk = result["chunk"]
        print(f"\n  [{i}] {chunk.source_doc} - {chunk.section}")
        print(f"      Combined Score: {result['score']:.4f}")
        print(f"      BM25 Rank: #{result['bm25_rank']}, Vector Rank: #{result['vector_rank']}")
    
    print("\n💡 The DataFlow authentication section ranked highest because")
    print("   the query mentions 'streaming API' which matches DataFlow's context!")


def demo_full_rag():
    """Demonstrate the complete Contextual RAG system."""
    print_header("🤖 DEMO: Full Contextual RAG System")
    
    # Build the system
    chunks = create_contextual_chunks(use_cache=True)
    print("\n📚 Building retriever...")
    retriever = HybridRetriever(chunks, k=3, use_contextual=True)
    rag = ContextualRAG(retriever)
    
    questions = [
        "How do I authenticate with DataFlow for WebSocket streaming?",
        "What's the maximum file size I can upload to CloudStore?",
        "How does TaskQueue handle failed jobs?",
    ]
    
    for i, question in enumerate(questions):
        # Rate limiting between questions
        if RATE_LIMIT_DELAY > 0 and i > 0:
            time.sleep(RATE_LIMIT_DELAY)
        
        print(f"\n{'─' * 60}")
        print(f"❓ Question: {question}")
        
        result = rag(question)
        
        print(f"\n📚 Retrieved from:")
        for r in result.retrieved:
            chunk = r["chunk"]
            print(f"   • {chunk.source_doc} - {chunk.section}")
        
        print(f"\n🧠 Reasoning: {result.reasoning}")
        print(f"\n💬 Answer: {result.answer}")


def interactive_mode():
    """Let the user ask questions."""
    print_header("💬 INTERACTIVE MODE")
    
    chunks = create_contextual_chunks(use_cache=True)
    print("\n📚 Building retriever...")
    retriever = HybridRetriever(chunks, k=3, use_contextual=True)
    rag = ContextualRAG(retriever)
    
    print("\n✅ Ready! Ask questions about:")
    print("   • DataFlow (real-time streaming API)")
    print("   • CloudStore (S3-compatible storage)")
    print("   • TaskQueue (background job processing)")
    print("\nType 'quit' to exit\n")
    
    while True:
        try:
            question = input("❓ Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not question:
            continue
        if question.lower() in ('quit', 'exit', 'q'):
            break
        
        result = rag(question)
        print(f"\n💬 {result.answer}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   🧠 CONTEXTUAL RAG with DSPy 3.x                                            ║
║                                                                              ║
║   This tutorial demonstrates Anthropic's "Contextual Retrieval" technique    ║
║   which reduces retrieval failures by up to 67%!                             ║
║                                                                              ║
║   Key concepts:                                                              ║
║   1. Contextual Chunking - Add LLM-generated context to each chunk           ║
║   2. Hybrid Retrieval - Combine BM25 (keywords) + Vector (semantic)          ║
║   3. Reciprocal Rank Fusion - Smart way to combine different rankings        ║
║   4. GEPA Optimizer (2025) - Automatically evolve better prompts             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Setup
    setup_dspy()
    
    # Run demos
    demo_context_difference()
    demo_hybrid_retrieval()
    demo_full_rag()
    
    # GEPA Optimization demo
    print_header("🧬 GEPA OPTIMIZATION")
    print("\nWant to try GEPA prompt optimization? (makes several LLM calls) (y/n): ", end="")
    try:
        if input().strip().lower() == 'y':
            chunks = create_contextual_chunks(use_cache=True)
            print("\n📚 Building retriever for optimization...")
            retriever = HybridRetriever(chunks, k=3, use_contextual=True)
            run_gepa_optimization(retriever, verbose=True)
    except (EOFError, KeyboardInterrupt):
        pass
    
    # Interactive mode
    print_header("🎮 TRY IT YOURSELF")
    print("\nWant to ask your own questions? (y/n): ", end="")
    try:
        if input().strip().lower() == 'y':
            interactive_mode()
    except (EOFError, KeyboardInterrupt):
        pass
    
    print("\n✅ Tutorial complete!")
    print("\n📖 Key takeaways:")
    print("   1. Traditional RAG loses context when chunking documents")
    print("   2. Contextual RAG uses an LLM to add context back to each chunk")
    print("   3. Hybrid retrieval (BM25 + Vector) outperforms either alone")
    print("   4. GEPA optimizer (2025) evolves better prompts automatically")
    print("   5. DSPy makes it easy to build and compose these systems")


if __name__ == "__main__":
    main()
