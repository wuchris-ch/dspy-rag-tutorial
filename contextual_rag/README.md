# 🧠 Contextual RAG with DSPy 3.x

This tutorial implements **Anthropic's Contextual Retrieval** technique (September 2024), which dramatically improves RAG system accuracy.

## 🎯 What is Contextual RAG?

Traditional RAG systems suffer from **context loss** when chunking documents:

```
Original Document:                    After Chunking (Context Lost!):
┌─────────────────────────────┐       ┌─────────────────────────────┐
│ ACME Corp Q3 2024 Report    │       │ Revenue grew by 15%         │
│                             │  ──►  │ compared to Q2...           │
│ Revenue grew by 15%         │       └─────────────────────────────┘
│ compared to Q2...           │           ↑
└─────────────────────────────┘           Which company? What quarter?
                                          The retriever doesn't know!
```

**Contextual RAG** fixes this by using an LLM to prepend context to each chunk:

```
┌─────────────────────────────────────────────────────────────────┐
│ [This chunk is from ACME Corp's Q3 2024 Financial Report,       │
│  specifically the Revenue section discussing quarterly growth]  │
│                                                                 │
│ Revenue grew by 15% compared to Q2...                           │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 Performance Improvements

According to Anthropic's research:

| Technique | Retrieval Failure Reduction |
|-----------|----------------------------|
| Contextual Embeddings only | 35% |
| Contextual Embeddings + BM25 | 49% |
| + Reranking | **67%** |

## 🏗️ Architecture

```
User Question
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│                    CONTEXTUAL CHUNKING                       │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │ Raw Chunk 1 │    │ Raw Chunk 2 │    │ Raw Chunk 3 │      │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘      │
│         │ LLM              │ LLM              │ LLM          │
│         ▼                  ▼                  ▼              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │ [Context]   │    │ [Context]   │    │ [Context]   │      │
│  │ Raw Chunk 1 │    │ Raw Chunk 2 │    │ Raw Chunk 3 │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    HYBRID RETRIEVAL                          │
│                                                              │
│   BM25 (Keywords)              Vector (Semantic)            │
│   ┌─────────────┐              ┌─────────────┐              │
│   │ Exact term  │              │ Meaning     │              │
│   │ matching    │              │ similarity  │              │
│   └──────┬──────┘              └──────┬──────┘              │
│          │                            │                      │
│          └──────────┬─────────────────┘                      │
│                     │                                        │
│                     ▼                                        │
│           Reciprocal Rank Fusion                            │
│           (Combines both rankings)                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    ANSWER GENERATION                         │
│                                                              │
│   Retrieved Context + Question ──► LLM ──► Answer           │
│                      (with Chain-of-Thought reasoning)       │
│                                                              │
│   🧬 GEPA can optimize these prompts automatically!         │
└─────────────────────────────────────────────────────────────┘
```

## 🧬 GEPA Optimizer (2025)

This tutorial also includes **GEPA** (Genetic-Pareto Evolutionary Algorithm), DSPy's newest prompt optimizer.

### Why GEPA?

Your retrieval might be perfect, but if the generation prompt is weak, answers will be poor. GEPA automatically evolves better prompts.

```
Generation 0 (Initial):              Generation N (Evolved):
┌──────────────────────────┐        ┌──────────────────────────────────┐
│ "Answer using context"   │  ───►  │ "You are an API documentation    │
│                          │        │  expert. Extract the specific    │
│                          │        │  details from context to answer  │
│                          │        │  accurately and concisely..."    │
└──────────────────────────┘        └──────────────────────────────────┘
```

### GEPA vs Previous Optimizers

| Metric | MIPROv2 | GEPA |
|--------|---------|------|
| Performance | baseline | **+10%** |
| LLM Calls | 35x | **1x** |
| Prompt Length | 9x | **1x** |

### How GEPA Works

1. **REFLECTION**: LLM analyzes why predictions failed
2. **EVOLUTION**: Successful prompt traits are combined (genetic algorithm)
3. **PARETO**: Maintains diverse solutions, not just one "best"

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   cd contextual_rag
   pip install -r requirements.txt
   ```

2. **Set up your API key:**
   - Make sure `.env` exists in the parent directory with your `GEMINI_API_KEY`
   - Get a free key at https://aistudio.google.com/apikey

3. **Run the tutorial:**
   ```bash
   python contextual_rag.py
   ```

## 📚 Key Concepts Explained

### 1. Contextual Chunking (DSPy Module)

```python
class ChunkContextualizer(dspy.Module):
    """Uses LLM to add context to each chunk."""
    
    def __init__(self):
        super().__init__()
        self.generate_context = dspy.Predict(GenerateChunkContext)
    
    def forward(self, chunk_content, document_title, section_title, document_summary):
        result = self.generate_context(
            document_title=document_title,
            section_title=section_title,
            full_document_summary=document_summary,
            chunk_content=chunk_content
        )
        return f"[{result.context_prefix}]\n\n{chunk_content}"
```

### 2. Hybrid Retrieval

**BM25** excels at finding exact keyword matches:
- Query: "WebSocket authentication" → Finds chunks containing "WebSocket" and "authentication"

**Vector Search** finds semantic similarity:
- Query: "How do I log in?" → Finds chunks about authentication even without those exact words

**Reciprocal Rank Fusion** combines them:
```python
def rrf_score(doc):
    return 1/(k + bm25_rank) + 1/(k + vector_rank)
```

### 3. DSPy Signatures

DSPy uses "Signatures" to define what the LLM should do:

```python
class AnswerFromContext(dspy.Signature):
    """Answer a question using the provided context."""
    
    context = dspy.InputField(desc="Retrieved documentation")
    question = dspy.InputField(desc="User's question")
    
    reasoning = dspy.OutputField(desc="Analysis of relevant context")
    answer = dspy.OutputField(desc="The answer")
```

## 🛠️ Technologies Used

| Component | Technology | Why |
|-----------|------------|-----|
| LLM | Gemini 2.5 Flash | Fast, capable, free tier available |
| Framework | DSPy 3.x | Modular LLM programming |
| Embeddings | sentence-transformers | Free, local, high quality |
| Keyword Search | rank-bm25 | Industry-standard BM25 |
| Vector Math | NumPy | Fast similarity computation |

## 🎓 Learning Path

After completing this tutorial, explore:

1. **Add Reranking**: Use a cross-encoder model to rerank retrieved chunks
2. **Scale Up**: Replace in-memory vectors with a vector database (Chroma, Pinecone)
3. **Optimize with DSPy**: Use DSPy's optimizers to improve prompt quality
4. **Production**: Add caching, async processing, and monitoring

## 📖 References

- [Anthropic's Contextual Retrieval Blog Post](https://www.anthropic.com/research/contextual-retrieval)
- [DSPy Documentation](https://dspy.ai/)
- [sentence-transformers](https://www.sbert.net/)
