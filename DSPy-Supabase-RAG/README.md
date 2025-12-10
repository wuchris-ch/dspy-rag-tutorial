# DSPy Supabase RAG Pipeline

A production-ready RAG (Retrieval-Augmented Generation) pipeline with comprehensive evaluation.

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **PDF Parsing** | Docling | Layout analysis, table detection, OCR |
| **Embeddings** | sentence-transformers | all-MiniLM-L6-v2 (384 dims) |
| **Vector Store** | Supabase pgvector | HNSW index, hybrid search |
| **Keyword Search** | rank-bm25 | BM25Okapi algorithm |
| **LLM** | Groq | kimi-k2-instruct (fast inference) |
| **Framework** | DSPy | Structured LLM programming |
| **Evaluation** | RAGAS + LLM-as-Judge | Multi-metric assessment |

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INGESTION PIPELINE                                │
└─────────────────────────────────────────────────────────────────────────────┘

   📄 PDF/DOCX/HTML                    
         │                             
         ▼                             
   ┌───────────────┐    Docling extracts text, tables, 
   │   Docling     │    formulas with layout analysis.
   │   Parser      │    Falls back to OCR (EasyOCR) for
   └───────┬───────┘    scanned documents.
           │                           
           ▼                           
   ┌───────────────┐    Splits into semantic chunks.
   │   Chunking    │    Optionally adds LLM-generated
   │  + Context    │    context prefixes (Anthropic's
   └───────┬───────┘    Contextual Retrieval technique).
           │                           
           ▼                           
   ┌───────────────┐    sentence-transformers generates
   │  Embeddings   │    384-dim vectors locally.
   │  (MiniLM)     │    No API costs, runs on CPU/GPU.
   └───────┬───────┘                   
           │                           
           ▼                           
   ┌───────────────┐    Stores vectors + metadata in
   │   Supabase    │    PostgreSQL with pgvector.
   │   pgvector    │    HNSW index for fast retrieval.
   └───────────────┘                   


┌─────────────────────────────────────────────────────────────────────────────┐
│                            QUERY PIPELINE                                   │
└─────────────────────────────────────────────────────────────────────────────┘

   ❓ User Question                    
         │                             
         ├──────────────┬──────────────┐
         ▼              ▼              │
   ┌───────────┐  ┌───────────┐        │
   │   BM25    │  │  Vector   │        │  HYBRID RETRIEVAL
   │  Search   │  │  Search   │        │  
   │ (keywords)│  │ (semantic)│        │  BM25 finds exact terms
   └─────┬─────┘  └─────┬─────┘        │  Vector finds meaning
         │              │              │
         └──────┬───────┘              │
                ▼                      │
   ┌─────────────────────┐             │
   │  Reciprocal Rank    │             │  RRF combines rankings
   │  Fusion (RRF)       │             │  without score normalization
   └──────────┬──────────┘             │
              │                        │
              ▼                        │
   ┌─────────────────────┐             
   │  Top-K Documents    │  Retrieved chunks with
   │  + Metadata         │  source, section, scores
   └──────────┬──────────┘             
              │                        
              ▼                        
   ┌─────────────────────┐  DSPy ChainOfThought
   │   Groq LLM          │  generates answer with
   │   (kimi-k2)         │  reasoning + sources
   └──────────┬──────────┘             
              │                        
              ▼                        
   💬 Answer + Reasoning + Sources     


┌─────────────────────────────────────────────────────────────────────────────┐
│                          EVALUATION PIPELINE                                │
└─────────────────────────────────────────────────────────────────────────────┘

   📊 Test Set (questions + expected answers)
         │
         ▼
   ┌─────────────────────────────────────────────────────────┐
   │                   EVALUATION METHODS                     │
   ├─────────────────┬───────────────────┬───────────────────┤
   │                 │                   │                   │
   │  ┌───────────┐  │  ┌─────────────┐  │  ┌─────────────┐  │
   │  │  RAGAS    │  │  │   DSPy      │  │  │ LLM-as-     │  │
   │  │ (OpenAI)  │  │  │ SemanticF1  │  │  │   Judge     │  │
   │  └─────┬─────┘  │  └──────┬──────┘  │  └──────┬──────┘  │
   │        │        │         │         │         │         │
   │  Industry       │  Uses your       │  Custom          │
   │  standard       │  configured      │  criteria,       │
   │  metrics        │  LLM (Groq)      │  no ground       │
   │                 │                   │  truth needed    │
   └────────┬────────┴─────────┬─────────┴─────────┬────────┘
            │                  │                   │
            └──────────────────┼───────────────────┘
                               ▼
   ┌─────────────────────────────────────────────────────────┐
   │                    METRICS COMPUTED                      │
   ├──────────────────────────┬──────────────────────────────┤
   │      RETRIEVAL           │        GENERATION            │
   ├──────────────────────────┼──────────────────────────────┤
   │  • Context Precision     │  • Faithfulness              │
   │    (relevant docs?)      │    (grounded in context?)    │
   │                          │                              │
   │  • Context Recall        │  • Answer Relevancy          │
   │    (all relevant found?) │    (addresses question?)     │
   │                          │                              │
   │  • Context Relevance     │  • Answer Correctness        │
   │    (overall quality)     │    (factually accurate?)     │
   └──────────────────────────┴──────────────────────────────┘
                               │
                               ▼
   ╔═══════════════════════════════════════════════════════════╗
   ║  Overall Score: 65.9%                                     ║
   ║  ├─ Retrieval:  49% (precision 44%, recall 54%)           ║
   ║  └─ Generation: 83% (faithfulness 76%, relevancy 90%)     ║
   ╚═══════════════════════════════════════════════════════════╝
```

---

## Evaluation System

### Latest Results (20 test samples)

| Category | Metric | Score | Notes |
|----------|--------|-------|-------|
| **Overall** | Combined | **65.9%** | Good - minor optimization needed |
| **Retrieval** | Context Precision | 43.6% | Room for improvement |
| | Context Recall | 54.2% | Moderate coverage |
| **Generation** | Faithfulness | 75.6% | Good grounding |
| | Answer Relevancy | 90.1% | Excellent |

*Evaluated with RAGAS using gpt-4o-mini as the judge model. RAG queries powered by Groq kimi-k2-instruct.*

### Three Evaluation Approaches

| Approach | Judge Model | Best For | Speed |
|----------|-------------|----------|-------|
| **RAGAS** | gpt-4o-mini (OpenAI) | Production benchmarks | ~2 min |
| **DSPy SemanticF1** | Your LLM (Groq/Gemini) | Answer correctness | Medium |
| **LLM-as-Judge** | Your LLM (Groq/Gemini) | Quick checks, no ground truth | Fast |

### Evaluation Metrics Explained

#### Retrieval Metrics

| Metric | What It Measures | How It's Computed |
|--------|------------------|-------------------|
| **Context Precision** | Are retrieved docs relevant? | LLM judges each chunk's relevance to query |
| **Context Recall** | Did we get all relevant info? | Compares retrieved vs. ground truth claims |
| **Context Relevance** | Overall retrieval quality | Combined precision/recall score |

#### Generation Metrics

| Metric | What It Measures | How It's Computed |
|--------|------------------|-------------------|
| **Faithfulness** | Is answer grounded in context? | Extracts claims, verifies each against context |
| **Answer Relevancy** | Does answer address the question? | Generates reverse questions, measures similarity |
| **Answer Correctness** | Is answer factually correct? | SemanticF1 vs. ground truth |

### Running Evaluation

Evaluation uses `save_questions_to_faq=False` so test queries do not pollute your FAQ store.

```python
from evaluation import PipelineEvaluator
from rag_pipeline import RAGSystem

# Initialize
rag = RAGSystem()
evaluator = PipelineEvaluator(rag)

# Quick eval (no ground truth needed)
result = evaluator.quick_eval([
    "What is the main topic?",
    "What are the key findings?",
])
print(result)

# Full eval with ground truth
result = evaluator.full_eval([
    {"question": "What is X?", "expected_answer": "X is..."},
])
evaluator.generate_report(result, "eval_results.json")
```

### Interpreting Results

| Score | Interpretation | Action |
|-------|----------------|--------|
| **> 80%** | Excellent | Production ready |
| **60-80%** | Good | Minor optimization |
| **40-60%** | Fair | Review retrieval/prompts |
| **< 40%** | Poor | Major debugging needed |

### Sample-Level Insights

Results include per-question scores, helping identify weak spots:

```json
{
  "question": "How does Docling handle tables?",
  "faithfulness": 1.0,        // ✓ Perfect - answer grounded in context
  "answer_relevancy": 0.90,   // ✓ Excellent - addresses the question
  "context_precision": 0.53,  // △ Moderate - some irrelevant chunks retrieved
  "context_recall": 1.0       // ✓ Perfect - all relevant info found
}
```

**Common patterns:**
- High faithfulness + low recall → Retrieval missing relevant docs
- Low faithfulness + high recall → Generation hallucinating despite good context
- Low relevancy → Question-answer mismatch, check prompt

---

## Quick Start

> **📖 See [START_HERE.md](START_HERE.md) for detailed step-by-step instructions**

```bash
# 1. Install (using uv - 10x faster than pip)
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# 2. Configure
cp .env.example .env  # Add your API keys

# 3. Setup Supabase (run SQL from START_HERE.md)

# 4. Download sample PDFs
uv run download_samples.py

# 5. Ingest & Query
uv run rag_pipeline.py ingest sample_pdfs/*.pdf
uv run rag_pipeline.py interactive
```

---

## Usage

### Python API

```python
from rag_pipeline import RAGSystem

# Initialize
rag = RAGSystem(
    llm_provider="groq",
    llm_model="moonshotai/kimi-k2-instruct",
    hybrid_retrieval=True,
    save_questions_to_faq=True,  # set False to skip FAQ logging
)

# Ingest documents
rag.ingest("document.pdf")

# Query
response = rag.query("What are the key findings?")
print(response.answer)
print(response.reasoning)
print(response.sources)
```

### CLI Commands

```bash
# Ingest
uv run rag_pipeline.py ingest document.pdf
uv run rag_pipeline.py ingest *.pdf

# Query
uv run rag_pipeline.py query "Your question here"
uv run rag_pipeline.py interactive
uv run rag_pipeline.py query "Your question here" --no-save-faq  # skip FAQ logging
uv run rag_pipeline.py interactive --no-save-faq  # interactive without logging

# Evaluate
uv run evaluation.py quick -q "Question 1" "Question 2"
uv run evaluation.py full -f test_set.json -o results.json

# Slower rate for free-tier APIs (default 3s delay)
uv run evaluation.py full -f test_set.json -o results.json --delay 5
```

---

## Configuration

### Required Environment Variables

```env
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=your-service-role-key
GROQ_API_KEY=gsk_xxx
```

### Optional Environment Variables

```env
OPENAI_API_KEY=sk-xxx        # For RAGAS evaluation
GEMINI_API_KEY=xxx           # Alternative LLM
```

### Model Options

| Provider | Model | Use Case |
|----------|-------|----------|
| **Groq** | `moonshotai/kimi-k2-instruct` | Default - powerful reasoning |
| **Groq** | `llama-3.3-70b-versatile` | Fast, general purpose |
| **Gemini** | `gemini-2.5-flash` | Alternative provider |

---

## Project Structure

```
DSPy-Supabase-RAG/
├── rag_pipeline.py       # Main RAG system
├── pdf_processor.py      # Docling PDF parsing
├── embeddings.py         # Embedding generation + Supabase
├── retriever.py          # Hybrid search (BM25 + vector)
├── evaluation.py         # RAGAS + LLM-as-Judge
├── download_samples.py   # Download test PDFs
├── requirements.txt      # Dependencies
├── pyproject.toml        # Modern Python config
├── .env.example          # Environment template
├── START_HERE.md         # Quick start guide
└── README.md             # This file
```

---

## Features

### Docling PDF Processing
- Layout analysis and table detection
- OCR fallback (EasyOCR/Tesseract)
- Formula and image extraction
- Multiple chunking strategies (semantic/fixed/paragraph)

### Hybrid Retrieval
- **BM25**: Exact keyword matching
- **Vector**: Semantic similarity (cosine distance)
- **RRF**: Reciprocal Rank Fusion combines both

### Contextual Chunking
Based on [Anthropic's Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) (reduces failures by 67%):

```
Before: "Revenue grew by 15%..."
After:  "[This chunk is from ACME's Q3 2024 report, discussing quarterly revenue growth.]
         Revenue grew by 15%..."
```

### FAQ Capture (Question Logging)
- Each answered question is stored back into Supabase as a FAQ chunk (metadata `type=faq`), so proven answers become retrievable context for future queries.
- Hybrid retrieval picks them up automatically; disable when you do not want this behavior with `RAGSystem(save_questions_to_faq=False)` or CLI `--no-save-faq`.

### Multi-Strategy Evaluation
- **RAGAS**: Industry-standard metrics (requires OpenAI)
- **SemanticF1**: DSPy's fact-based comparison
- **LLM-as-Judge**: Custom criteria, explainable scores

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `SUPABASE_URL not found` | Copy `.env.example` to `.env`, add credentials |
| `match_documents not found` | Run SQL setup in Supabase (see START_HERE.md) |
| `OcrOptions error` | Update to latest Docling: `pip install -U docling` |
| Slow embeddings | Use GPU: `EmbeddingGenerator(device="cuda")` |

---

## License

MIT
