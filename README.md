# 🧠 DSPy RAG Tutorial

Learn how **Retrieval-Augmented Generation (RAG)** works using DSPy — a framework that treats LLM prompts as modular, optimizable programs.

## What is RAG?

**RAG = Retrieval + Augmented + Generation**

1. **Retrieval**: Find relevant documents from a knowledge base
2. **Augmented**: Add those documents to the LLM's context
3. **Generation**: LLM generates an answer using that context

**Why RAG matters:**
- LLMs have knowledge cutoffs (they don't know recent info)
- LLMs hallucinate (confidently say wrong things)
- RAG grounds the LLM in your actual data

## What is DSPy?

DSPy treats LLM interactions as **programming**, not prompting:

| Traditional Prompting | DSPy Approach |
|-----------------------|---------------|
| Hand-write prompts | Define **Signatures** (input/output contracts) |
| Copy-paste examples | DSPy generates prompts automatically |
| Trial and error | **Optimize** prompts with training data |
| Fragile, hard to change | Modular, testable, maintainable |

## Architecture

This tutorial uses:
- **LLM**: Google Gemini 2.5 Flash (fast, cost-effective)
- **Framework**: DSPy 3.0.4 (latest version)
- **Optimizer**: GEPA (Genetic-Pareto) - 10%+ better than MIPROv2, 35x more efficient

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up your API key (get one at https://aistudio.google.com/apikey)
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 4. Run the basic tutorial
python rag_example.py

# 5. Run the advanced tutorial (optional)
python rag_advanced.py
```

## Two Tutorials: Basic vs Advanced

### 📘 Basic RAG (`rag_example.py`)

**Perfect for beginners!** Learn the fundamentals:

- ✅ **Knowledge Base**: Programming language facts (Python, JavaScript, Rust, Go)
- ✅ **Simple Retrieval**: Keyword-based document matching
- ✅ **DSPy Signatures**: Define input/output contracts
- ✅ **RAG Module**: Combine retrieval + generation
- ✅ **Chain of Thought**: Add step-by-step reasoning
- ✅ **Interactive Mode**: Ask your own questions

**Run it:**
```bash
python rag_example.py
```

**What you'll see:**
- How retrieval finds relevant documents
- How DSPy generates prompts from signatures
- How RAG combines context with questions
- Chain-of-thought reasoning in action

---

### 🚀 Advanced RAG (`rag_advanced.py`)

**For those ready to level up!** Advanced patterns:

- ✅ **TF-IDF Retrieval**: Smarter retrieval that weighs rare words higher
- ✅ **Confidence Levels**: LLM reports how confident it is
- ✅ **Multi-Hop Reasoning**: Extract facts first, then synthesize answers
- ✅ **🧬 GEPA Optimizer**: Automatically evolve better prompts!

**Run it:**
```bash
python rag_advanced.py
```

**What you'll see:**
- TF-IDF vs keyword retrieval comparison
- Confidence scoring for answers
- Multi-stage reasoning pipelines
- **GEPA optimization** - watch DSPy automatically improve prompts!

**GEPA Features:**
- 🎯 **10%+ better** than MIPROv2
- ⚡ **35x fewer** LLM calls needed
- 📝 **9x shorter** prompts generated
- 🧠 Uses **reflection** to learn from mistakes

---

## What You'll Learn

### Step 1: Knowledge Base
```python
# RAG needs chunks of text to retrieve from
docs = [
    "Python was created by Guido van Rossum in 1991...",
    "JavaScript was created by Brendan Eich in 1995...",
]
```

### Step 2: Configure DSPy
```python
import dspy

lm = dspy.LM(model="gemini/gemini-2.5-flash", api_key=api_key)
dspy.configure(lm=lm)
```

### Step 3: Define a Signature
```python
class AnswerQuestion(dspy.Signature):
    """Answer a question using the provided context."""
    
    context = dspy.InputField(desc="Relevant information")
    question = dspy.InputField(desc="The question to answer")
    answer = dspy.OutputField(desc="A clear, accurate answer")
```

### Step 4: Build a Retriever
```python
class SimpleRetriever:
    def __call__(self, query: str) -> list[str]:
        # Find and return relevant documents
        ...
```

### Step 5: Create the RAG Module
```python
class RAG(dspy.Module):
    def forward(self, question: str):
        # 1. Retrieve relevant docs
        docs = self.retriever(question)
        context = "\n".join(docs)
        
        # 2. Generate answer with context
        return self.generate(context=context, question=question)
```

## Key DSPy Concepts

### Signatures
Define **what** the LLM should do (input → output):
```python
class Summarize(dspy.Signature):
    text = dspy.InputField()
    summary = dspy.OutputField()
```

### Modules
Define **how** to orchestrate LLM calls:
- `dspy.Predict(sig)` — Simple prompt/response
- `dspy.ChainOfThought(sig)` — Adds step-by-step reasoning
- `dspy.ReAct(sig)` — Adds tool use

### Optimizers

**GEPA (Recommended)** - Newest, most efficient:
```python
optimizer = dspy.GEPA(
    metric=my_metric,
    auto="light",  # or "medium", "heavy"
    reflection_lm=reflection_model
)
optimized_rag = optimizer.compile(rag, trainset=examples)
```

**MIPROv2** - Older, still solid:
```python
optimizer = dspy.MIPROv2(metric=my_metric)
optimized_rag = optimizer.compile(rag, trainset=examples)
```

## Project Structure

```
RAG-DSPy/
├── rag_example.py      # 📘 Basic RAG tutorial (start here!)
├── rag_advanced.py     # 🚀 Advanced RAG + GEPA optimization
├── requirements.txt    # Dependencies (DSPy 3.0.4+)
├── .env.example        # API key template
├── .env                # Your GEMINI_API_KEY (create this)
└── README.md           # You are here!
```

## Example Output

### Basic RAG
```
❓ Question: Who created Python and when?

💬 Answer: Python was created by Guido van Rossum and first released in 1991.


❓ Question: Which language should I learn for systems programming?

🔍 Reasoning: Looking at the context, I see information about Rust and Go.
   Rust focuses on memory safety without garbage collection, using an 
   ownership system. It's been voted "most loved" language and is used
   by major companies. Go is simpler but has garbage collection...

💬 Answer: For systems programming, Rust is an excellent choice because it 
   provides memory safety guarantees at compile time without garbage 
   collection overhead...
```

### Advanced RAG with GEPA
```
🧬 Running GEPA Optimization...
   (This uses reflection to evolve better prompts)

📊 BEFORE Optimization:
  Q: When did the first Moon landing happen?
  A: 1969...

📊 AFTER Optimization:
  Q: When did the first Moon landing happen?
  A: 1969...

✅ GEPA found optimized prompts for your RAG!
   The optimized module now has better instructions.
```

## Comparison: Basic vs Advanced

| Feature | Basic (`rag_example.py`) | Advanced (`rag_advanced.py`) |
|---------|-------------------------|------------------------------|
| **Retrieval** | Keyword matching | TF-IDF (semantic weighting) |
| **Reasoning** | Chain-of-thought | Multi-hop (extract → synthesize) |
| **Confidence** | No | Yes (HIGH/MEDIUM/LOW) |
| **Optimization** | None | GEPA optimizer |
| **Knowledge Base** | Programming languages | Space exploration |
| **Best For** | Learning fundamentals | Production patterns |

## Next Steps

1. **Run both tutorials**: Start with basic, then try advanced
2. **Add vector embeddings**: Replace keyword matching with semantic search
3. **Use a vector database**: Chroma, Pinecone, or Weaviate for production
4. **Chunk your own data**: Load PDFs, websites, or databases
5. **Experiment with GEPA**: Add more training examples and optimize!

## Resources

- [DSPy Documentation](https://dspy.ai/)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)
- [GEPA Paper](https://arxiv.org/abs/2507.19457) - Genetic-Pareto optimization
- [RAG Paper](https://arxiv.org/abs/2005.11401) - Original RAG research
- [Gemini API](https://aistudio.google.com/apikey) - Get your API key

## Troubleshooting

**"GEMINI_API_KEY not found"**
- Make sure you've created `.env` from `.env.example`
- Add your key: `GEMINI_API_KEY=your-key-here`

**GEPA optimization fails**
- Make sure you have at least 3-6 training examples
- Try `auto="light"` for faster, cheaper runs
- Check that your reflection_lm is configured

**Import errors**
- Make sure virtual environment is activated: `source venv/bin/activate`
- Reinstall: `pip install -r requirements.txt --upgrade`
