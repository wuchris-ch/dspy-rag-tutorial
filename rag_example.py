"""
🧠 DSPy RAG Tutorial - Understanding Retrieval-Augmented Generation

This example teaches you how RAG works with DSPy by building a knowledge base
about programming languages. You'll learn:

1. What RAG is and why it matters
2. How DSPy structures LLM programs as "modules"
3. How retrieval finds relevant context
4. How generation uses that context to answer questions

Run with: python rag_example.py
"""

import os
from dotenv import load_dotenv
import dspy

# ═══════════════════════════════════════════════════════════════════════════════
# 📚 STEP 1: Understanding the Knowledge Base
# ═══════════════════════════════════════════════════════════════════════════════
# 
# RAG needs a "knowledge base" - a collection of text chunks that contain
# information the LLM might not know (or might hallucinate about).
# 
# Each chunk should be:
#   - Self-contained (makes sense on its own)
#   - Focused on one topic
#   - Not too long (typically 100-500 tokens)

PROGRAMMING_KNOWLEDGE = [
    # Python facts
    "Python was created by Guido van Rossum and first released in 1991. It emphasizes code readability with significant whitespace. Python is named after Monty Python, not the snake.",
    
    "Python uses dynamic typing and garbage collection. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
    
    "Python's package manager is pip, and packages are hosted on PyPI (Python Package Index). Virtual environments (venv) help isolate project dependencies.",
    
    # JavaScript facts
    "JavaScript was created by Brendan Eich in just 10 days in 1995 while working at Netscape. Despite the name, it has no relation to Java - the name was a marketing decision.",
    
    "JavaScript is the only programming language that runs natively in web browsers. Node.js, created in 2009, allows JavaScript to run on servers.",
    
    "JavaScript uses prototype-based inheritance rather than classical inheritance. ES6 (2015) added classes, arrow functions, let/const, and promises.",
    
    # Rust facts
    "Rust was created by Mozilla and first released in 2010. It focuses on memory safety without garbage collection, using a unique ownership system.",
    
    "Rust's borrow checker enforces memory safety at compile time. This prevents common bugs like null pointer dereferences, buffer overflows, and data races.",
    
    "Rust has been voted 'most loved programming language' in Stack Overflow surveys for multiple years. It's used in Firefox, Dropbox, and Discord.",
    
    # Go facts
    "Go (Golang) was created at Google by Robert Griesemer, Rob Pike, and Ken Thompson. It was designed to handle Google's scale of software development.",
    
    "Go compiles to native machine code and has built-in garbage collection. Its goroutines provide lightweight concurrency that's much cheaper than OS threads.",
    
    "Go has a deliberately simple syntax with no generics until version 1.18 (2022). The 'gofmt' tool enforces a single code style for all Go programs.",
    
    # General programming concepts
    "Compiled languages (like Go, Rust, C++) convert source code to machine code before execution. Interpreted languages (like Python, JavaScript) execute code line by line.",
    
    "Static typing (Go, Rust, Java) catches type errors at compile time. Dynamic typing (Python, JavaScript, Ruby) checks types at runtime, offering flexibility but potentially hiding bugs.",
    
    "Garbage collection automatically frees unused memory. Languages like Python, JavaScript, Go, and Java use it. Rust and C++ require manual memory management (or Rust's ownership system).",
]

# ═══════════════════════════════════════════════════════════════════════════════
# 🔧 STEP 2: Configure DSPy with an LLM
# ═══════════════════════════════════════════════════════════════════════════════
# 
# DSPy needs a "language model" (LM) to generate text. We'll use Google's Gemini 2.5 Flash,
# but DSPy supports many providers (OpenAI, Anthropic, local models, etc.)

def setup_dspy():
    """Configure DSPy with Gemini."""
    load_dotenv()  # Load GEMINI_API_KEY from .env file
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "❌ GEMINI_API_KEY not found!\n"
            "   1. Copy .env.example to .env\n"
            "   2. Add your Gemini API key (get one at https://aistudio.google.com/apikey)"
        )
    
    # Configure the language model
    # Gemini 2.5 Flash is fast and capable
    lm = dspy.LM(model="gemini/gemini-2.5-flash", api_key=api_key)
    dspy.configure(lm=lm)
    
    print("✅ DSPy configured with Gemini 2.5 Flash")
    return lm


# ═══════════════════════════════════════════════════════════════════════════════
# 📝 STEP 3: Define a Signature (What the LLM should do)
# ═══════════════════════════════════════════════════════════════════════════════
# 
# A "Signature" in DSPy defines the input/output contract of an LLM call.
# It's like a function signature but for language models.
# 
# DSPy uses signatures to:
#   - Generate effective prompts automatically
#   - Validate inputs/outputs
#   - Enable optimization and fine-tuning

class AnswerQuestion(dspy.Signature):
    """Answer a question using the provided context. Be concise and accurate."""
    
    # Input fields - what we'll provide to the LLM
    context = dspy.InputField(desc="Relevant information to answer the question")
    question = dspy.InputField(desc="The question to answer")
    
    # Output field - what we expect back
    answer = dspy.OutputField(desc="A clear, accurate answer based on the context")


class AnswerWithReasoning(dspy.Signature):
    """Answer a question by first reasoning through the context step by step."""
    
    context = dspy.InputField(desc="Relevant information to answer the question")
    question = dspy.InputField(desc="The question to answer")
    
    # Multiple outputs - DSPy generates both!
    reasoning = dspy.OutputField(desc="Step-by-step reasoning through the context")
    answer = dspy.OutputField(desc="The final answer based on the reasoning")


# ═══════════════════════════════════════════════════════════════════════════════
# 🔍 STEP 4: Build a Simple Retriever
# ═══════════════════════════════════════════════════════════════════════════════
# 
# Retrieval finds the most relevant chunks from our knowledge base.
# This simple implementation uses keyword matching.
# 
# Production systems use:
#   - Vector embeddings (OpenAI, sentence-transformers)
#   - Vector databases (Pinecone, Weaviate, Chroma, FAISS)

class SimpleRetriever:
    """
    A simple keyword-based retriever for learning.
    
    How it works:
    1. Takes a query (the user's question)
    2. Scores each document by how many query words it contains
    3. Returns the top-k highest scoring documents
    """
    
    def __init__(self, documents: list[str], k: int = 3):
        self.documents = documents
        self.k = k
    
    def __call__(self, query: str) -> list[str]:
        """Retrieve the k most relevant documents for the query."""
        
        # Simple scoring: count matching words (case-insensitive)
        query_words = set(query.lower().split())
        
        scored_docs = []
        for doc in self.documents:
            doc_words = set(doc.lower().split())
            # Score = number of query words found in document
            score = len(query_words & doc_words)
            scored_docs.append((score, doc))
        
        # Sort by score (highest first) and return top k
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:self.k]]


# ═══════════════════════════════════════════════════════════════════════════════
# 🤖 STEP 5: Build the RAG Module
# ═══════════════════════════════════════════════════════════════════════════════
# 
# A DSPy "Module" combines retrieval + generation into a single callable unit.
# This is the heart of RAG!

class RAG(dspy.Module):
    """
    Retrieval-Augmented Generation module.
    
    The RAG process:
    1. RETRIEVE: Find relevant documents for the question
    2. AUGMENT: Add those documents as context
    3. GENERATE: Have the LLM answer using that context
    """
    
    def __init__(self, retriever, use_reasoning: bool = False):
        super().__init__()
        self.retriever = retriever
        
        # Choose which signature to use
        if use_reasoning:
            # ChainOfThought adds step-by-step reasoning
            self.generate = dspy.ChainOfThought(AnswerWithReasoning)
        else:
            # Predict is the simplest predictor
            self.generate = dspy.Predict(AnswerQuestion)
        
        self.use_reasoning = use_reasoning
    
    def forward(self, question: str):
        """
        Process a question through the RAG pipeline.
        
        This is where the magic happens!
        """
        # Step 1: RETRIEVE relevant context
        retrieved_docs = self.retriever(question)
        context = "\n\n".join(retrieved_docs)
        
        # Step 2 & 3: AUGMENT prompt with context and GENERATE answer
        result = self.generate(context=context, question=question)
        
        # Attach retrieved docs for inspection
        result.retrieved_docs = retrieved_docs
        
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# 🎮 STEP 6: Interactive Demo
# ═══════════════════════════════════════════════════════════════════════════════

def print_divider(title: str = ""):
    print(f"\n{'═' * 70}")
    if title:
        print(f"  {title}")
        print('═' * 70)

def demo_retrieval_only():
    """Show how retrieval works independently."""
    print_divider("🔍 DEMO: Understanding Retrieval")
    
    retriever = SimpleRetriever(PROGRAMMING_KNOWLEDGE, k=3)
    
    query = "Who created Python?"
    print(f"\nQuery: '{query}'")
    print("\nRetrieved documents:")
    
    for i, doc in enumerate(retriever(query), 1):
        print(f"\n  [{i}] {doc[:100]}...")

def demo_simple_rag():
    """Show basic RAG in action."""
    print_divider("🤖 DEMO: Simple RAG")
    
    retriever = SimpleRetriever(PROGRAMMING_KNOWLEDGE, k=3)
    rag = RAG(retriever, use_reasoning=False)
    
    questions = [
        "Who created Python and when?",
        "What makes Rust special for memory safety?",
        "What's the difference between compiled and interpreted languages?",
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        result = rag(question)
        print(f"💬 Answer: {result.answer}")

def demo_rag_with_reasoning():
    """Show RAG with chain-of-thought reasoning."""
    print_divider("🧠 DEMO: RAG with Chain-of-Thought Reasoning")
    
    retriever = SimpleRetriever(PROGRAMMING_KNOWLEDGE, k=3)
    rag = RAG(retriever, use_reasoning=True)
    
    question = "Which language should I learn for systems programming, and why?"
    
    print(f"\n❓ Question: {question}")
    result = rag(question)
    
    print(f"\n📚 Retrieved Context:")
    for i, doc in enumerate(result.retrieved_docs, 1):
        print(f"   [{i}] {doc[:80]}...")
    
    print(f"\n🔍 Reasoning:\n   {result.reasoning}")
    print(f"\n💬 Answer: {result.answer}")

def interactive_mode():
    """Let the user ask their own questions."""
    print_divider("💬 INTERACTIVE MODE")
    
    retriever = SimpleRetriever(PROGRAMMING_KNOWLEDGE, k=3)
    rag = RAG(retriever, use_reasoning=True)
    
    print("\nAsk questions about programming languages!")
    print("Topics covered: Python, JavaScript, Rust, Go, and general concepts")
    print("Type 'quit' to exit\n")
    
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
# 🚀 Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   🧠 DSPy RAG Tutorial                                                       ║
║   Understanding Retrieval-Augmented Generation                               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Setup DSPy
    setup_dspy()
    
    # Run demos
    demo_retrieval_only()
    demo_simple_rag()
    demo_rag_with_reasoning()
    
    # Optional: interactive mode
    print_divider()
    print("\nWant to try asking your own questions? (y/n): ", end="")
    try:
        if input().strip().lower() == 'y':
            interactive_mode()
    except (EOFError, KeyboardInterrupt):
        pass
    
    print("\n✅ Tutorial complete! Check the code comments to learn more.")


if __name__ == "__main__":
    main()

