"""
🚀 Advanced DSPy RAG Example

This example shows more sophisticated RAG patterns:
1. TF-IDF based retrieval (better than keyword matching)
2. Multi-hop reasoning
3. Handling "I don't know" cases
4. 🧬 GEPA Optimizer - automatically improve prompts!

Run with: python rag_advanced.py
"""

import os
import math
from collections import Counter
from dotenv import load_dotenv
import dspy

# ═══════════════════════════════════════════════════════════════════════════════
# 📚 Knowledge Base: Space Exploration Facts
# ═══════════════════════════════════════════════════════════════════════════════

SPACE_KNOWLEDGE = [
    # Moon missions
    "Apollo 11 was the first crewed mission to land on the Moon in 1969. Neil Armstrong and Buzz Aldrin walked on the lunar surface while Michael Collins orbited above.",
    
    "The Apollo program ran from 1961 to 1972 and successfully landed 12 astronauts on the Moon across 6 missions (Apollo 11, 12, 14, 15, 16, and 17).",
    
    "The last humans to walk on the Moon were Eugene Cernan and Harrison Schmitt during Apollo 17 in December 1972. No human has returned since.",
    
    # Mars exploration
    "The Mars Perseverance rover landed in Jezero Crater in February 2021. It carries Ingenuity, the first helicopter to fly on another planet.",
    
    "Mars has two small moons: Phobos and Deimos. Phobos orbits so close to Mars that it will eventually crash into the planet or break apart.",
    
    "A day on Mars (called a 'sol') is 24 hours and 37 minutes. A year on Mars is 687 Earth days.",
    
    # Space agencies
    "NASA (National Aeronautics and Space Administration) was founded in 1958. It's the United States' space agency and has led many historic missions.",
    
    "SpaceX, founded by Elon Musk in 2002, became the first private company to send astronauts to the International Space Station in 2020.",
    
    "The European Space Agency (ESA) is an intergovernmental organization with 22 member states. It operates the Ariane rocket family.",
    
    # International Space Station
    "The International Space Station (ISS) orbits Earth at about 400 km altitude, traveling at 28,000 km/h. It completes one orbit every 90 minutes.",
    
    "The ISS has been continuously occupied since November 2000. It's a collaboration between NASA, Roscosmos, JAXA, ESA, and CSA.",
    
    "The ISS is about the size of a football field and weighs approximately 420,000 kg. It's the largest human-made structure in space.",
    
    # Telescopes and discovery
    "The James Webb Space Telescope (JWST) launched in December 2021. It's the most powerful space telescope ever built, observing in infrared light.",
    
    "The Hubble Space Telescope has been operating since 1990. It orbits Earth and has captured some of the most iconic images of distant galaxies.",
    
    "Exoplanets are planets outside our solar system. Over 5,000 exoplanets have been confirmed as of 2023, with thousands more candidates.",
    
    # Future missions
    "NASA's Artemis program aims to return humans to the Moon and establish a sustainable presence. Artemis I completed an uncrewed test flight in 2022.",
    
    "SpaceX's Starship is designed to carry humans to Mars. It's the largest and most powerful rocket ever built, standing 120 meters tall.",
    
    "The Europa Clipper mission, launching in 2024, will study Jupiter's moon Europa, which may have a subsurface ocean that could harbor life.",
]


# ═══════════════════════════════════════════════════════════════════════════════
# 🔍 TF-IDF Retriever (More Sophisticated than Keyword Matching)
# ═══════════════════════════════════════════════════════════════════════════════

class TFIDFRetriever:
    """
    TF-IDF (Term Frequency - Inverse Document Frequency) retriever.
    
    This is smarter than simple keyword matching because:
    - TF: Words that appear often in a document are important for that document
    - IDF: Words that appear in many documents are less important overall
    
    Example: "the" appears everywhere (low IDF), but "astronaut" is specific (high IDF)
    """
    
    def __init__(self, documents: list[str], k: int = 3):
        self.documents = documents
        self.k = k
        self.doc_tokens = [self._tokenize(doc) for doc in documents]
        self.idf = self._compute_idf()
    
    def _tokenize(self, text: str) -> list[str]:
        """Convert text to lowercase tokens, removing punctuation."""
        import re
        text = text.lower()
        tokens = re.findall(r'\b[a-z]+\b', text)
        return tokens
    
    def _compute_idf(self) -> dict[str, float]:
        """Compute IDF for all terms in the corpus."""
        doc_count = len(self.documents)
        term_doc_counts = Counter()
        
        for tokens in self.doc_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                term_doc_counts[token] += 1
        
        idf = {}
        for term, count in term_doc_counts.items():
            # Standard IDF formula with smoothing
            idf[term] = math.log((doc_count + 1) / (count + 1)) + 1
        
        return idf
    
    def _compute_tfidf(self, tokens: list[str]) -> dict[str, float]:
        """Compute TF-IDF vector for a list of tokens."""
        tf = Counter(tokens)
        tfidf = {}
        for term, count in tf.items():
            tfidf[term] = count * self.idf.get(term, 1.0)
        return tfidf
    
    def _cosine_similarity(self, vec1: dict, vec2: dict) -> float:
        """Compute cosine similarity between two sparse vectors."""
        common_terms = set(vec1.keys()) & set(vec2.keys())
        if not common_terms:
            return 0.0
        
        dot_product = sum(vec1[t] * vec2[t] for t in common_terms)
        norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def __call__(self, query: str) -> list[str]:
        """Retrieve top-k documents most similar to the query."""
        query_tokens = self._tokenize(query)
        query_vec = self._compute_tfidf(query_tokens)
        
        scores = []
        for i, doc_tokens in enumerate(self.doc_tokens):
            doc_vec = self._compute_tfidf(doc_tokens)
            score = self._cosine_similarity(query_vec, doc_vec)
            scores.append((score, i, self.documents[i]))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[0], reverse=True)
        
        return [doc for score, idx, doc in scores[:self.k]]


# ═══════════════════════════════════════════════════════════════════════════════
# 📝 Advanced Signatures
# ═══════════════════════════════════════════════════════════════════════════════

class AnswerWithConfidence(dspy.Signature):
    """Answer a question based on the context. If the context doesn't contain 
    enough information, say 'I don't have enough information to answer this.'"""
    
    context = dspy.InputField(desc="Retrieved documents that may contain the answer")
    question = dspy.InputField(desc="The user's question")
    
    confidence = dspy.OutputField(desc="HIGH, MEDIUM, or LOW based on how well the context answers the question")
    answer = dspy.OutputField(desc="The answer, or an explanation of what information is missing")


class ExtractFacts(dspy.Signature):
    """Extract key facts from the context that are relevant to answering the question."""
    
    context = dspy.InputField(desc="Retrieved documents")
    question = dspy.InputField(desc="The question to answer")
    
    facts = dspy.OutputField(desc="Bullet-pointed list of relevant facts extracted from the context")


class SynthesizeAnswer(dspy.Signature):
    """Synthesize a comprehensive answer from the extracted facts."""
    
    facts = dspy.InputField(desc="Relevant facts")
    question = dspy.InputField(desc="The original question")
    
    answer = dspy.OutputField(desc="A comprehensive answer synthesized from the facts")


# ═══════════════════════════════════════════════════════════════════════════════
# 🤖 Advanced RAG Modules
# ═══════════════════════════════════════════════════════════════════════════════

class RAGWithConfidence(dspy.Module):
    """RAG that reports its confidence in the answer."""
    
    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever
        self.generate = dspy.ChainOfThought(AnswerWithConfidence)
    
    def forward(self, question: str):
        docs = self.retriever(question)
        context = "\n\n".join(docs)
        result = self.generate(context=context, question=question)
        result.retrieved_docs = docs
        return result


class MultiHopRAG(dspy.Module):
    """
    Multi-hop RAG: Extract facts first, then synthesize an answer.
    
    This helps with complex questions that require combining information
    from multiple sources.
    """
    
    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever
        self.extract = dspy.Predict(ExtractFacts)
        self.synthesize = dspy.Predict(SynthesizeAnswer)
    
    def forward(self, question: str):
        # Step 1: Retrieve
        docs = self.retriever(question)
        context = "\n\n".join(docs)
        
        # Step 2: Extract relevant facts
        extraction = self.extract(context=context, question=question)
        
        # Step 3: Synthesize answer from facts
        result = self.synthesize(facts=extraction.facts, question=question)
        
        # Attach intermediate results for inspection
        result.retrieved_docs = docs
        result.extracted_facts = extraction.facts
        
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# 🧬 GEPA Optimizer - Automatic Prompt Evolution
# ═══════════════════════════════════════════════════════════════════════════════
# 
# GEPA (Genetic-Pareto) uses evolutionary algorithms + reflection to find
# optimal prompts. It's more efficient than MIPROv2:
#   - 10%+ better performance
#   - 35x fewer LLM calls needed
#   - 9x shorter prompts generated

# Training data: Question + Expected Answer pairs
TRAINING_DATA = [
    dspy.Example(
        question="When did the first Moon landing happen?",
        expected_answer="1969"
    ).with_inputs("question"),
    
    dspy.Example(
        question="What rover landed on Mars in 2021?",
        expected_answer="Perseverance"
    ).with_inputs("question"),
    
    dspy.Example(
        question="How long is a day on Mars?",
        expected_answer="24 hours and 37 minutes"
    ).with_inputs("question"),
    
    dspy.Example(
        question="When was NASA founded?",
        expected_answer="1958"
    ).with_inputs("question"),
    
    dspy.Example(
        question="What is the ISS orbital altitude?",
        expected_answer="400 km"
    ).with_inputs("question"),
    
    dspy.Example(
        question="Who were the first humans on the Moon?",
        expected_answer="Neil Armstrong and Buzz Aldrin"
    ).with_inputs("question"),
]


def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """
    GEPA metric function with feedback.
    
    GEPA is special because it can use textual feedback to guide evolution.
    This function returns both a score AND feedback about what went wrong.
    """
    expected = gold.expected_answer.lower()
    actual = pred.answer.lower() if hasattr(pred, 'answer') else ""
    
    # Check if the key information is in the answer
    if expected in actual:
        return 1.0  # Perfect match
    
    # Partial credit for relevant answers
    expected_words = set(expected.split())
    actual_words = set(actual.split())
    overlap = len(expected_words & actual_words) / len(expected_words) if expected_words else 0
    
    if overlap > 0.5:
        score = 0.7
        feedback = f"Partially correct. Expected '{gold.expected_answer}' but got related content."
    elif overlap > 0:
        score = 0.3
        feedback = f"Contains some relevant info but missing key details. Expected: '{gold.expected_answer}'"
    else:
        score = 0.0
        feedback = f"Incorrect. Expected answer to contain '{gold.expected_answer}' but got: '{actual[:100]}...'"
    
    # Return score with feedback for GEPA's reflection
    from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback
    return ScoreWithFeedback(score=score, feedback=feedback)


class SimpleRAGForOptimization(dspy.Module):
    """A simple RAG module that GEPA will optimize."""
    
    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever
        self.generate = dspy.Predict("context, question -> answer")
    
    def forward(self, question: str):
        docs = self.retriever(question)
        context = "\n\n".join(docs)
        return self.generate(context=context, question=question)


# ═══════════════════════════════════════════════════════════════════════════════
# 🎮 Demo Functions
# ═══════════════════════════════════════════════════════════════════════════════

def setup_dspy():
    """Configure DSPy with Gemini."""
    load_dotenv()
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("❌ GEMINI_API_KEY not found! Copy .env.example to .env and add your key.")
    
    lm = dspy.LM(model="gemini/gemini-2.5-flash", api_key=api_key)
    dspy.configure(lm=lm)
    print("✅ DSPy configured with Gemini 2.5 Flash\n")


def print_divider(title: str = ""):
    print(f"\n{'═' * 70}")
    if title:
        print(f"  {title}")
        print('═' * 70)


def demo_tfidf_retrieval():
    """Compare keyword vs TF-IDF retrieval."""
    print_divider("🔍 DEMO: TF-IDF vs Keyword Retrieval")
    
    from rag_example import SimpleRetriever
    
    query = "Which space telescope is the most powerful?"
    
    print(f"\nQuery: '{query}'\n")
    
    # Keyword retrieval
    keyword_retriever = SimpleRetriever(SPACE_KNOWLEDGE, k=2)
    keyword_results = keyword_retriever(query)
    
    print("📝 Keyword Retrieval (counts matching words):")
    for i, doc in enumerate(keyword_results, 1):
        print(f"   [{i}] {doc[:80]}...")
    
    # TF-IDF retrieval
    tfidf_retriever = TFIDFRetriever(SPACE_KNOWLEDGE, k=2)
    tfidf_results = tfidf_retriever(query)
    
    print("\n🎯 TF-IDF Retrieval (weighs rare/important words higher):")
    for i, doc in enumerate(tfidf_results, 1):
        print(f"   [{i}] {doc[:80]}...")


def demo_confidence():
    """Show RAG with confidence levels."""
    print_divider("📊 DEMO: RAG with Confidence Levels")
    
    retriever = TFIDFRetriever(SPACE_KNOWLEDGE, k=3)
    rag = RAGWithConfidence(retriever)
    
    # Question that CAN be answered
    print("\n❓ Question: When did Apollo 11 land on the Moon?")
    result = rag("When did Apollo 11 land on the Moon?")
    print(f"📈 Confidence: {result.confidence}")
    print(f"💬 Answer: {result.answer}")
    
    # Question that CANNOT be answered well
    print("\n❓ Question: What is the population of Mars?")
    result = rag("What is the population of Mars?")
    print(f"📈 Confidence: {result.confidence}")
    print(f"💬 Answer: {result.answer}")


def demo_multihop():
    """Show multi-hop reasoning for complex questions."""
    print_divider("🔗 DEMO: Multi-Hop RAG")
    
    retriever = TFIDFRetriever(SPACE_KNOWLEDGE, k=4)
    rag = MultiHopRAG(retriever)
    
    question = "Compare the Moon and Mars exploration programs - how are they similar and different?"
    
    print(f"\n❓ Question: {question}")
    result = rag(question)
    
    print(f"\n📚 Retrieved {len(result.retrieved_docs)} documents")
    
    print(f"\n📋 Extracted Facts:\n{result.extracted_facts}")
    
    print(f"\n💬 Synthesized Answer:\n{result.answer}")


def demo_gepa_optimization():
    """
    🧬 DEMO: GEPA Optimizer
    
    This shows how GEPA automatically evolves better prompts for your RAG system.
    GEPA uses:
    - Genetic algorithms to breed successful prompts
    - Pareto optimization to maintain diverse solutions
    - Reflection to learn from failures
    """
    print_divider("🧬 DEMO: GEPA Optimizer")
    
    print("""
    GEPA (Genetic-Pareto) is DSPy's newest optimizer that:
    
    ✅ Outperforms MIPROv2 by 10%+
    ✅ Uses 35x fewer LLM calls
    ✅ Generates 9x shorter prompts
    ✅ Uses reflection to learn from mistakes
    
    Let's optimize our RAG module!
    """)
    
    # Create the base RAG module
    retriever = TFIDFRetriever(SPACE_KNOWLEDGE, k=3)
    rag = SimpleRAGForOptimization(retriever)
    
    # Test BEFORE optimization
    print("📊 BEFORE Optimization:")
    print("-" * 40)
    
    test_questions = [
        "When did the first Moon landing happen?",
        "What rover is on Mars?",
    ]
    
    for q in test_questions:
        result = rag(question=q)
        print(f"  Q: {q}")
        print(f"  A: {result.answer[:100]}...")
        print()
    
    # Run GEPA optimization
    print("🧬 Running GEPA Optimization...")
    print("   (This uses reflection to evolve better prompts)")
    print()
    
    try:
        # GEPA needs a reflection LM - use Gemini for reflection too
        api_key = os.getenv("GEMINI_API_KEY")
        reflection_lm = dspy.LM(
            model="gemini/gemini-2.5-flash",
            api_key=api_key,
            temperature=1.0,  # Higher temp for creative reflection
            max_tokens=8000
        )
        
        optimizer = dspy.GEPA(
            metric=gepa_metric,
            auto="light",  # "light" = fast, "medium" = balanced, "heavy" = thorough
            reflection_lm=reflection_lm,
        )
        
        optimized_rag = optimizer.compile(
            rag,
            trainset=TRAINING_DATA,
        )
        
        # Test AFTER optimization
        print("\n📊 AFTER Optimization:")
        print("-" * 40)
        
        for q in test_questions:
            result = optimized_rag(question=q)
            print(f"  Q: {q}")
            print(f"  A: {result.answer[:100]}...")
            print()
        
        print("✅ GEPA found optimized prompts for your RAG!")
        print("   The optimized module now has better instructions.")
        
    except Exception as e:
        print(f"⚠️  GEPA optimization requires more training data or API calls.")
        print(f"   Error: {e}")
        print("\n   For production use, you'd want:")
        print("   - More training examples (10-50+)")
        print("   - auto='medium' or 'heavy' for better results")


def interactive_mode():
    """Interactive Q&A about space."""
    print_divider("🚀 INTERACTIVE: Ask About Space!")
    
    retriever = TFIDFRetriever(SPACE_KNOWLEDGE, k=3)
    rag = RAGWithConfidence(retriever)
    
    print("\nAsk questions about space exploration!")
    print("Topics: Moon missions, Mars rovers, ISS, telescopes, future missions")
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
        print(f"\n📈 Confidence: {result.confidence}")
        print(f"💬 {result.answer}\n")


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   🚀 Advanced DSPy RAG Tutorial                                              ║
║   Better Retrieval, Multi-Hop Reasoning & GEPA Optimization                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    setup_dspy()
    
    demo_tfidf_retrieval()
    demo_confidence()
    demo_multihop()
    
    # Ask if user wants to run GEPA (it makes API calls)
    print_divider()
    print("\n🧬 Want to try GEPA optimization? (makes several API calls) (y/n): ", end="")
    try:
        if input().strip().lower() == 'y':
            demo_gepa_optimization()
    except (EOFError, KeyboardInterrupt):
        pass
    
    print_divider()
    print("\nWant to ask your own questions? (y/n): ", end="")
    try:
        if input().strip().lower() == 'y':
            interactive_mode()
    except (EOFError, KeyboardInterrupt):
        pass
    
    print("\n✅ Advanced tutorial complete!")


if __name__ == "__main__":
    main()

