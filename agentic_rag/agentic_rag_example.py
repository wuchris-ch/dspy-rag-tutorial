"""
🤖 Agentic RAG with DSPy 3.x - A Complete Tutorial

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IS AGENTIC RAG?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Regular RAG (Retrieval-Augmented Generation):
    Question → Retrieve Documents → Generate Answer → Done
    
    - One-shot process
    - Fixed retrieval step
    - No reasoning about what to do next

Agentic RAG:
    Question → Think → Choose Action → Execute → Observe → Think → ... → Answer
    
    - Multi-step reasoning loop (ReAct pattern)
    - Agent DECIDES which tools to use
    - Can chain multiple actions together
    - Self-corrects based on observations

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE ReAct PATTERN (Reasoning + Acting)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ReAct is a paradigm where the LLM alternates between:
    
    1. THOUGHT: Reason about the current situation
       "I need to find information about X before I can answer"
       
    2. ACTION: Call a tool to gather information
       search_knowledge("X")
       
    3. OBSERVATION: See what the tool returned
       "X was invented in 1995 by..."
       
    4. REPEAT until the agent has enough information to answer

This is called an "agent loop" - the model keeps acting until it's done.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DSPy's ReAct Implementation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DSPy provides `dspy.ReAct` which:
    - Takes a signature (input → output specification)
    - Takes a list of tools (Python functions the agent can call)
    - Automatically handles the thought/action/observation loop
    - Returns the final answer when the agent decides it's done

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OPTIMIZATION WITH GEPA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GEPA = Genetic-Pareto Reflective Prompt Evolution

GEPA is DSPy's reflective optimizer that:
    - Analyzes what went well and what didn't in your program
    - Uses LLM reflection to propose better prompts
    - Leverages textual feedback (not just scores) to improve
    - Can achieve significant gains in just a few iterations

The key insight: GEPA reads FEEDBACK about why answers were wrong,
not just whether they were right/wrong. This helps it fix specific issues.

Run with: python agentic_rag_example.py
"""

import os
import math
from datetime import datetime
from typing import Annotated
from dotenv import load_dotenv
import dspy
from dspy import GEPA  # DSPy's reflective prompt optimizer
from dspy.teleprompt.gepa.gepa import ScoreWithFeedback  # For GEPA feedback metrics


# ═══════════════════════════════════════════════════════════════════════════════
# 🔧 STEP 1: Define Our Tools
# ═══════════════════════════════════════════════════════════════════════════════
#
# Tools are just Python functions that the agent can call. Each tool should:
#   1. Have a clear docstring (the agent reads this to understand the tool)
#   2. Have typed parameters (helps the agent know what to pass)
#   3. Return a string (the observation the agent sees)
#
# DSPy automatically extracts:
#   - Function name (how the agent calls it)
#   - Docstring (helps agent understand when to use it)
#   - Parameter types (from type hints)


# ──────────────────────────────────────────────────────────────────────────────
# Tool 1: Calculator
# ──────────────────────────────────────────────────────────────────────────────
# A simple but powerful calculator that handles math expressions safely.
# The agent can use this for any arithmetic the question requires.

def calculate(expression: Annotated[str, "A mathematical expression like '2 + 2' or 'sqrt(16)'"]) -> str:
    """
    Evaluate a mathematical expression and return the result.
    Supports: +, -, *, /, ** (power), sqrt(), sin(), cos(), tan(), log(), abs(), round()
    Examples: "2 + 2", "sqrt(16)", "2 ** 10", "sin(3.14159 / 2)"
    """
    try:
        # Define safe math functions the agent can use
        safe_functions = {
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "abs": abs,
            "round": round,
            "pow": pow,
            "pi": math.pi,
            "e": math.e,
        }
        
        # Evaluate the expression in a restricted namespace (no builtins = safer)
        result = eval(expression, {"__builtins__": {}}, safe_functions)
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {str(e)}"


# ──────────────────────────────────────────────────────────────────────────────
# Tool 2: Knowledge Base Search
# ──────────────────────────────────────────────────────────────────────────────
# This is our RAG component - searching a knowledge base for relevant info.
# In production, this would query a vector database. Here we use keyword matching.

# Our knowledge base - facts the LLM might not know or might hallucinate about
KNOWLEDGE_BASE = [
    # Company facts (fictional but realistic)
    "TechCorp was founded in 2019 by Sarah Chen and Marcus Williams in Austin, Texas. The company specializes in AI-powered supply chain optimization.",
    "TechCorp raised a $50 million Series B round in March 2023, led by Sequoia Capital. Total funding to date is $73 million.",
    "TechCorp has 247 employees as of Q4 2024. The engineering team is 89 people, and the company is headquartered in Austin with offices in San Francisco and London.",
    "TechCorp's main product is 'ChainMind' - an AI platform that predicts supply chain disruptions 2-3 weeks before they occur with 94% accuracy.",
    "TechCorp's annual recurring revenue (ARR) grew from $12M in 2022 to $47M in 2024, representing 291% growth over two years.",
    
    # Technical specifications
    "ChainMind uses a proprietary transformer architecture trained on 15 years of global shipping data, weather patterns, and economic indicators.",
    "ChainMind integrates with SAP, Oracle, and Microsoft Dynamics. API rate limits are 1000 requests per minute on the Enterprise plan.",
    "The ChainMind API uses REST with OAuth 2.0 authentication. Response times average 120ms for prediction queries.",
    
    # Industry context
    "The global supply chain management software market was valued at $19.3 billion in 2023 and is projected to reach $31.7 billion by 2028.",
    "TechCorp's main competitors are Llamasoft (acquired by Coupa), Blue Yonder, and Kinaxis. TechCorp differentiates through AI prediction accuracy.",
    
    # Historical events  
    "In September 2023, TechCorp's ChainMind correctly predicted the semiconductor shortage in Taiwan 18 days before major disruptions occurred.",
    "TechCorp won the 'AI Startup of the Year' award at the 2024 Supply Chain Innovation Summit in Las Vegas.",
]


def search_knowledge(
    query: Annotated[str, "Search query to find relevant information in the knowledge base"]
) -> str:
    """
    Search the knowledge base for information relevant to the query.
    Returns the most relevant facts found. Use this to look up specific
    information about TechCorp, ChainMind, or supply chain topics.
    """
    # Simple keyword-based retrieval (production would use embeddings)
    query_words = set(query.lower().split())
    
    scored_docs = []
    for doc in KNOWLEDGE_BASE:
        doc_words = set(doc.lower().split())
        # Score by number of matching words
        score = len(query_words & doc_words)
        if score > 0:
            scored_docs.append((score, doc))
    
    # Sort by score and take top 3
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in scored_docs[:3]]
    
    if not top_docs:
        return "No relevant information found in the knowledge base."
    
    return "Found the following relevant information:\n" + "\n---\n".join(top_docs)


# ──────────────────────────────────────────────────────────────────────────────
# Tool 3: Current Date/Time
# ──────────────────────────────────────────────────────────────────────────────
# Gives the agent awareness of the current time - useful for time-sensitive queries.

def get_current_datetime() -> str:
    """
    Get the current date and time. Use this when the question involves
    'today', 'now', 'current', or needs to know the present date/time.
    """
    now = datetime.now()
    return f"Current date and time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}"


# ──────────────────────────────────────────────────────────────────────────────
# Tool 4: Unit Converter
# ──────────────────────────────────────────────────────────────────────────────
# Demonstrates a tool with multiple parameters.

def convert_units(
    value: Annotated[float, "The numeric value to convert"],
    from_unit: Annotated[str, "The unit to convert from (e.g., 'miles', 'kg', 'celsius')"],
    to_unit: Annotated[str, "The unit to convert to (e.g., 'km', 'pounds', 'fahrenheit')"]
) -> str:
    """
    Convert a value from one unit to another. Supports:
    - Distance: miles <-> km, feet <-> meters
    - Weight: kg <-> pounds, grams <-> ounces
    - Temperature: celsius <-> fahrenheit
    """
    conversions = {
        ("miles", "km"): lambda x: x * 1.60934,
        ("km", "miles"): lambda x: x / 1.60934,
        ("feet", "meters"): lambda x: x * 0.3048,
        ("meters", "feet"): lambda x: x / 0.3048,
        ("kg", "pounds"): lambda x: x * 2.20462,
        ("pounds", "kg"): lambda x: x / 2.20462,
        ("grams", "ounces"): lambda x: x * 0.035274,
        ("ounces", "grams"): lambda x: x / 0.035274,
        ("celsius", "fahrenheit"): lambda x: (x * 9/5) + 32,
        ("fahrenheit", "celsius"): lambda x: (x - 32) * 5/9,
    }
    
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key](value)
        return f"{value} {from_unit} = {result:.4f} {to_unit}"
    else:
        return f"Cannot convert from {from_unit} to {to_unit}. Supported conversions: distance (miles/km, feet/meters), weight (kg/pounds, grams/ounces), temperature (celsius/fahrenheit)"


# ═══════════════════════════════════════════════════════════════════════════════
# 📊 TRAINING DATA FOR OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════
#
# To optimize with GEPA, we need examples of questions with expected answers.
# GEPA will learn from these to improve the agent's prompts.

TRAINING_DATA = [
    # Simple knowledge lookups
    dspy.Example(
        question="When was TechCorp founded?",
        expected_answer="2019"
    ).with_inputs("question"),
    
    dspy.Example(
        question="Who are the founders of TechCorp?",
        expected_answer="Sarah Chen and Marcus Williams"
    ).with_inputs("question"),
    
    dspy.Example(
        question="What is TechCorp's main product called?",
        expected_answer="ChainMind"
    ).with_inputs("question"),
    
    # Questions requiring calculation
    dspy.Example(
        question="What is 15% of TechCorp's $47 million ARR?",
        expected_answer="7.05 million"  # or $7,050,000
    ).with_inputs("question"),
    
    dspy.Example(
        question="If ChainMind's API rate limit is 1000 requests per minute, how many requests per second is that?",
        expected_answer="16.67"  # 1000/60
    ).with_inputs("question"),
    
    # Multi-step reasoning
    dspy.Example(
        question="What is TechCorp's revenue growth from 2022 to 2024 as a percentage?",
        expected_answer="291%"  # (47-12)/12 * 100
    ).with_inputs("question"),
    
    dspy.Example(
        question="How much total funding has TechCorp raised?",
        expected_answer="$73 million"
    ).with_inputs("question"),
    
    # Unit conversion
    dspy.Example(
        question="Convert 100 miles to kilometers",
        expected_answer="160.934"
    ).with_inputs("question"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# 🔧 STEP 2: Configure DSPy
# ═══════════════════════════════════════════════════════════════════════════════

def setup_dspy():
    """Configure DSPy with Gemini (matching your existing setup)."""
    load_dotenv()
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "❌ GEMINI_API_KEY not found!\n"
            "   1. Create a .env file in the project root\n"
            "   2. Add: GEMINI_API_KEY=your_key_here\n"
            "   3. Get a key at https://aistudio.google.com/apikey"
        )
    
    # Using Gemini 2.5 Flash - fast and capable for agent tasks
    lm = dspy.LM(model="gemini/gemini-2.5-flash", api_key=api_key)
    dspy.configure(lm=lm)
    
    print("✅ DSPy configured with Gemini 2.5 Flash")
    return lm


# ═══════════════════════════════════════════════════════════════════════════════
# 🤖 STEP 3: Create the Agentic RAG System
# ═══════════════════════════════════════════════════════════════════════════════
#
# dspy.ReAct creates an agent that:
#   1. Reads the question
#   2. Thinks about what tools might help
#   3. Calls tools and observes results
#   4. Repeats until it can answer
#   5. Returns the final answer
#
# The key insight: the LLM DECIDES which tools to use based on:
#   - The question asked
#   - The tool docstrings (it reads these!)
#   - Previous observations (what it learned from tool calls)

def create_agentic_rag():
    """
    Create an agentic RAG system using DSPy's ReAct.
    
    ReAct = Reasoning + Acting
    The agent thinks step by step while taking actions.
    """
    
    # Our toolkit - the agent can use any of these
    tools = [
        search_knowledge,  # RAG - search our knowledge base
        calculate,         # Math operations
        get_current_datetime,  # Time awareness
        convert_units,     # Unit conversions
    ]
    
    # Create the ReAct agent
    # - signature: defines input (question) → output (answer)
    # - tools: list of functions the agent can call
    # - max_iters: maximum reasoning steps (prevents infinite loops)
    
    agent = dspy.ReAct(
        signature="question -> answer",  # Simple: takes question, returns answer
        tools=tools,
        max_iters=10  # Allow up to 10 reasoning steps
    )
    
    return agent


# ═══════════════════════════════════════════════════════════════════════════════
# 🧬 STEP 4: GEPA Optimization
# ═══════════════════════════════════════════════════════════════════════════════
#
# GEPA (Genetic-Pareto Reflective Prompt Evolution) optimizes your agent by:
#   1. Running the agent on training examples
#   2. Comparing outputs to expected answers
#   3. Reflecting on WHY things went wrong (using LLM)
#   4. Proposing improved prompts based on that reflection
#   5. Repeating until performance improves
#
# The key difference from other optimizers: GEPA uses TEXTUAL FEEDBACK,
# not just scores. This helps it understand specific failure modes.

def metric_with_feedback(example, prediction, trace=None):
    """
    Evaluation metric for GEPA that provides textual feedback.
    
    GEPA is special because it can use FEEDBACK (not just scores) to improve.
    This metric checks if the answer contains the expected information
    and provides specific feedback about what went wrong.
    """
    expected = example.expected_answer.lower()
    predicted = prediction.answer.lower() if hasattr(prediction, 'answer') else str(prediction).lower()
    
    # Check if the expected answer is contained in the prediction
    # (We use containment rather than exact match for flexibility)
    is_correct = expected in predicted or predicted in expected
    
    # For numerical answers, try fuzzy matching
    try:
        # Extract numbers from both strings
        import re
        expected_nums = re.findall(r'[\d.]+', expected)
        predicted_nums = re.findall(r'[\d.]+', predicted)
        
        if expected_nums and predicted_nums:
            for exp_num in expected_nums:
                for pred_num in predicted_nums:
                    if abs(float(exp_num) - float(pred_num)) < 0.1:
                        is_correct = True
                        break
    except:
        pass
    
    score = 1.0 if is_correct else 0.0
    
    # Generate feedback for GEPA - this is the key part!
    # GEPA uses this feedback to understand HOW to improve
    if is_correct:
        feedback = f"✓ Correct! The answer '{predicted[:100]}' matches expected '{expected}'."
    else:
        feedback = (
            f"✗ Incorrect. Expected answer containing '{expected}', "
            f"but got '{predicted[:100]}'. "
            f"The agent may need to use the right tools or reason more carefully."
        )
    
    # Return ScoreWithFeedback for GEPA to use
    return ScoreWithFeedback(score=score, feedback=feedback)


def optimize_agent_with_gepa(agent, trainset=None, auto="light"):
    """
    Optimize the agent using GEPA.
    
    GEPA will:
    1. Run the agent on training examples
    2. Collect feedback about failures
    3. Reflect on the feedback using an LLM
    4. Propose improved prompts
    5. Evaluate and keep the best versions
    
    Args:
        agent: The dspy.ReAct agent to optimize
        trainset: Training examples (uses TRAINING_DATA if None)
        auto: Optimization budget - "light", "medium", or "heavy"
    
    Returns:
        The optimized agent
    """
    if trainset is None:
        trainset = TRAINING_DATA
    
    print(f"\n🧬 Starting GEPA optimization with {len(trainset)} training examples...")
    print(f"   Using '{auto}' optimization budget")
    
    # Create GEPA optimizer
    # - metric: Our feedback-providing metric
    # - auto: Controls optimization budget ("light", "medium", or "heavy")
    optimizer = GEPA(
        metric=metric_with_feedback,
        auto=auto,  # Use "medium" or "heavy" for more thorough optimization
    )
    
    # Compile (optimize) the agent
    # This runs the training loop and returns an improved agent
    optimized_agent = optimizer.compile(
        agent,
        trainset=trainset,
    )
    
    print("✅ GEPA optimization complete!")
    return optimized_agent


def evaluate_agent(agent, testset=None):
    """
    Evaluate the agent on a test set.
    
    Returns accuracy and shows detailed results.
    """
    if testset is None:
        testset = TRAINING_DATA  # Use same data for demo purposes
    
    print(f"\n📊 Evaluating agent on {len(testset)} examples...")
    
    correct = 0
    results = []
    
    for example in testset:
        try:
            prediction = agent(question=example.question)
            result = metric_with_feedback(example, prediction)
            
            if result.score >= 0.5:
                correct += 1
            
            results.append({
                'question': example.question,
                'expected': example.expected_answer,
                'predicted': prediction.answer if hasattr(prediction, 'answer') else str(prediction),
                'correct': result.score >= 0.5,
                'feedback': result.feedback
            })
        except Exception as e:
            results.append({
                'question': example.question,
                'expected': example.expected_answer,
                'predicted': f"Error: {e}",
                'correct': False,
                'feedback': f"Error during evaluation: {e}"
            })
    
    accuracy = correct / len(testset) * 100
    print(f"   Accuracy: {correct}/{len(testset)} = {accuracy:.1f}%")
    
    return accuracy, results


# ═══════════════════════════════════════════════════════════════════════════════
# 🎮 STEP 5: Demo the Agent
# ═══════════════════════════════════════════════════════════════════════════════

def print_divider(title: str = ""):
    """Print a visual divider."""
    print(f"\n{'═' * 70}")
    if title:
        print(f"  {title}")
        print('═' * 70)


def demo_simple_query(agent):
    """Demo: A simple knowledge-base lookup."""
    print_divider("📚 Demo 1: Simple Knowledge Lookup")
    
    question = "When was TechCorp founded and who are the founders?"
    print(f"\n❓ Question: {question}")
    
    result = agent(question=question)
    print(f"\n💬 Answer: {result.answer}")


def demo_multi_step_reasoning(agent):
    """Demo: Question requiring multiple tool calls."""
    print_divider("🧠 Demo 2: Multi-Step Reasoning")
    
    question = "What is TechCorp's revenue growth percentage from 2022 to 2024, and what is 10% of their current ARR?"
    print(f"\n❓ Question: {question}")
    print("   (This requires: 1) looking up revenue data, 2) calculating growth %, 3) calculating 10% of ARR)")
    
    result = agent(question=question)
    print(f"\n💬 Answer: {result.answer}")


def demo_mixed_tools(agent):
    """Demo: Combining knowledge search with calculations."""
    print_divider("🔀 Demo 3: Combining Multiple Tools")
    
    question = "TechCorp's ChainMind API has a rate limit. If I make 3 requests per second, how many minutes until I hit the limit?"
    print(f"\n❓ Question: {question}")
    print("   (This requires: 1) looking up the rate limit, 2) calculating time)")
    
    result = agent(question=question)
    print(f"\n💬 Answer: {result.answer}")


def demo_time_aware(agent):
    """Demo: Time-aware questions."""
    print_divider("⏰ Demo 4: Time-Aware Query")
    
    question = "How long ago was TechCorp founded? Give me the answer in years."
    print(f"\n❓ Question: {question}")
    print("   (This requires: 1) getting current date, 2) looking up founding date, 3) calculating)")
    
    result = agent(question=question)
    print(f"\n💬 Answer: {result.answer}")


def demo_unit_conversion(agent):
    """Demo: Unit conversions with context."""
    print_divider("📐 Demo 5: Unit Conversion with Context")
    
    question = "TechCorp has offices in Austin and London. If it's 25 celsius in London, what's that in fahrenheit?"
    print(f"\n❓ Question: {question}")
    
    result = agent(question=question)
    print(f"\n💬 Answer: {result.answer}")


def interactive_mode(agent):
    """Let the user ask their own questions."""
    print_divider("💬 INTERACTIVE MODE")
    
    print("""
Ask the agent anything! It has access to:
  📚 Knowledge base about TechCorp and ChainMind
  🔢 Calculator for math expressions
  ⏰ Current date/time
  📐 Unit converter

Type 'quit' to exit.
""")
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not question:
            continue
        if question.lower() in ('quit', 'exit', 'q'):
            break
        
        try:
            result = agent(question=question)
            print(f"\n💬 {result.answer}")
        except Exception as e:
            print(f"\n❌ Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   🤖 Agentic RAG with DSPy 3.x + GEPA Optimization                           ║
║   Understanding Agents, Tools, and Prompt Optimization                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

This tutorial demonstrates:
  • Agents that REASON about what to do (ReAct pattern)
  • Tools that agents can USE to gather information
  • GEPA optimization to improve agent performance

The key concepts:
  • Regular RAG: Always retrieves, then generates (fixed pipeline)
  • Agentic RAG: Decides WHEN and WHAT to retrieve (dynamic)
  • GEPA: Reflects on failures to improve prompts automatically
    """)
    
    # Setup
    setup_dspy()
    agent = create_agentic_rag()
    
    print("\n🔧 Agent created with tools:")
    print("   • search_knowledge - Search the TechCorp knowledge base")
    print("   • calculate - Evaluate math expressions")
    print("   • get_current_datetime - Get current date/time")
    print("   • convert_units - Convert between units")
    
    # Ask user what they want to do
    print_divider("Choose Mode")
    print("""
What would you like to do?

  1. Run demos (see the agent in action)
  2. Run GEPA optimization (improve the agent)
  3. Both (demos first, then optimization)
  4. Interactive mode only
    """)
    
    try:
        choice = input("Enter choice (1-4) [default: 1]: ").strip() or "1"
    except (EOFError, KeyboardInterrupt):
        choice = "1"
    
    if choice in ("1", "3"):
        # Run demos
        demo_simple_query(agent)
        demo_multi_step_reasoning(agent)
        demo_mixed_tools(agent)
        demo_time_aware(agent)
        demo_unit_conversion(agent)
    
    if choice in ("2", "3"):
        # Run GEPA optimization
        print_divider("🧬 GEPA Optimization")
        print("""
GEPA (Genetic-Pareto Reflective Prompt Evolution) will:
  1. Run the agent on training examples
  2. Analyze what went wrong using LLM reflection
  3. Propose improved prompts based on feedback
  4. Keep the best-performing versions

This may take a few minutes and uses additional API calls.
        """)
        
        try:
            proceed = input("Run GEPA optimization? (y/n) [default: y]: ").strip().lower() or "y"
        except (EOFError, KeyboardInterrupt):
            proceed = "n"
            
        if proceed == "y":
            # Evaluate before optimization
            print("\n📊 Baseline evaluation (before optimization):")
            baseline_acc, _ = evaluate_agent(agent)
            
            # Run GEPA optimization
            optimized_agent = optimize_agent_with_gepa(agent, auto="light")
            
            # Evaluate after optimization
            print("\n📊 Post-optimization evaluation:")
            optimized_acc, _ = evaluate_agent(optimized_agent)
            
            improvement = optimized_acc - baseline_acc
            print(f"\n📈 Improvement: {improvement:+.1f}% accuracy")
            
            # Use optimized agent for interactive mode
            agent = optimized_agent
    
    if choice == "4" or choice in ("1", "3"):
        # Interactive mode
        print_divider()
        try:
            response = input("\nWant to ask your own questions? (y/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            response = "n"
        if response == 'y':
            interactive_mode(agent)
    
    print("\n" + "═" * 70)
    print("  📖 Tutorial Complete!")
    print("═" * 70)
    print("""
Key Takeaways:
  1. Tools are just Python functions with docstrings
  2. dspy.ReAct creates an agent that reasons + acts
  3. The agent reads tool docstrings to decide what to call
  4. Multi-step reasoning happens automatically
  5. GEPA optimizes prompts using reflective feedback
  6. This is the foundation of AI agents!

Next steps:
  • Try adding your own tools
  • Experiment with different questions
  • Run GEPA with more training data for better results
  • Check out the DSPy docs: https://dspy.ai/
    """)


if __name__ == "__main__":
    main()

