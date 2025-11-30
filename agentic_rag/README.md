# 🤖 Agentic RAG with DSPy 3.x + GEPA Optimization

This tutorial teaches you how **agentic RAG** works - where an AI agent **reasons** about what information it needs and **decides** which tools to use. It also shows how to **optimize** the agent using **GEPA** (Genetic-Pareto Reflective Prompt Evolution).

## What's the Difference?

| Regular RAG | Agentic RAG | + GEPA |
|-------------|-------------|--------|
| Fixed pipeline: Retrieve → Generate | Dynamic: Think → Act → Observe → Repeat | Learns from failures |
| Always retrieves the same way | Decides IF and WHAT to retrieve | Improves over time |
| One-shot process | Multi-step reasoning | Reflective optimization |
| Simple questions only | Complex, multi-part questions | Better prompts automatically |

## The ReAct Pattern

This example uses the **ReAct** (Reasoning + Acting) pattern:

```
Question: "What's 10% of TechCorp's current ARR?"

THOUGHT: I need to find TechCorp's ARR first
ACTION: search_knowledge("TechCorp ARR revenue")
OBSERVATION: "TechCorp's ARR grew from $12M to $47M in 2024..."

THOUGHT: Now I can calculate 10% of $47M
ACTION: calculate("47 * 0.10")
OBSERVATION: "Result: 4.7"

THOUGHT: I have the answer now
ANSWER: "10% of TechCorp's current ARR ($47M) is $4.7 million"
```

## Tools in This Example

| Tool | What It Does |
|------|--------------|
| `search_knowledge` | Searches a knowledge base about TechCorp |
| `calculate` | Evaluates math expressions (sqrt, sin, etc.) |
| `get_current_datetime` | Returns current date/time |
| `convert_units` | Converts between units (miles↔km, °C↔°F) |

## Running the Example

```bash
# Make sure you're in the project root with your venv activated
cd /path/to/RAG-DSPy

# Install dependencies (if not already installed)
pip install -r agentic_rag/requirements.txt

# Set your API key in .env file
echo "GEMINI_API_KEY=your_key_here" > .env

# Run the example
python agentic_rag/agentic_rag_example.py
```

## Key Concepts

### 1. Tools are Python Functions

```python
def calculate(expression: str) -> str:
    """Evaluate a math expression."""  # Agent reads this!
    return str(eval(expression))
```

The agent:
- Reads the function name and docstring
- Understands what the tool does
- Decides when to call it based on the question

### 2. dspy.ReAct Creates the Agent

```python
agent = dspy.ReAct(
    signature="question -> answer",  # Input/output spec
    tools=[search_knowledge, calculate, ...],  # Available tools
    max_iters=10  # Max reasoning steps
)
```

### 3. The Agent Loop

```python
result = agent(question="How old is TechCorp?")
# Internally:
# 1. Reason about the question
# 2. Decide to call get_current_datetime
# 3. Observe the result
# 4. Decide to call search_knowledge
# 5. Observe founding date
# 6. Calculate the difference
# 7. Return final answer
```

## Adding Your Own Tools

```python
def my_custom_tool(param: str) -> str:
    """Describe what this tool does. Be specific!"""
    # Your logic here
    return "Result as a string"

# Add to the tools list
tools = [search_knowledge, calculate, my_custom_tool]
agent = dspy.ReAct(signature="question->answer", tools=tools)
```

## GEPA Optimization

GEPA (Genetic-Pareto Reflective Prompt Evolution) improves your agent automatically:

```python
from dspy import GEPA

# Define a metric that provides feedback
def metric_with_feedback(example, prediction, trace=None):
    is_correct = expected in prediction.answer.lower()
    feedback = "Correct!" if is_correct else f"Expected {expected}, got {prediction.answer}"
    return dspy.Prediction(score=1.0 if is_correct else 0.0, feedback=feedback)

# Create optimizer
optimizer = GEPA(
    metric=metric_with_feedback,
    max_iterations=3,
    num_candidates=3,
)

# Optimize the agent
optimized_agent = optimizer.compile(agent, trainset=training_examples)
```

### How GEPA Works

1. **Run** the agent on training examples
2. **Collect feedback** about what went wrong (not just scores!)
3. **Reflect** using an LLM to understand failure patterns
4. **Propose** improved prompts based on reflection
5. **Evaluate** and keep the best versions

The key insight: GEPA uses **textual feedback**, not just pass/fail scores. This helps it understand *why* things went wrong and fix specific issues.

## Learn More

- [DSPy Documentation](https://dspy.ai/)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)
- [ReAct Paper](https://arxiv.org/abs/2210.03629) - The original research
- [GEPA Paper](https://arxiv.org/abs/2504.00536) - Reflective Prompt Evolution

