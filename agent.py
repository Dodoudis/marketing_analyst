"""
agent.py

Core agent loop for the Marketing Analytics Agent.
Handles the conversation, tool use, and response generation.

Usage:
    python agent.py
"""

import os
import json
from dotenv import load_dotenv
import anthropic
from tools import TOOL_DEFINITIONS, dispatch_tool, get_schema

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2048
MAX_TOOL_ROUNDS = 8  # prevents infinite loops if the LLM keeps calling tools

SYSTEM_PROMPT = f"""You are a marketing analytics assistant with direct access to a 
marketing performance database.

Your job is to answer questions about campaign performance clearly and accurately.
Always ground your answers in data — never guess or make up numbers.

WORKFLOW:
1. Call get_schema() first if you're unsure which table or column to use.
2. Call run_query() with a SQL SELECT to fetch the data you need.
3. Call summarise() to format the result.
4. Write a clear, concise answer in plain language. Include the key numbers.
   Format tables using markdown when returning multiple rows.

RULES:
- Always use the smallest table that answers the question (see routing rules in schema).
- When aggregating CAC, ROAS, or CTR across rows, recalculate from raw metrics:
    cac  = SUM(spend) / NULLIF(SUM(customers), 0)
    roas = SUM(revenue) / NULLIF(SUM(spend), 0)
    ctr  = SUM(clicks) / NULLIF(SUM(impressions), 0)
- Rates are decimals — multiply by 100 when presenting as percentages.
- If a query returns no rows, say so clearly and suggest why.
- If a query fails, read the error, fix the SQL, and retry once.
- Keep answers focused. Lead with the key insight, then the supporting data.

DATABASE SCHEMA:
{get_schema()}
"""

# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent(user_question: str, conversation_history: list) -> tuple[str, list]:
    """
    Runs one turn of the agent loop.

    Args:
        user_question:        The user's natural language question.
        conversation_history: List of prior messages (role/content dicts).
                              Pass [] for a fresh conversation.

    Returns:
        (answer, updated_history) — the LLM's final answer and the full
        updated conversation history for the next turn.
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Append the new user message
    conversation_history = conversation_history + [
        {"role": "user", "content": user_question}
    ]

    tool_rounds = 0

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=conversation_history,
        )

        # Append assistant response to history
        conversation_history.append({
            "role": "assistant",
            "content": response.content,
        })

        # If the LLM is done, return the final text answer
        if response.stop_reason == "end_turn":
            final_answer = _extract_text(response.content)
            return final_answer, conversation_history

        # If the LLM wants to use tools
        if response.stop_reason == "tool_use":
            if tool_rounds >= MAX_TOOL_ROUNDS:
                return (
                    "I reached the maximum number of tool calls without a final answer. "
                    "Try rephrasing your question.",
                    conversation_history,
                )

            tool_results = _execute_tool_calls(response.content)

            # Append tool results as a user message (Anthropic format)
            conversation_history.append({
                "role": "user",
                "content": tool_results,
            })

            tool_rounds += 1
            continue

        # Unexpected stop reason
        return (
            f"Unexpected stop reason: {response.stop_reason}",
            conversation_history,
        )


def _execute_tool_calls(content_blocks: list) -> list:
    """
    Finds all tool_use blocks in the response, dispatches each one,
    and returns a list of tool_result blocks in Anthropic format.
    """
    results = []
    for block in content_blocks:
        if block.type != "tool_use":
            continue

        tool_name = block.name
        tool_input = block.input

        print(f"\n  [tool] {tool_name}({_format_input(tool_input)})")

        output = dispatch_tool(tool_name, tool_input)

        print(f"  [result] {str(output)[:120]}{'...' if len(str(output)) > 120 else ''}")

        results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": str(output),
        })

    return results


def _extract_text(content_blocks: list) -> str:
    """Pulls plain text out of a response content block list."""
    parts = []
    for block in content_blocks:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts).strip()


def _format_input(tool_input: dict) -> str:
    """Formats tool input for readable terminal logging."""
    if "sql" in tool_input:
        sql = tool_input["sql"].replace("\n", " ").strip()
        return f'sql="{sql[:80]}{"..." if len(sql) > 80 else ""}"'
    return json.dumps(tool_input)[:100]


# ---------------------------------------------------------------------------
# Terminal chat interface
# ---------------------------------------------------------------------------

def main():
    print("\n Marketing Analytics Agent")
    print(" " + "─" * 40)
    print(" Type your question or 'quit' to exit.\n")

    example_questions = [
        "Which channel had the best ROAS in 2024?",
        "What was the CAC by channel last quarter?",
        "Show me MQL trend by month for LinkedIn Ads.",
        "Which campaign had the lowest cost per MQL?",
        "Compare spend vs revenue by channel.",
    ]

    print(" Example questions:")
    for q in example_questions:
        print(f"   • {q}")
    print()

    conversation_history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break

        if user_input.lower() == "reset":
            conversation_history = []
            print("Conversation reset.\n")
            continue

        print("\nAgent: thinking...\n")

        answer, conversation_history = run_agent(user_input, conversation_history)

        print(f"\nAgent: {answer}\n")
        print("─" * 50)


if __name__ == "__main__":
    main()
