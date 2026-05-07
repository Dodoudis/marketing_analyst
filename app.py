"""
app.py

Streamlit UI for the Marketing Analytics Agent.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import ast
import anthropic
from tools import TOOL_DEFINITIONS, dispatch_tool, get_schema

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Marketing Analytics Agent | Dodoudis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

.stApp { background-color: #0a0a0f; color: #e8e6f0; }

section[data-testid="stSidebar"] {
    background-color: #0f0f1a;
    border-right: 1px solid #1e1e30;
}
section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }

.agent-title {
    font-size: 1.6rem; font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; letter-spacing: -0.02em; margin: 0;
}
.agent-subtitle {
    font-size: 0.75rem; color: #6b6b8a; margin: 0;
    font-family: 'DM Mono', monospace; letter-spacing: 0.05em;
}

.msg-user {
    background: #13131f; border: 1px solid #1e1e30;
    border-radius: 12px 12px 4px 12px;
    padding: 14px 18px; margin: 12px 0; margin-left: 15%;
    font-size: 0.92rem; line-height: 1.6; color: #c4c0d8;
}
.msg-agent {
    background: #0f0f1a; border: 1px solid #252540;
    border-left: 3px solid #a78bfa;
    border-radius: 4px 12px 12px 12px;
    padding: 14px 18px; margin: 12px 0; margin-right: 5%;
    font-size: 0.92rem; line-height: 1.7; color: #d4d0e8;
}
.msg-label {
    font-family: 'DM Mono', monospace; font-size: 0.65rem;
    letter-spacing: 0.1em; text-transform: uppercase;
    margin-bottom: 6px; opacity: 0.5;
}
.msg-label-user { color: #60a5fa; }
.msg-label-agent { color: #a78bfa; }

.tool-log {
    background: #080810; border: 1px solid #1a1a2e;
    border-radius: 8px; padding: 10px 14px; margin: 6px 0;
    font-family: 'DM Mono', monospace; font-size: 0.72rem; color: #4ade80;
}

.stButton > button {
    background: #0f0f1a !important; border: 1px solid #252540 !important;
    color: #9d9ab8 !important; border-radius: 8px !important;
    font-size: 0.78rem !important; font-family: 'Syne', sans-serif !important;
    padding: 8px 12px !important; text-align: left !important;
    width: 100% !important; transition: all 0.15s ease !important;
}
.stButton > button:hover {
    border-color: #a78bfa !important; color: #e8e6f0 !important;
    background: #13131f !important;
}

.metric-card {
    background: #0f0f1a; border: 1px solid #1e1e30;
    border-radius: 10px; padding: 16px; text-align: center;
}
.metric-value { font-size: 1.4rem; font-weight: 700; color: #a78bfa; }
.metric-label {
    font-size: 0.7rem; color: #6b6b8a;
    font-family: 'DM Mono', monospace; letter-spacing: 0.05em;
    text-transform: uppercase; margin-top: 4px;
}

.status-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #0a1a0a; border: 1px solid #1a3a1a;
    border-radius: 20px; padding: 4px 12px;
    font-family: 'DM Mono', monospace; font-size: 0.68rem;
    color: #4ade80; letter-spacing: 0.05em;
}
.status-dot {
    width: 6px; height: 6px; background: #4ade80;
    border-radius: 50%; animation: pulse 2s infinite;
}
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

.sql-block {
    background: #080810; border: 1px solid #1a1a2e;
    border-radius: 8px; padding: 12px 16px;
    font-family: 'DM Mono', monospace; font-size: 0.75rem;
    color: #7dd3fc; white-space: pre-wrap; word-break: break-all;
    margin-top: 8px;
}

.section-label {
    font-family: 'DM Mono', monospace; font-size: 0.65rem;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: #4a4a6a; margin-bottom: 12px; margin-top: 24px;
}

hr { border-color: #1e1e30 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2048
MAX_TOOL_ROUNDS = 8

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
- If a query returns no rows, say so and suggest why.
- If a query fails, read the error, fix the SQL, and retry once.
- Lead with the key insight, then supporting data.

DATABASE SCHEMA:
{get_schema()}
"""

EXAMPLE_QUESTIONS = [
    "Which channel had the best ROAS in 2024?",
    "What was the CAC by channel last quarter?",
    "Show me MQL trend by month for LinkedIn Ads.",
    "Which campaign had the lowest cost per MQL?",
    "Compare spend vs revenue by channel.",
    "What's the lead to MQL conversion rate by channel?",
]

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "last_sql" not in st.session_state:
    st.session_state.last_sql = None

# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_agent(user_question: str):
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

    st.session_state.conversation_history.append(
        {"role": "user", "content": user_question}
    )

    tool_rounds = 0
    tool_log = []
    last_sql = None

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=st.session_state.conversation_history,
        )

        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": response.content,
        })

        if response.stop_reason == "end_turn":
            final_answer = _extract_text(response.content)
            return final_answer, tool_log, last_sql

        if response.stop_reason == "tool_use":
            if tool_rounds >= MAX_TOOL_ROUNDS:
                # Reset broken history to prevent cascade errors
                st.session_state.conversation_history = []
                return (
                    "I reached the maximum number of tool calls. "
                    "Conversation has been reset — please try again.",
                    tool_log,
                    last_sql,
                )

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_name = block.name
                tool_input = block.input

                if tool_name == "run_query" and "sql" in tool_input:
                    last_sql = tool_input["sql"]

                output = dispatch_tool(tool_name, tool_input)
                tool_log.append({
                    "tool": tool_name,
                    "input": tool_input,
                    "output": str(output)[:300],
                })

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(output),
                })

            st.session_state.conversation_history.append({
                "role": "user",
                "content": tool_results,
            })

            tool_rounds += 1
            continue

        return f"Unexpected stop reason: {response.stop_reason}", tool_log, last_sql


def _extract_text(content_blocks):
    return "\n".join(
        block.text for block in content_blocks if hasattr(block, "text")
    ).strip()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style="margin-bottom: 1.5rem; padding-bottom: 1.5rem; border-bottom: 1px solid #1e1e30;">
        <p class="agent-title">MKT Agent</p>
        <p class="agent-subtitle">CLAUDE + BIGQUERY</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="status-badge">
        <div class="status-dot"></div>
        CONNECTED TO BIGQUERY
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-label">Example Questions</p>', unsafe_allow_html=True)
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, key=f"btn_{q}"):
            st.session_state["pending_question"] = q

    st.markdown("---")
    st.markdown('<p class="section-label">Session</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.total_queries}</div>
            <div class="metric-label">Queries</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        turns = len([m for m in st.session_state.display_messages if m["role"] == "user"])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{turns}</div>
            <div class="metric-label">Turns</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    if st.button("🔄 Reset conversation", key="reset_btn"):
        st.session_state.conversation_history = []
        st.session_state.display_messages = []
        st.session_state.total_queries = 0
        st.session_state.last_sql = None
        st.rerun()

    st.markdown("---")
    st.markdown('<p class="section-label">Schema</p>', unsafe_allow_html=True)
    with st.expander("View tables"):
        st.markdown("""
        <div style="font-family: 'DM Mono', monospace; font-size: 0.7rem; color: #6b6b8a; line-height: 1.8;">
        mart_campaign_performance<br>
        mart_campaign_daily<br>
        mart_campaign_monthly<br>
        mart_channel_summary<br>
        mart_monthly_channel_summary
        </div>""", unsafe_allow_html=True)

    if st.session_state.last_sql:
        st.markdown('<p class="section-label">Last SQL</p>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="sql-block">{st.session_state.last_sql}</div>',
            unsafe_allow_html=True
        )

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.markdown("""
<div style="margin-bottom: 1.5rem;">
    <h1 style="font-size: 1.8rem; font-weight: 800; color: #e8e6f0; margin: 0; letter-spacing: -0.02em;">
        Marketing Analytics Agent | Dodoudis
    </h1>
    <p style="color: #6b6b8a; font-size: 0.85rem; margin-top: 4px; font-family: 'DM Mono', monospace;">
        Ask anything about your campaign performance
    </p>
</div>
""", unsafe_allow_html=True)

# Render chat history
for msg in st.session_state.display_messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="msg-user">
            <div class="msg-label msg-label-user">YOU</div>
            {msg["content"]}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="msg-agent">
            <div class="msg-label msg-label-agent">AGENT</div>
            {msg["content"]}
        </div>""", unsafe_allow_html=True)

        if msg.get("tool_log"):
            with st.expander(f"🔧 {len(msg['tool_log'])} tool calls", expanded=False):
                for t in msg["tool_log"]:
                    st.markdown(f"""
                    <div class="tool-log">
                    → {t['tool']}()<br>
                    {t['output'][:200]}{'...' if len(t['output']) > 200 else ''}
                    </div>""", unsafe_allow_html=True)

        if msg.get("dataframe") is not None:
            st.dataframe(msg["dataframe"], use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Handle input
# ---------------------------------------------------------------------------

pending = st.session_state.pop("pending_question", None)
user_input = st.chat_input("Ask about your campaign performance...") or pending

if user_input:
    st.session_state.display_messages.append({
        "role": "user", "content": user_input,
    })
    st.session_state.total_queries += 1

    with st.spinner("Thinking..."):
        answer, tool_log, last_sql = run_agent(user_input)

    if last_sql:
        st.session_state.last_sql = last_sql

    # Extract dataframe from last successful query for inline display
    df_result = None
    for t in reversed(tool_log):
        if t["tool"] == "run_query":
            try:
                raw = dispatch_tool("run_query", t["input"])
                result = ast.literal_eval(raw) if isinstance(raw, str) else raw
                if isinstance(result, dict) and result.get("success") and result.get("data"):
                    df_result = pd.DataFrame(result["data"])
            except Exception:
                pass
            break

    st.session_state.display_messages.append({
        "role": "assistant",
        "content": answer,
        "tool_log": tool_log,
        "dataframe": df_result,
    })

    st.rerun()
