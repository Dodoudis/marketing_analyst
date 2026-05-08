# Marketing Analytics Agent

A conversational AI agent that lets you query marketing campaign performance data in plain English. Ask questions about ROAS, CAC, MQL trends, and channel performance — the agent writes the SQL, queries BigQuery, and returns structured insights in real time.

Built with the Anthropic Claude API (tool use), BigQuery, and Streamlit.

---

## Demo

> [Video](https://www.loom.com/share/20d8c08797b341ada37a0f1764abd47a)

---

## What It Does

Instead of writing SQL or navigating dashboards, you ask questions:

- *"Which channel had the best ROAS in 2024?"*
- *"What was the CAC by channel last quarter?"*
- *"Show me MQL trend by month for LinkedIn Ads."*
- *"Which campaign had the lowest cost per MQL?"*

The agent handles the rest — it picks the right table, writes the SQL, executes it against BigQuery, and returns a plain-English answer with an inline data table.

---

## Architecture

```
User question
      │
      ▼
LLM Agent (Claude API)
  ├── get_schema()        — understands available tables and columns
  ├── run_query(sql)      — executes SQL against BigQuery
  └── summarise(result)   — formats rows for the final answer
      │
      ▼
BigQuery (5 views at different granularities)
      │
      ▼
Plain-English answer + data table
```

### Tool Use Loop

The agent runs autonomously in a loop:

1. Receives the user question and the full schema in the system prompt
2. Calls `get_schema()` if it needs to confirm column details
3. Writes a SQL query and calls `run_query()` — the function qualifies table names, blocks write operations, and returns a structured result dict
4. Calls `summarise()` to format the rows as readable text
5. Writes the final answer in plain language with key numbers

Failed queries are caught and returned to the LLM as error messages — the agent self-corrects and retries without crashing.

---

## Data Layer

Five BigQuery views at different granularities, each pre-calculating KPIs from raw metrics:

| View | Granularity | Use case |
|---|---|---|
| `mart_campaign_performance` | Day × Campaign | Raw data, daily drill-down |
| `mart_campaign_daily` | Day × Campaign | Daily trends with clean KPIs |
| `mart_campaign_monthly` | Month × Campaign | Period comparisons (Q1 vs Q2) |
| `mart_channel_summary` | Channel (full period) | Top-level channel benchmarking |
| `mart_monthly_channel_summary` | Month × Channel | Channel trends over time |

The system prompt includes routing rules that direct the LLM to the smallest table that answers each question — reducing BigQuery scan costs and LLM token usage.

### KPI Definitions

All KPIs are recalculated from raw metrics when aggregating across rows:

```sql
cac  = SUM(spend) / NULLIF(SUM(customers), 0)
roas = SUM(revenue) / NULLIF(SUM(spend), 0)
ctr  = SUM(clicks) / NULLIF(SUM(impressions), 0)
```

---

## Stack

| Layer | Technology |
|---|---|
| LLM | Anthropic Claude (`claude-sonnet-4-6`) |
| Tool use | Anthropic Python SDK |
| Database | Google BigQuery |
| Auth | Google Service Account (via Streamlit secrets) |
| UI | Streamlit |
| Data processing | pandas |
| Language | Python 3.12 |

---

## Project Structure

```
├── app.py                  — Streamlit UI and agent loop
├── tools.py                — Tool functions and Claude tool definitions
├── bigquery_views.sql      — BigQuery view definitions
├── marketing_sample_data.csv  — Synthetic dataset (4,026 rows)
├── requirements.txt
```

---

## Key Engineering Decisions

**Table routing in the system prompt**
Rather than always querying the raw table, the system prompt includes explicit routing rules. "Which channel has the best ROAS?" hits a 4-row summary view. "Show me daily spend last week" hits the daily view. This keeps queries fast and cheap.

**KPI enforcement at the prompt level**
The system prompt explicitly prohibits averaging pre-calculated KPI columns when grouping across rows. CAC, ROAS, and CTR are always recalculated from raw metrics — preventing a common LLM mistake that produces wrong numbers silently.

**Automatic history reset on failure**
If the agent hits the maximum tool rounds, the conversation history is automatically cleared before returning the error message. This prevents the `tool_use` / `tool_result` mismatch error that causes cascade failures on the next question.

**BigQuery client caching**
The BigQuery client is cached with `@st.cache_resource` so authentication only happens once per session, not on every query.

---

## Dataset

The synthetic dataset covers 12 months of daily campaign data (Jan–Dec 2024) across four channels and nine campaigns:

| Channel | Campaigns |
|---|---|
| Google Ads | Brand Search, Competitor Search, Display Retargeting |
| LinkedIn Ads | Lead Gen Form - CFO, Sponsored Content - IT, InMail - HR |
| Meta Ads | Prospecting - Lookalike, Retargeting - Site Visitors, DPA - Catalogue |
| Criteo | Retargeting - All Visitors, Dynamic Retargeting - Cart |

Each channel has a distinct performance profile (CPCs, lead rates, close rates, average deal size) reflecting realistic B2B fintech acquisition patterns.

---

## Author

**Moschos Dodoudis**
Growth & Performance Marketing | AI Engineering
[LinkedIn](https://linkedin.com/in/moschos-dodoudis)
