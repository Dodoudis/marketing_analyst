"""
tools.py

Tool functions for the Marketing Analytics Agent.
Uses Streamlit secrets for all credentials.
"""

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# BigQuery client
# ---------------------------------------------------------------------------

def _get_bq_client():
    from google.cloud import bigquery
    from google.oauth2 import service_account

    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/bigquery"]
    )
    project = st.secrets["GCP_PROJECT_ID"]
    return bigquery.Client(credentials=credentials, project=project)


# ---------------------------------------------------------------------------
# Tool 1: get_schema
# ---------------------------------------------------------------------------

def get_schema() -> str:
    schema = """
Tables and views available in the marketing dataset:

1. mart_campaign_performance    — RAW. One row per day per campaign. Use only
                                  for daily granularity or when no aggregated
                                  view covers the question.

2. mart_campaign_daily          — One row per day per campaign (KPIs recalculated
                                  cleanly). Use for daily trend queries.

3. mart_campaign_monthly        — One row per month per campaign. Use for
                                  period comparisons (Q1 vs Q2, month-over-month).

4. mart_channel_summary         — One row per channel, full period. Use for
                                  top-level channel performance questions.

5. mart_monthly_channel_summary — One row per month per channel. Use for
                                  channel trend questions without daily noise.

COLUMN REFERENCE (all views share the same columns unless noted):

  date / month          DATE    — day or first day of month
  channel               STRING  — 'Google Ads', 'LinkedIn Ads', 'Meta Ads', 'Criteo'
  campaign_name         STRING  — campaign name (not in channel summary views)
  spend                 FLOAT   — EUR
  impressions           INTEGER
  clicks                INTEGER
  leads                 INTEGER
  mqls                  INTEGER
  customers             INTEGER
  revenue               FLOAT   — EUR
  cac                   FLOAT   — cost per customer (NULL when customers = 0)
  roas                  FLOAT   — revenue / spend
  ctr                   FLOAT   — clicks / impressions (decimal, e.g. 0.05 = 5%)
  lead_to_mql_rate      FLOAT   — decimal
  mql_to_customer_rate  FLOAT   — decimal
  avg_cpc               FLOAT   — channel summary views only
  cost_per_lead         FLOAT   — channel summary views only
  cost_per_mql          FLOAT   — channel summary views only

ROUTING RULES (always pick the smallest table that answers the question):
  - "which channel has best ROAS?"       → mart_channel_summary
  - "LinkedIn trend by month"            → mart_monthly_channel_summary
  - "best campaign in Q1"                → mart_campaign_monthly
  - "daily spend last 7 days"            → mart_campaign_daily
  - "raw row for specific date/campaign" → mart_campaign_performance
"""
    return schema.strip()


# ---------------------------------------------------------------------------
# Tool 2: run_query
# ---------------------------------------------------------------------------

def run_query(sql: str) -> dict:
    sql = sql.strip()

    # Block write operations
    forbidden = ["insert", "update", "delete", "drop", "create", "alter", "truncate"]
    if any(kw in sql.lower() for kw in forbidden):
        return {
            "success": False, "data": [], "columns": [],
            "row_count": 0, "error": "Only SELECT queries are permitted.",
        }

    try:
        client = _get_bq_client()
        project = st.secrets["GCP_PROJECT_ID"]
        dataset = st.secrets["BQ_DATASET"]

        table_names = [
            "mart_campaign_performance",
            "mart_campaign_daily",
            "mart_campaign_monthly",
            "mart_channel_summary",
            "mart_monthly_channel_summary",
        ]

        qualified_sql = sql
        for table in table_names:
            qualified_sql = qualified_sql.replace(
                table, f"`{project}.{dataset}.{table}`"
            )

        df = client.query(qualified_sql).to_dataframe()
        df = df.round(4)

        return {
            "success": True,
            "data": df.to_dict(orient="records"),
            "columns": list(df.columns),
            "row_count": len(df),
            "error": None,
        }

    except Exception as e:
        return {
            "success": False, "data": [], "columns": [],
            "row_count": 0, "error": str(e),
        }


# ---------------------------------------------------------------------------
# Tool 3: summarise
# ---------------------------------------------------------------------------

def summarise(result: dict, max_rows: int = 50) -> str:
    if not result["success"]:
        return f"Query failed: {result['error']}"

    if result["row_count"] == 0:
        return "The query returned no rows."

    df = pd.DataFrame(result["data"])

    truncated = False
    if len(df) > max_rows:
        df = df.head(max_rows)
        truncated = True

    summary = f"Query returned {result['row_count']} row(s).\n\n"
    summary += df.to_string(index=False)

    if truncated:
        summary += f"\n\n[Showing first {max_rows} rows only]"

    return summary


# ---------------------------------------------------------------------------
# Tool definitions for the Claude API
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "get_schema",
        "description": (
            "Returns the schema of the marketing data tables. "
            "Call this first before writing any SQL to confirm column names and types."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run_query",
        "description": (
            "Executes a SQL SELECT query against the marketing dataset in BigQuery. "
            "When aggregating KPIs like CAC or ROAS across rows, always recalculate "
            "from raw metrics: SUM(spend) / NULLIF(SUM(customers), 0). "
            "Returns rows as a list of dicts plus success/error status."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "The SQL SELECT query to execute."}
            },
            "required": ["sql"],
        },
    },
    {
        "name": "summarise",
        "description": (
            "Converts a run_query result into a readable string. "
            "Call this after run_query to format data before writing the final answer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "result": {"type": "object", "description": "The dict returned by run_query."},
                "max_rows": {"type": "integer", "description": "Max rows to include. Default 50."},
            },
            "required": ["result"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

def dispatch_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "get_schema":
        return get_schema()

    elif tool_name == "run_query":
        result = run_query(tool_input["sql"])
        return str(result)

    elif tool_name == "summarise":
        result = tool_input["result"]
        max_rows = tool_input.get("max_rows", 50)
        if isinstance(result, str):
            import ast
            try:
                result = ast.literal_eval(result)
            except Exception:
                return "Could not parse query result for summarisation."
        return summarise(result, max_rows=max_rows)

    else:
        return f"Unknown tool: {tool_name}"
