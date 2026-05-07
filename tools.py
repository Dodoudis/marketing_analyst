import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# BigQuery client (lazy init so SQLite fallback works without GCP credentials)
# ---------------------------------------------------------------------------

def _get_bq_client():
    from google.cloud import bigquery
    from google.oauth2 import service_account

    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not key_path:
        raise EnvironmentError("GOOGLE_APPLICATION_CREDENTIALS not set in .env")

    credentials = service_account.Credentials.from_service_account_file(
        key_path,
        scopes=["https://www.googleapis.com/auth/bigquery"]
    )
    project = os.getenv("GCP_PROJECT_ID")
    return bigquery.Client(credentials=credentials, project=project)


# ---------------------------------------------------------------------------
# SQLite fallback (uses marketing_sample_data.csv loaded into a local DB)
# ---------------------------------------------------------------------------

def _get_sqlite_connection():
    import sqlite3
    db_path = os.getenv("SQLITE_DB_PATH", "marketing.db")
    return sqlite3.connect(db_path)


def _use_bigquery() -> bool:
    return os.getenv("USE_BIGQUERY", "false").lower() == "true"


# ---------------------------------------------------------------------------
# Tool 1: get_schema
# Returns table schema as a string injected into the LLM system prompt.
# ---------------------------------------------------------------------------

def get_schema() -> str:
    """
    Returns the schema of the tables as a plain-text string
    the LLM can use to write accurate SQL.
    """
    schema = """
Tables and views available in the marketing dataset:

1. mart_campaign_performance  — RAW. One row per day per campaign. Use only 
                                for daily granularity or when no aggregated 
                                view covers the question.

2. mart_campaign_daily        — One row per day per campaign (same as raw 
                                but KPIs recalculated cleanly). Use for 
                                daily trend queries.

3. mart_campaign_monthly      — One row per month per campaign. Use for 
                                period comparisons (Q1 vs Q2, month-over-month).

4. mart_channel_summary       — One row per channel, full period. Use for 
                                top-level channel performance questions.

5. mart_monthly_channel_summary — One row per month per channel. Use for 
                                  channel trend questions without daily noise.

COLUMN REFERENCE (all views share the same columns unless noted):

  date / month          DATE        — day or first day of month
  channel               STRING      — 'Google Ads', 'LinkedIn Ads', 'Meta Ads', 'Criteo'
  campaign_name         STRING      — campaign name (not in channel summary views)
  spend                 FLOAT       — EUR
  impressions           INTEGER
  clicks                INTEGER
  leads                 INTEGER
  mqls                  INTEGER
  customers             INTEGER
  revenue               FLOAT       — EUR
  cac                   FLOAT       — cost per customer (NULL when customers = 0)
  roas                  FLOAT       — revenue / spend
  ctr                   FLOAT       — clicks / impressions (decimal, e.g. 0.05 = 5%)
  lead_to_mql_rate      FLOAT       — decimal
  mql_to_customer_rate  FLOAT       — decimal
  avg_cpc               FLOAT       — channel summary views only
  cost_per_lead         FLOAT       — channel summary views only
  cost_per_mql          FLOAT       — channel summary views only

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
# Executes SQL and returns a pandas DataFrame.
# ---------------------------------------------------------------------------

def run_query(sql: str) -> dict:
    """
    Executes a SQL SELECT query against BigQuery or the local SQLite fallback.

    Returns a dict with:
      - success (bool)
      - data (list of dicts) — rows as records, empty list on error
      - columns (list of str)
      - row_count (int)
      - error (str or None)
    """
    sql = sql.strip()

    # Block any write operations
    forbidden = ["insert", "update", "delete", "drop", "create", "alter", "truncate"]
    if any(kw in sql.lower() for kw in forbidden):
        return {
            "success": False,
            "data": [],
            "columns": [],
            "row_count": 0,
            "error": "Only SELECT queries are permitted.",
        }

    try:
        if _use_bigquery():
            client = _get_bq_client()
            project = os.getenv("GCP_PROJECT_ID")
            dataset = os.getenv("BQ_DATASET", "aiagent")
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
                    table,
                    f"`{project}.{dataset}.{table}`",
                )
            df = client.query(qualified_sql).to_dataframe()
        else:
            conn = _get_sqlite_connection()
            df = pd.read_sql_query(sql, conn)
            conn.close()

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
            "success": False,
            "data": [],
            "columns": [],
            "row_count": 0,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Tool 3: summarise
# Converts a query result dict into a readable string for the LLM.
# ---------------------------------------------------------------------------

def summarise(result: dict, max_rows: int = 50) -> str:
    """
    Converts a run_query result dict into a clean string the LLM can read
    and use to write its final answer.

    Caps output at max_rows rows to keep the LLM context manageable.
    """
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
# Tool definitions for the Claude API (tool_use format)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "get_schema",
        "description": (
            "Returns the schema of the marketing data tables. "
            "Always call this first before writing any SQL so you know "
            "the exact column names, types, and business definitions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "run_query",
        "description": (
            "Executes a SQL SELECT query against the marketing dataset. "
            "Use standard SQL compatible with BigQuery. "
            "When aggregating KPIs like CAC or ROAS across rows, always "
            "recalculate from raw metrics (e.g. SUM(spend) / NULLIF(SUM(customers), 0)) "
            "rather than averaging the pre-calculated columns. "
            "Returns rows as a list of dicts plus success/error status."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "The SQL SELECT query to execute.",
                }
            },
            "required": ["sql"],
        },
    },
    {
        "name": "summarise",
        "description": (
            "Converts a run_query result into a readable string. "
            "Call this after run_query to format the data before writing "
            "your final answer to the user."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "result": {
                    "type": "object",
                    "description": "The dict returned by run_query.",
                },
                "max_rows": {
                    "type": "integer",
                    "description": "Max rows to include in the summary. Default 50.",
                },
            },
            "required": ["result"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool dispatcher — called by the agent loop
# ---------------------------------------------------------------------------

def dispatch_tool(tool_name: str, tool_input: dict) -> str:
    """
    Routes a tool call from the LLM to the correct Python function.
    Returns the result as a string.
    """
    if tool_name == "get_schema":
        return get_schema()

    elif tool_name == "run_query":
        result = run_query(tool_input["sql"])
        return str(result)

    elif tool_name == "summarise":
        result = tool_input["result"]
        max_rows = tool_input.get("max_rows", 50)
        # result may arrive as a string if the LLM serialised it
        if isinstance(result, str):
            import ast
            try:
                result = ast.literal_eval(result)
            except Exception:
                return "Could not parse query result for summarisation."
        return summarise(result, max_rows=max_rows)

    else:
        return f"Unknown tool: {tool_name}"
