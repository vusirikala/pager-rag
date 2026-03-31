import os
import json
import asyncio
from datetime import timedelta
from dateutil import parser
from elasticsearch import AsyncElasticsearch
from google import genai
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.markdown import Markdown

load_dotenv()
console = Console()

es_client = AsyncElasticsearch("http://localhost:9200")
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

INDEX_NAME = "pager-rag-logs"
LLM_MODEL = "gemini-2.5-flash"


async def retrieve_by_trace(trace_ids: list[str], env: str, size: int = 200) -> list[dict]:
    """Deterministic retrieval: fetch all logs matching specific trace IDs."""
    result = await es_client.search(
        index=INDEX_NAME,
        query={
            "bool": {
                "filter": [
                    {"term": {"environment": env}},
                    {"terms": {"trace_id": list(trace_ids)}}
                ]
            }
        },
        sort=[{"timestamp": "asc"}],
        size=size,
    )
    return result["hits"]["hits"]


async def retrieve_by_time_window(service: str, env: str, start: str, end: str, size: int = 200) -> list[dict]:
    """Deterministic retrieval: fetch all logs for a service within a time window."""
    result = await es_client.search(
        index=INDEX_NAME,
        query={
            "bool": {
                "filter": [
                    {"term": {"environment": env}},
                    {"term": {"service": service}},
                    {"range": {"timestamp": {"gte": start, "lte": end}}}
                ]
            }
        },
        sort=[{"timestamp": "asc"}],
        size=size,
    )
    return result["hits"]["hits"]


async def retrieve_errors_in_window(env: str, start: str, end: str, size: int = 50) -> list[dict]:
    """Fetch ERROR/CRITICAL logs across all services in a time window to find correlated failures."""
    result = await es_client.search(
        index=INDEX_NAME,
        query={
            "bool": {
                "filter": [
                    {"term": {"environment": env}},
                    {"terms": {"level": ["ERROR", "CRITICAL", "FATAL"]}},
                    {"range": {"timestamp": {"gte": start, "lte": end}}}
                ]
            }
        },
        sort=[{"timestamp": "asc"}],
        size=size,
    )
    return result["hits"]["hits"]


async def execute_rag(alert: dict):
    console.rule(f"[bold red]Investigating Incident: {alert['title']}")

    alert_desc = alert.get("description", "")
    env = alert.get("environment", "unknown")
    service = alert.get("service_affected", "")
    alert_time = parser.parse(alert.get("created_at"))

    start_time = (alert_time - timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    end_time = (alert_time + timedelta(minutes=1)).isoformat().replace("+00:00", "Z")

    # ---------------------------------------------------------
    # STEP 1: Pull all logs for the affected service in the time window
    # ---------------------------------------------------------
    console.print(f"\n[cyan]1. Time-Window Retrieval:[/cyan] Fetching all '{service}' logs from [{start_time}] to [{end_time}] in '{env}'...")

    service_logs = await retrieve_by_time_window(service, env, start_time, end_time)
    console.print(f"   Found [bold]{len(service_logs)}[/bold] logs for '{service}'.")

    # ---------------------------------------------------------
    # STEP 2: Extract trace IDs and expand across microservices
    # ---------------------------------------------------------
    console.print("\n[cyan]2. Trace Expansion:[/cyan] Extracting trace IDs to follow requests across all microservices...")

    trace_ids = set()
    for hit in service_logs:
        tid = hit["_source"].get("trace_id", "")
        if tid:
            trace_ids.add(tid)

    trace_logs = []
    if trace_ids:
        console.print(f"   Found [bold]{len(trace_ids)}[/bold] unique trace IDs. Querying for the full distributed traces...")
        trace_logs = await retrieve_by_trace(list(trace_ids), env)
        console.print(f"   Expanded to [bold]{len(trace_logs)}[/bold] logs across all services.")

    # ---------------------------------------------------------
    # STEP 3: Find correlated errors across OTHER services
    # ---------------------------------------------------------
    console.print("\n[cyan]3. Correlated Errors:[/cyan] Scanning for ERROR/CRITICAL logs across all services in the same window...")

    error_logs = await retrieve_errors_in_window(env, start_time, end_time)
    console.print(f"   Found [bold]{len(error_logs)}[/bold] error-level logs across the environment.")

    # ---------------------------------------------------------
    # STEP 4: Deduplicate and build chronological timeline
    # ---------------------------------------------------------
    seen_ids = set()
    all_logs = []
    for hit in trace_logs + service_logs + error_logs:
        doc_id = hit["_id"]
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            all_logs.append(hit)

    all_logs.sort(key=lambda x: x["_source"].get("timestamp", ""))

    context_str = ""
    for hit in all_logs:
        log = hit["_source"]
        ts = log.get("timestamp", "")
        svc = log.get("service", "unknown")
        level = log.get("level", "INFO")
        msg = log.get("message", "")
        context_str += f"[{ts}] {svc} [{level}]: {msg}\n"

    console.print(f"\n   Compiled a timeline of [bold]{len(all_logs)}[/bold] unique, chronologically-sorted logs.")

    if not all_logs:
        console.print("[red]No logs found for this alert. Check that ingestion has completed.[/red]")
        return

    # ---------------------------------------------------------
    # STEP 5: Send to LLM for Root Cause Analysis
    # ---------------------------------------------------------
    console.print("\n[cyan]4. Reasoning:[/cyan] Sending the timeline to Gemini for Root Cause Analysis...\n")

    prompt = f"""You are an expert Site Reliability Engineer investigating a PagerDuty alert.

ALERT DETAILS:
- Title: {alert.get('title')}
- Description: {alert_desc}
- Service Affected: {service}
- Environment: {env}
- Time: {alert.get('created_at')}

Below is the exact chronological timeline of logs and Kubernetes events from the affected service,
its distributed traces across microservices, and any correlated errors in the environment.

LOG TIMELINE ({len(all_logs)} logs):
{context_str}

Please provide:
1. A clear, concise summary of the Root Cause.
2. The timeline of events (what happened first, how it cascaded).
3. Suggested immediate mitigation steps.
"""

    response = await gemini_client.aio.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )

    console.print(Panel(Markdown(response.text), title="Root Cause Analysis", border_style="green"))


async def main():
    try:
        with open("data/alerts.json", "r") as f:
            alerts = json.load(f)
    except FileNotFoundError:
        console.print("[red]data/alerts.json not found. Run generate_mock_data.py first.[/red]")
        return

    if not alerts:
        console.print("No alerts found in the mock data.")
        return

    while True:
        table = Table(title="Active PagerDuty Alerts")
        table.add_column("ID", justify="right", style="cyan", no_wrap=True)
        table.add_column("Env", style="magenta")
        table.add_column("Service", style="yellow")
        table.add_column("Alert", style="red")

        for i, alert in enumerate(alerts):
            table.add_row(
                str(i),
                alert.get("environment", "unknown"),
                alert.get("service_affected", "unknown"),
                alert.get("title", "Alert")
            )

        console.print(table)

        choice = Prompt.ask("Enter Alert ID to investigate (or 'q' to quit)")
        if choice.lower() == 'q':
            break

        try:
            alert_idx = int(choice)
            if 0 <= alert_idx < len(alerts):
                await execute_rag(alerts[alert_idx])
            else:
                console.print("[red]Invalid ID.[/red]")
        except ValueError:
            console.print("[red]Please enter a number.[/red]")

    await es_client.close()

if __name__ == "__main__":
    asyncio.run(main())
