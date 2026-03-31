import os
import json
import asyncio
from datetime import datetime, timedelta
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
EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"

async def get_embedding(text: str) -> list[float]:
    """Get vector embedding from Google Gemini."""
    try:
        from google.genai import types
        response = await gemini_client.aio.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=768)
        )
        if not response.embeddings:
            return [0.0] * 768
        return response.embeddings[0].values
    except Exception as e:
        console.print(f"[red]Failed to get embedding: {e}[/red]")
        return [0.0] * 768

async def execute_rag(alert: dict):
    console.rule(f"[bold red]Investigating Incident: {alert['title']}")
    
    # Extract details from the alert
    alert_desc = alert.get("description", "")
    env = alert.get("environment", "unknown")
    service = alert.get("service_affected", "")
    alert_time = parser.parse(alert.get("created_at"))
    
    # ---------------------------------------------------------
    # STEP 1: ANCHOR - Semantic Search for the exact error log
    # ---------------------------------------------------------
    console.print(f"\n[cyan]1. Anchoring:[/cyan] Searching Pinecone namespace '{env}' for error signatures matching the alert...")
    
    query_text = f"Error in {service}: {alert_desc}"
    query_vector = await get_embedding(query_text)
    
    # Define a time window (e.g., 5 minutes before the alert fired)
    start_time_iso = (alert_time - timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    end_time_iso = (alert_time + timedelta(minutes=1)).isoformat().replace("+00:00", "Z")
    
    # We want to find the specific error logs. We use semantic similarity + metadata filters
    # Elasticsearch KNN requires the query vector
    query_body = {
        "size": 5,
        "query": {
            "bool": {
                "must": [
                    {
                        "knn": {
                            "field": "embedding",
                            "query_vector": query_vector,
                            "k": 5,
                            "num_candidates": 50
                        }
                    }
                ],
                "filter": [
                    {"term": {"environment": env}},
                    {"range": {"timestamp": {"gte": start_time_iso, "lte": end_time_iso}}}
                ]
            }
        }
    }
    
    anchor_response = await es_client.search(index=INDEX_NAME, body=query_body)
    anchor_hits = anchor_response["hits"]["hits"]
    
    if not anchor_hits:
        console.print("[red]No logs found in this time window/environment![/red]")
        return
        
    console.print(f"   Found [bold]{len(anchor_hits)}[/bold] relevant error logs.")
    
    # ---------------------------------------------------------
    # STEP 2: EXPAND - Trace-based and Time-based expansion
    # ---------------------------------------------------------
    console.print("\n[cyan]2. Expanding:[/cyan] Extracting trace IDs and fetching the full distributed trace history...")
    
    trace_ids = set()
    for hit in anchor_hits:
        tid = hit["_source"].get("trace_id")
        if tid and tid != "":
            trace_ids.add(tid)
            
    expanded_logs = []
    
    if trace_ids:
        console.print(f"   Found {len(trace_ids)} unique trace IDs. Querying Elasticsearch for the full journey...")
        try:
            trace_query = {
                "size": 100,
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"environment": env}},
                            {"terms": {"trace_id": list(trace_ids)}}
                        ]
                    }
                }
            }
            trace_results = await es_client.search(index=INDEX_NAME, body=trace_query)
            expanded_logs = trace_results["hits"]["hits"]
        except Exception as e:
            console.print(f"[red]Error during trace expansion: {e}[/red]")
            expanded_logs = anchor_hits
    else:
        # If no trace IDs (e.g., OOMKilled K8s event), we just use the anchor logs
        console.print("   No Trace IDs found. Relying on time-window anchor logs.")
        expanded_logs = anchor_hits
        
    # Sort logs chronologically to help the LLM understand the sequence
    expanded_logs.sort(key=lambda x: x["_source"].get("timestamp", ""))
    
    # Build the Context Payload
    context_str = ""
    for hit in expanded_logs:
        log = hit["_source"]
        service_name = log.get('service', 'unknown')
        level = log.get('level', 'INFO')
        ts = log.get('timestamp', '')
        text = log.get('text', '')
        context_str += f"[{ts}] {service_name} [{level}]: {text}\n"

    console.print(f"   Compiled a highly-focused timeline of [bold]{len(expanded_logs)}[/bold] logs.")
    
    # ---------------------------------------------------------
    # STEP 3: REASON - Send to LLM for RCA
    # ---------------------------------------------------------
    console.print("\n[cyan]3. Reasoning:[/cyan] Sending the timeline to Gemini for Root Cause Analysis...")
    
    prompt = f"""
    You are an expert Site Reliability Engineer investigating a PagerDuty alert.
    
    ALERT DETAILS:
    Title: {alert.get('title')}
    Description: {alert_desc}
    Service Affected: {service}
    Environment: {env}
    Time: {alert.get('created_at')}
    
    Below is the exact chronological timeline of logs and Kubernetes events that led up to this alert. 
    We used Trace ID Expansion to gather the full distributed journey of the failing requests.
    
    LOG TIMELINE:
    {context_str}
    
    Please provide:
    1. A clear, concise summary of the Root Cause.
    2. The timeline of events (what happened first, how did it cascade).
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
        # Build UI Table
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

if __name__ == "__main__":
    asyncio.run(main())
