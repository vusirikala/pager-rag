import os
import json
import asyncio
from datetime import datetime, timedelta
from dateutil import parser
from pinecone.grpc import PineconeGRPC as Pinecone
from google import genai
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.markdown import Markdown

load_dotenv()
console = Console()

# Initialize API Clients
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index("pager-rag-logs")
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-004"
LLM_MODEL = "gemini-2.5-flash"

async def get_embedding(text: str) -> list[float]:
    """Get vector embedding from Google Gemini."""
    response = await gemini_client.aio.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text
    )
    return response.embeddings[0].values

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
    anchor_results = index.query(
        namespace=env,
        vector=query_vector,
        top_k=5,
        include_metadata=True,
        filter={
            "timestamp": {"$gte": start_time_iso, "$lte": end_time_iso},
            # "level": {"$in": ["ERROR", "WARN"]} # Optionally filter to just errors
        }
    )
    
    if not anchor_results.matches:
        console.print("[red]No logs found in this time window/environment![/red]")
        return
        
    console.print(f"   Found [bold]{len(anchor_results.matches)}[/bold] relevant error logs.")
    
    # ---------------------------------------------------------
    # STEP 2: EXPAND - Trace-based and Time-based expansion
    # ---------------------------------------------------------
    console.print("\n[cyan]2. Expanding:[/cyan] Extracting trace IDs and fetching the full distributed trace history...")
    
    trace_ids = set()
    for match in anchor_results.matches:
        tid = match.metadata.get("trace_id")
        if tid and tid != "":
            trace_ids.add(tid)
            
    expanded_logs = []
    
    if trace_ids:
        console.print(f"   Found {len(trace_ids)} unique trace IDs. Querying Pinecone for the full journey...")
        
        # We don't even need vector search here. We just use metadata filtering.
        # But Pinecone requires a vector or id for query, so we use a dummy vector 
        # (or just use the same query vector but rely entirely on the filter)
        # Note: the namespace MUST match the anchor log's namespace
        # We query for ALL logs that share this trace ID, regardless of what service emitted them!
        try:
            trace_results = index.query(
                namespace=env,
                vector=[0.0] * 768, 
                top_k=100, # Get the whole trace journey across all microservices
                include_metadata=True,
                filter={
                    "trace_id": {"$in": list(trace_ids)}
                }
            )
            expanded_logs = trace_results.matches
        except Exception as e:
            console.print(f"[red]Error during trace expansion: {e}[/red]")
            expanded_logs = anchor_results.matches
    else:
        # If no trace IDs (e.g., OOMKilled K8s event), we just use the anchor logs
        # or we could do a secondary query for ALL logs around that specific second
        console.print("   No Trace IDs found. Relying on time-window anchor logs.")
        expanded_logs = anchor_results.matches
        
    # Sort logs chronologically to help the LLM understand the sequence
    expanded_logs.sort(key=lambda x: x.metadata.get("timestamp", ""))
    
    # Build the Context Payload
    context_str = ""
    for log in expanded_logs:
        service_name = log.metadata.get('service', 'unknown')
        level = log.metadata.get('level', 'INFO')
        ts = log.metadata.get('timestamp', '')
        text = log.metadata.get('text', '')
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
