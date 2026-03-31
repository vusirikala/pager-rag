import os
import json
import asyncio
from typing import List, Dict, Any
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import track

load_dotenv()
console = Console()

es_client = AsyncElasticsearch("http://localhost:9200")

INDEX_NAME = "pager-rag-logs"

async def setup_elasticsearch():
    """Ensure the Elasticsearch index exists with the correct mapping for structured log data."""
    exists = await es_client.indices.exists(index=INDEX_NAME)
    if exists:
        console.print(f"[yellow]Deleting existing index '{INDEX_NAME}' for a fresh start...[/yellow]")
        await es_client.indices.delete(index=INDEX_NAME)

    mapping = {
        "properties": {
            "timestamp": {"type": "date"},
            "environment": {"type": "keyword"},
            "service": {"type": "keyword"},
            "level": {"type": "keyword"},
            "trace_id": {"type": "keyword"},
            "type": {"type": "keyword"},
            "message": {"type": "text"},
        }
    }
    await es_client.indices.create(index=INDEX_NAME, mappings=mapping)
    console.print(f"[green]Index '{INDEX_NAME}' created with explicit keyword mappings.[/green]")

async def process_and_index(data: List[Dict[Any, Any]], item_type: str):
    """Index logs/events into Elasticsearch as plain structured documents (no embeddings)."""
    if not data:
        return

    console.print(f"Processing {len(data)} {item_type}...")

    BATCH_SIZE = 500
    indexed = 0

    for i in track(range(0, len(data), BATCH_SIZE), description=f"Indexing {item_type}"):
        batch = data[i:i + BATCH_SIZE]
        actions = []

        for j, item in enumerate(batch):
            doc = {
                "_index": INDEX_NAME,
                "_id": f"{item_type}-{i + j}",
                "timestamp": item.get("timestamp"),
                "environment": item.get("environment", "unknown"),
                "service": item.get("service", item.get("name", "unknown")),
                "level": item.get("level", "INFO"),
                "trace_id": item.get("trace_id", ""),
                "type": item_type,
                "message": item.get("message", json.dumps(item)),
            }
            actions.append(doc)

        try:
            await async_bulk(es_client, actions)
            indexed += len(batch)
        except Exception as e:
            if hasattr(e, 'errors'):
                console.print(f"[red]BulkIndexError: {e.errors[0]}[/red]")
            else:
                console.print(f"[red]Error indexing to ES: {e}[/red]")

    console.print(f"[green]Successfully indexed {indexed} {item_type} documents.[/green]")

async def main():
    console.rule("[bold blue]Pager-RAG Elasticsearch Ingestion Script")

    await setup_elasticsearch()

    try:
        with open("data/logs.json", "r") as f:
            logs = json.load(f)
        with open("data/k8s_events.json", "r") as f:
            events = json.load(f)
    except FileNotFoundError:
        console.print("[red]Error: data/logs.json or k8s_events.json not found. Run generate_mock_data.py first.[/red]")
        await es_client.close()
        return

    console.print(f"[yellow]Loaded {len(logs)} logs and {len(events)} events for ingestion.[/yellow]")

    await process_and_index(logs, "app_log")
    await process_and_index(events, "k8s_event")

    await es_client.indices.refresh(index=INDEX_NAME)

    stats = await es_client.count(index=INDEX_NAME)
    console.print(f"\n[bold green]Ingestion Complete![/bold green]")
    console.print(f"Total documents in '{INDEX_NAME}': {stats['count']}")

    await es_client.close()

if __name__ == "__main__":
    asyncio.run(main())
