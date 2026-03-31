import os
import json
import asyncio
from typing import List, Dict, Any
from google import genai
from google.genai import types
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import track

load_dotenv()
console = Console()

# Initialize API Clients
gemini_api_key = os.environ.get("GEMINI_API_KEY")

if not gemini_api_key:
    console.print("[red]Error: Missing GEMINI_API_KEY in .env file.[/red]")
    exit(1)

es_client = AsyncElasticsearch("http://localhost:9200")
gemini_client = genai.Client(api_key=gemini_api_key)

INDEX_NAME = "pager-rag-logs"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 768

async def setup_elasticsearch():
    """Ensure the Elasticsearch index exists with the correct mapping for vectors and text."""
    exists = await es_client.indices.exists(index=INDEX_NAME)
    if not exists:
        console.print(f"[yellow]Creating Elasticsearch index '{INDEX_NAME}'...[/yellow]")
        mapping = {
            "mappings": {
                "properties": {
                    "timestamp": {"type": "date"},
                    "environment": {"type": "keyword"},
                    "service": {"type": "keyword"},
                    "level": {"type": "keyword"},
                    "trace_id": {"type": "keyword"},
                    "type": {"type": "keyword"},
                    "text": {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": EMBEDDING_DIMENSION,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }
        }
        await es_client.indices.create(index=INDEX_NAME, mappings=mapping["mappings"])
        console.print(f"[green]Index '{INDEX_NAME}' created successfully.[/green]")
    else:
        console.print(f"[green]Index '{INDEX_NAME}' already exists.[/green]")

async def get_embedding(text: str) -> List[float]:
    """Get vector embedding from Google Gemini."""
    try:
        response = await gemini_client.aio.models.embed_content(
            model='gemini-embedding-001',
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSION)
        )
        if not response.embeddings:
            return [0.0] * EMBEDDING_DIMENSION
        return response.embeddings[0].values
    except Exception as e:
        console.print(f"[red]Failed to get embedding: {e}[/red]")
        return [0.0] * EMBEDDING_DIMENSION

async def process_and_upsert(data: List[Dict[Any, Any]], item_type: str):
    """
    Process logs or events, generate embeddings, and index them to Elasticsearch.
    """
    if not data:
        return
        
    console.print(f"Processing {len(data)} {item_type}...")
    
    BATCH_SIZE = 50 
    
    for i in track(range(0, len(data), BATCH_SIZE), description=f"Indexing {item_type}"):
        batch = data[i:i + BATCH_SIZE]
        actions = []
        
        # Concurrently fetch embeddings for the batch
        texts_to_embed = [json.dumps({k: v for k, v in item.items() if k not in ['_scenario_id']}) for item in batch]
        
        # Use simple for loop instead of gather to avoid hitting rate limits or weird async gemini bugs
        embeddings = []
        for text in texts_to_embed:
            emb = await get_embedding(text)
            embeddings.append(emb)
                
        for j, (item, embedding, text) in enumerate(zip(batch, embeddings, texts_to_embed)):
            doc_id = f"{item_type}-{i + j}"
            
            doc = {
                "_index": INDEX_NAME,
                "_id": doc_id,
                "timestamp": item.get("timestamp"),
                "environment": item.get("environment", "unknown"),
                "service": item.get("service", item.get("name", "unknown")),
                "level": item.get("level", "INFO"),
                "trace_id": item.get("trace_id", ""),
                "type": item_type,
                "text": text,
                "embedding": embedding
            }
            actions.append(doc)
            
        try:
            await async_bulk(es_client, actions)
        except Exception as e:
            if hasattr(e, 'errors'):
                console.print(f"[red]BulkIndexError: {e.errors[0]}[/red]")
            else:
                console.print(f"[red]Error indexing to ES: {e}[/red]")

async def main():
    console.rule("[bold blue]Pager-RAG Elasticsearch Ingestion Script")
    
    await setup_elasticsearch()
    
    # 1. Load Data
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
    
    # 2. Process and Upsert
    await process_and_upsert(logs, "app_log")
    await process_and_upsert(events, "k8s_event")
    
    # Force refresh
    await es_client.indices.refresh(index=INDEX_NAME)
    
    # Print Stats
    stats = await es_client.count(index=INDEX_NAME)
    console.print("\n[bold green]Ingestion Complete![/bold green]")
    console.print(f"Total documents in Elasticsearch index '{INDEX_NAME}': {stats['count']}")
    
    await es_client.close()

if __name__ == "__main__":
    asyncio.run(main())