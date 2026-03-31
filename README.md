# Pager-RAG: AI SRE Assistant

A proof-of-concept for a RAG-powered incident response tool designed to help Site Reliability Engineers (SREs) debug PagerDuty alerts.

This tool implements an **"Anchor and Expand"** retrieval strategy using Elasticsearch and Google Gemini:
1. **Anchor (Semantic Search)**: Uses Gemini embeddings to find the exact error logs in Elasticsearch that semantically match a PagerDuty alert description within a 5-minute time window.
2. **Expand (Trace Retrieval)**: Extracts the `trace_id` from the anchor logs and performs a native metadata query to pull the entire distributed journey of the failing request across all microservices.
3. **Reason (LLM)**: Feeds the highly-focused, chronological trace context to Gemini to generate a Root Cause Analysis and suggested mitigations.

## Prerequisites

- Python 3.9+
- Docker & Docker Compose (for running local Elasticsearch)
- A Google Gemini API Key

## Setup

1. **Clone the repository and set up a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory and add your Google Gemini API key:
   ```env
   GEMINI_API_KEY="your_api_key_here"
   ```

## Running the Pipeline

### 1. Start Elasticsearch
We use a local instance of Elasticsearch 8.x to store logs, metadata, and vector embeddings.
```bash
docker-compose up -d
```
Wait a few seconds for Elasticsearch to start up. You can verify it's running by hitting `http://localhost:9200` in your browser.

### 2. Generate Mock Data (Optional)
*Note: A massive dataset of logs, Kubernetes events, and PagerDuty alerts may already exist in the `data/` directory. You only need to run this if you want to generate fresh scenarios.*

This script uses Gemini to generate ~50,000 realistic background noise logs, interspersed with specific error scenarios (OOMKills, DB timeouts, API Latency, etc.).
```bash
python3 generate_mock_data.py
```

### 3. Ingest Data into Elasticsearch
This script reads the mock data, uses Gemini (`text-embedding-004`) to generate vector embeddings for the logs, and indexes them into Elasticsearch.
```bash
python3 ingest.py
```

### 4. Run the RAG CLI
Start the interactive terminal application. You will be presented with a list of active PagerDuty alerts. Type the ID of an alert to watch the Anchor -> Expand -> Reason pipeline execute in real time.
```bash
python3 rag_cli.py
```

## Architecture

* **Storage Engine**: Elasticsearch (Local Docker)
* **Embedding Model**: Google `text-embedding-004` (768 dimensions)
* **LLM**: Google `gemini-2.5-flash`
* **UI**: Rich (Python CLI)