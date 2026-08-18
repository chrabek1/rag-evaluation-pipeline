# RAG Evaluation Pipeline

> Reproducible local environment for building and evaluating retrieval strategies over a fixed document corpus.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Docker Compose](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)
![pgvector](https://img.shields.io/badge/pgvector-vector%20search-orange)

## Overview

RAG Evaluation Pipeline is a local Retrieval-Augmented Generation (RAG) project for retrieval experiments and evaluation. It loads a corpus of prepared document chunks from CSV, generates dense embeddings with a configured Sentence Transformers model (by default [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3)), and stores them in PostgreSQL with `pgvector` for vector similarity search.

## Evaluation inputs

The runtime pipeline consumes two prepared, versioned artifacts:

- `dane.csv` — the fixed chunk corpus indexed by the retrieval system;
- `golden_dataset.json` — validated questions and their relevant chunk IDs.

Golden-dataset preparation is a separate, offline workflow. It is not executed
when the services start, when the corpus is indexed, or when retrieval is run.
Its scripts, intermediate Open RAGBench data, annotation instructions, and
reproduction steps are documented in
[`dataset_preparation/README.md`](dataset_preparation/README.md).

This boundary keeps the evaluation pipeline independent from the source and
annotation method used to create a golden dataset. The pipeline only needs a
dataset that conforms to the expected input contract.

## Architecture

```mermaid
flowchart LR
  subgraph INDEXING_FLOW [Indexing flow]
    A[CSV corpus] --> B[CsvChunkLoader]
    B --> C[IndexingService]
    C --> D[EmbeddingClient]
    D --> E[embedding_service]
    E --> F[Configured embedding model]
    C --> G[ChunkRepository]
    G --> H[(PostgreSQL + pgvector)]
  end

  subgraph RETRIEVAL_FLOW [Retrieval flow]
    Q[query] --> R[RetrievalService]
    R --> D
    D --> E
    R --> G
    G --> H
    H --> T[Top-k RetrievedChunk]
  end
```

The environment consists of three Docker services:

- `backend` — corpus loading, indexing, retrieval and persistence
- `embedding_service` — FastAPI service that generates dense embeddings
- `postgres` — PostgreSQL 16 with `pgvector`

## Current Status

Implemented:

- fixed-corpus CSV loading with stable chunk identifiers
- document chunk model and embedding payload validation
- batched embedding generation and database writes
- idempotent indexing with PostgreSQL UPSERT
- repository-level persistence logic for chunk records
- Docker Compose environment with service health checks
- FastAPI embedding service with model and schema validation
- automated unit and integration tests for backend and embedding service (44 passing backend tests)
- reproducible full-corpus indexing from an empty database
- vector retrieval with cosine similarity
- configurable `top_k` for retrieval
- `RetrievedChunk` model including a `score` field
- CLI retrieval entrypoint: `scripts/search_chunks.py`
- logging for indexing and retrieval
- dedicated, automatically-created `rag_eval_test` database for tests
- Open RAGBench subset preparation for manual annotation
- manually curated evidence annotations and validated golden dataset

Planned:

- retrieval evaluation metrics
- end-to-end evaluation pipeline / benchmark automation

## Quick Start

### Requirements

- Docker
- Docker Compose

No local Python installation is required.

### 1. Configure the environment

Create a `.env` file in the project root (see `.env.example` below):

```dotenv
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_SERVICE_URL=http://embedding_service:8001
DATABASE_URL=postgresql://rag_eval:rag_eval@postgres:5432/rag_eval
TEST_DATABASE_URL=postgresql://rag_eval:rag_eval@postgres:5432/rag_eval_test
CORPUS_PATH=/data/dane.csv
GOLDEN_DATASET_PATH=/data/golden_dataset.json

# Optional host paths used by Docker Compose
CORPUS_HOST_PATH=./dane.csv
GOLDEN_DATASET_HOST_PATH=./golden_dataset.json
```

- The backend loads configuration from the file specified as `env_file` in `docker-compose.yml`.
- For local development copy `.env.example` → `.env` and adjust values as needed.
- Do not commit `.env` to the repository.

### Replacing the input data or embedding model

The runtime configuration is independent from the included example dataset.
To use another corpus or golden dataset, set their host paths in `.env`:

```dotenv
CORPUS_HOST_PATH=./path/to/chunks.csv
GOLDEN_DATASET_HOST_PATH=./path/to/golden_dataset.json
```

Docker Compose mounts these files at the container paths configured by
`CORPUS_PATH` and `GOLDEN_DATASET_PATH`. The corpus must be a UTF-8 CSV file
with `filename` and `content` columns. The golden dataset contract is described
in [`dataset_preparation/README.md`](dataset_preparation/README.md#golden-dataset-contract).

To use another Sentence Transformers embedding model, change:

```dotenv
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Rebuild the embedding service after changing the model:

```bash
docker compose up -d --build embedding_service
```

The backend obtains the model name and vector dimension dynamically from the
embedding service. If the new model has a different vector dimension, the
existing pgvector column is incompatible and indexing stops with an explicit
error. Recreate the PostgreSQL data volume and index the complete corpus again:

```bash
# Warning: this removes the local PostgreSQL data stored by this Compose project.
docker compose down -v
docker compose up -d --build
docker compose exec backend uv run python scripts/index_chunks.py
```

### 2. Start the services

```bash
docker compose up -d --build
docker compose ps
```

The first build may take longer because the embedding model needs to be downloaded.

### 3. Index the corpus

```bash
docker compose exec backend uv run python scripts/index_chunks.py
```

For the included dataset, a successful run prints:

```text
Indexed 735 chunks.
```

Indexing is idempotent, so running the command again updates existing rows instead of creating duplicates.

### 4. Verify the result

```bash
docker compose exec postgres \
  psql -U rag_eval -d rag_eval \
  -c "SELECT COUNT(*) FROM chunks;"
```

The expected row count for the included dataset is `735`.

### Search the corpus

Run an example retrieval request against the indexed corpus:

```bash
docker compose exec backend \
  uv run python scripts/search_chunks.py \
  "What is heteroskedasticity?" \
  --top-k 5
```

This returns the top-k retrieved chunks for the query using the configured similarity search.

## Database Schema

| Column | Description |
| --- | --- |
| `id` | Internal database identifier |
| `chunk_id` | Stable unique chunk identifier |
| `filename` | Source document filename |
| `content` | Original chunk text |
| `embedding` | Vector embedding whose dimension is obtained dynamically from the embedding service |

Before initializing the `chunks` table, the backend requests the configured
model name and embedding dimension from the embedding service `/info` endpoint.
The retrieved dimension is used when declaring the pgvector column.

If the table already exists with a different vector dimension, initialization
fails with an explicit error. Changing the embedding model therefore requires
recreating the stored embeddings.

Note: retrieval is implemented using cosine similarity over stored vector embeddings. The `score` value on `RetrievedChunk` is the similarity/score returned by the search and is not an evaluation metric.
## Testing

The project includes both unit and integration tests for the backend and embedding service.

Run the backend test suite:

```bash
docker compose exec backend uv run pytest tests -v
```

Run the embedding service test suite:

```bash
docker compose exec embedding_service pytest
```

The backend test suite covers:

- unit tests
- repository integration tests
- indexing pipeline integration tests
- retrieval pipeline integration tests

Current test status: 44 passing backend tests.

## Project Structure

```text
rag-evaluation-pipeline/
├── backend/
│   ├── app/
│   │   ├── clients/          # external service clients
│   │   ├── core/             # application configuration
│   │   ├── db/               # database connection and schema
│   │   ├── loaders/          # corpus loading
│   │   ├── models/           # domain models
│   │   │   ├── chunk.py
│   │   │   └── retrieved_chunk.py
│   │   ├── repositories/     # persistence layer
│   │   └── services/         # application services
│   │       └── retrieval_service.py
│   
│   ├── scripts/              # executable backend scripts
│   │   ├── index_chunks.py
│   │   └── search_chunks.py
│   ├── tests/
│   │   ├── integration/
│   │   │   ├── test_chunk_repository.py
│   │   │   ├── test_indexing_pipeline.py
│   │   │   └── test_retrieval_pipeline.py
│   │   └── unit/
│   │       ├── test_chunk.py
│   │       └── test_retrieval_service.py
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── uv.lock
├── embedding_service/
│   ├── app/                  # embedding API and model service
│   ├── tests/
│   │   ├── integration/      # API contract tests
│   │   └── unit/             # service logic tests
│   ├── Dockerfile
│   └── requirements.txt
├── dataset_preparation/      # offline golden-dataset preparation workflow
│   ├── open_rag_data/        # manifests, annotations and ignored source data
│   ├── build_golden_dataset.py
│   ├── prepare_annotation_dataset.py
│   └── README.md
├── docs/                     # research and evaluation notes
├── docker/
│   └── postgres/
│       └── init.sql
├── docker-compose.yml
├── dane.csv                  # fixed corpus consumed by the pipeline
├── golden_dataset.json       # validated evaluation input
├── .env.example
├── README.md
```

## Reproducibility

To verify the complete pipeline from a clean database, remove the containers and PostgreSQL volume before repeating the Quick Start steps:

```bash
docker compose down -v
docker compose up -d --build
docker compose exec backend uv run python scripts/index_chunks.py
```

A successful clean run should produce the same `735` indexed chunks.

The `rag_eval_test` database is created automatically using `docker/postgres/init.sql` when the PostgreSQL service initializes.

Quick verification of retrieval after indexing (example):

```bash
docker compose exec backend \
  uv run python scripts/search_chunks.py \
  "What is heteroskedasticity?" \
  --top-k 5
```

This runs a retrieval request against the indexed corpus. Retrieval uses cosine similarity over stored vectors; the `score` field on returned `RetrievedChunk` items reflects the similarity value used for ranking and is not an external evaluation metric.
