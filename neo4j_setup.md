# Neo4j setup for this QA-RAG workflow

## Option A: Neo4j Desktop (easy)

1. Install Neo4j Desktop.
2. Create a local DBMS (Neo4j 5.x recommended).
3. Set username/password (default user is `neo4j`).
4. Start DBMS.
5. Use these env vars before starting APIs:
   - `NEO4J_URI=bolt://localhost:7687`
   - `NEO4J_USER=neo4j`
   - `NEO4J_PASSWORD=<your-password>`

## Option B: Docker

```bash
docker run --name neo4j-qa-rag \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  -e NEO4J_server_memory_heap_initial__size=1G \
  -e NEO4J_server_memory_heap_max__size=2G \
  -d neo4j:5
```

Then use:
- `NEO4J_URI=bolt://localhost:7687`
- `NEO4J_USER=neo4j`
- `NEO4J_PASSWORD=password`

## Verify connection

Open Browser at http://localhost:7474 and run:

```cypher
RETURN 1;
```

## What gets stored

- `(:Project {name})`
- `(:SRS {id, source_path, text})`
- `(:Chunk {id, text, source:'srs'})`
- `(:TestCase {id, title, last_verdict, last_run_at})`
- `(:TestRun {id, verdict, notes, created_at})`

Relationships:
- `(Project)-[:HAS_SRS]->(SRS)-[:HAS_CHUNK]->(Chunk)`
- `(Project)-[:HAS_TEST]->(TestCase)-[:HAS_RUN]->(TestRun)`

## Minimal startup sequence

1. Start Neo4j.
2. Start `local_rag_api.py`.
3. Ingest `SRS1.txt` via gateway `/srs/ingest`.
4. Start gateway and run test loop.
