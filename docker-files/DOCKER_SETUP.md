# Docker Setup for DOC_QA_AG_RAG

This guide explains how to run the Agentic RAG document analysis system using Docker.

## Quick Start

### Option 1: Using Docker Compose (Recommended)

Run both Ollama and the Flask app together:

```bash
docker-compose up -d
```

This will:
- Start Ollama service on port 11434
- Start Flask app on port 5000
- Create persistent volumes for uploads and vector database
- Set up automatic health checks

Access the app at: **http://localhost:5000**

### Option 2: Docker Build Only

Build the image:

```bash
docker build -t doc-qa-ag-rag:latest .
```

Run the container (requires Ollama running separately):

```bash
docker run -d \
  --name doc-qa-ag-rag \
  -p 5000:5000 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  doc-qa-ag-rag:latest
```

## Prerequisites

- Docker >= 20.10
- Docker Compose >= 1.29 (for Option 1)
- 8GB RAM minimum (for Ollama + Flask app)
- 10GB disk space (for Ollama models)

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_HOST` | `http://ollama:11434` | Ollama service endpoint |
| `FLASK_ENV` | `production` | Flask environment |
| `PORT` | `5000` | Flask server port |

## Volumes

### `uploads/`
- **Purpose**: Store uploaded documents
- **Persistence**: Yes
- **Size**: Depends on uploaded files

### `chroma_db/`
- **Purpose**: Vector database with embeddings
- **Persistence**: Yes
- **Size**: Grows with documents indexed

### `ollama-data/` (docker-compose only)
- **Purpose**: Ollama models and cache
- **Persistence**: Yes
- **Size**: ~5-10GB per model

## Common Commands

```bash
# View logs
docker-compose logs -f doc-qa-app

# Check status
docker-compose ps

# Stop services
docker-compose down

# Stop and remove volumes (⚠️ deletes data)
docker-compose down -v

# Rebuild image
docker-compose build --no-cache

# Access shell inside container
docker exec -it doc-qa-ag-rag /bin/bash

# Check Flask app status
curl http://localhost:5000/api/status
```

## Performance Tuning

### For Limited Memory
```yaml
# In docker-compose.yml, add under doc-qa-app service:
deploy:
  resources:
    limits:
      memory: 4G
    reservations:
      memory: 2G
```

### For GPU Support (CUDA)
```yaml
# Replace ollama service image with:
image: ollama/ollama:latest-gpu

# Add runtime configuration:
runtime: nvidia
```

## Troubleshooting

### App can't connect to Ollama
```bash
# Check if Ollama is healthy
docker-compose ps

# Verify Ollama is running
docker logs ollama-service
```

### Vector database is corrupted
```bash
# Remove and recreate
docker-compose down
rm -rf chroma_db/
docker-compose up
```

### Out of memory errors
```bash
# Increase Docker memory limit in Docker Desktop settings
# or use resource limits in docker-compose.yml
```

### Port already in use
```bash
# Change port mapping in docker-compose.yml
ports:
  - "5001:5000"  # Use 5001 instead of 5000
```

## File Structure in Docker

```
/app/
├── web_app_up.py              # Flask server
├── agentic_rag_doc_analysis.py # RAG logic
├── rag_evaluator.py           # RAGAS evaluation tool
├── static/
│   ├── script_up.js
│   └── style_up.css
├── templates/
│   └── index_up.html
├── uploads/                   # Uploaded documents (volume)
└── chroma_db/                 # Vector database (volume)
```

## Running Evaluation Inside Container

You can run RAGAS evaluation inside the Docker container:

```bash
# Access container shell
docker exec -it doc-qa-ag-rag /bin/bash

# Run evaluation
python rag_evaluator.py
```

Or run it directly without entering the shell:

```bash
docker exec doc-qa-ag-rag python rag_evaluator.py
```

## Image Size Optimization

The multi-stage build keeps the image lean:
- **Final image size**: ~3-4GB (includes Python runtime + dependencies)
- **Builder stage**: Discarded after build (not included in final image)
- **Excluded files**: Documentation, evaluation tools, IDE configs, git history

## Production Deployment

### Using Kubernetes
```bash
kubectl apply -f k8s-deployment.yaml
```

### Using Docker Swarm
```bash
docker stack deploy -c docker-compose.yml rag-app
```

### Health Check
The container includes a health check that verifies:
- Flask app is responsive
- `/api/status` endpoint is accessible
- Checks every 30 seconds with 3 retries

## Security Notes

⚠️ **For Production**:
- Change `FLASK_ENV` from `production` to use proper wsgi server (gunicorn)
- Add authentication to API endpoints
- Use secrets for sensitive environment variables
- Enable HTTPS/SSL
- Run as non-root user
- Set resource limits

## Cleanup

Remove all Docker artifacts:

```bash
# Stop and remove containers
docker-compose down

# Remove image
docker rmi doc-qa-ag-rag:latest

# Remove volumes (⚠️ deletes all data)
docker volume rm doc_qa_ag_rag_ollama-data

# Prune unused images
docker image prune -a
```

## References

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Ollama Docker Hub](https://hub.docker.com/r/ollama/ollama)
