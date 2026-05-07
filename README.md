---
title: PaperTrail
emoji: 📚
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# PaperTrail: The Research Memory Agent

## Quick Setup (5 minutes)

### 1. Backend

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your Groq API key (or GEMINI_API_KEY as fallback)
export GROQ_API_KEY="gsk_your-key-here"
# Optional: pick a different Groq model
# export GROQ_MODEL="llama-3.1-8b-instant"

# Run the server
python main.py
```

Backend runs at `http://localhost:8000`

### 2. Frontend

The frontend is a React component (`PaperTrail.jsx`). You have two options:

**Option A: Quick — Use Vite**
```bash
npm create vite@latest frontend -- --template react
cd frontend
npm install

# Replace src/App.jsx with the PaperTrail.jsx content
# Then run:
npm run dev
```

**Option B: Even quicker — Single HTML file**
See `index.html` in the frontend folder (if provided), which loads React via CDN.

### 3. Demo Flow

1. Open the frontend in browser
2. Go to **Upload** tab → upload 2-3 research PDFs
3. Switch to **Knowledge Graph** → show the auto-generated entity graph
4. Go to **Ask** tab → ask a cross-paper question like:
   - "What methods are used across these papers?"
   - "Which papers evaluate on the same datasets?"
   - "Compare the approaches used in my papers"
5. Show the cited answer with source references

## Architecture

```
PDF Upload → Text Extraction (PyMuPDF)
           → Entity Extraction (GPT-4o)
           → Knowledge Graph (NetworkX)
           → Vector Embeddings (ChromaDB)

Query → Vector Search (ChromaDB)
      → Graph Traversal (NetworkX)
      → Answer Generation (GPT-4o)
      → Cited Response
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /upload | Upload and process a PDF |
| POST | /note | Add a text note |
| POST | /query | Ask a question (GraphRAG) |
| GET | /papers | List all papers |
| GET | /graph | Get knowledge graph (nodes + edges) |
| GET | /stats | System statistics |
| DELETE | /reset | Reset everything |

## Without an API Key

The system still works without an API key — it just skips entity extraction (no graph building) and uses only vector search for queries. For the demo, you really want the API key to show the full pipeline.

## Deployment (single container)

The included `Dockerfile` builds the frontend and serves it through the FastAPI
backend on a single port — same shape as a Streamlit Cloud deploy.

### Hugging Face Spaces (free, recommended)

1. Create a new Space → SDK: **Docker** (the README frontmatter already declares
   this so HF will pick it up automatically).
2. Push this repo to the Space.
3. In the Space's **Settings → Variables and secrets**, add `GROQ_API_KEY`.
4. Wait for the build. The app appears at `https://huggingface.co/spaces/<you>/<name>`.

### Render / Fly.io / Railway

Any "deploy from Dockerfile" provider works. Just set `GROQ_API_KEY` in the
service's environment. The container listens on `$PORT` (default 7860).

### Local Docker test

```bash
docker build -t papertrail .
docker run -p 7860:7860 -e GROQ_API_KEY=$GROQ_API_KEY papertrail
# open http://localhost:7860
```

> **Note**: ChromaDB and the knowledge graph are in-memory, so data is lost on
> restart. For persistence, mount a volume and switch ChromaDB to a persistent
> client (`chromadb.PersistentClient(path="/data/chroma")`).
