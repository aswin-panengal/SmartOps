# SmartOps 

SmartOps is a full-stack AI operations assistant that combines tabular CSV analytics with PDF semantic search in a single chat-driven workflow.

## What it actually does

- **Unified ask endpoint**: `/api/ask` accepts a question and optional CSV/PDF upload, then routes to the correct engine automatically.
- **CSV analytics engine**: parses uploaded CSVs into Pandas, caches the DataFrame per `session_id`, builds a lightweight data blueprint, and uses Google Gemini-generated Python to answer questions safely.
- **PDF RAG engine**: extracts PDF text with `pypdf`, chunks it, embeds chunks with Google Gemini embeddings, stores vectors in Qdrant, and performs semantic search for question answering.
- **Session memory**: keeps conversation history and summary state in memory, and uses it to improve routing and follow-up queries.
- **Multi-session UI**: frontend supports multiple active chat workspaces, drag-and-drop file upload, session reset, and backend health status.
- **Safe fallback handling**: detects API rate-limit failures and returns user-friendly retry messages.

## Architecture

### Backend
- `FastAPI` app with endpoints for CSV analysis, PDF ingestion/querying, and session clearing.
- `langgraph` agent graph routes between:
  - `clarify` for follow-up/confirmations,
  - `csv` for tabular analytics,
  - `pdf` for document QA.
- CSV engine uses:
  - `pandas` + `numpy`
  - in-memory session caching of parsed DataFrames
  - Gemini-generated Python code execution in a restricted sandbox
  - up to 3 self-correction retries on execution failures
- PDF engine uses:
  - `pypdf` text extraction
  - overlapping text chunks + embeddings
  - `Qdrant` vector database for semantic retrieval
  - Google Gemini semantic answer generation
- Session memory:
  - stores chat history and summaries to keep prompts manageable
  - summarizes older messages when sessions grow long

### Frontend
- `Next.js 14` + `React 18` + `TypeScript`
- `Tailwind CSS` styling with custom inline style fallbacks
- `react-markdown` for rendering assistant content
- multi-workspace session management with active session state
- drag/drop and file picker for `.csv` and `.pdf`
- status indicator for backend availability

## Tech stack

### Backend
- Python 3.11
- FastAPI
- Uvicorn
- pandas
- numpy
- pypdf
- Qdrant client
- Google Gemini via `langchain-google-genai`
- `langgraph` for agent routing
- `python-multipart`, `python-dotenv`, `pydantic-settings`
- Docker + Docker Compose for backend & Qdrant

### Frontend
- Next.js 14
- React 18
- TypeScript
- Tailwind CSS
- react-markdown
- lucide-react

## How to run locally

### Backend
1. Start Docker Desktop.
2. From `backend/`:
```bash
docker compose up --build
```
3. Alternatively, with the virtual environment active:
```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend
From `frontend/`:
```bash
npm install
npm run dev
```

## Project structure

- `backend/app/main.py`: FastAPI app and health checks
- `backend/app/api/routes.py`: API routes for ask, CSV, PDF, session
- `backend/app/agent/graph.py`: router + CSV/PDF agent nodes
- `backend/app/engines/analytical.py`: CSV analysis engine
- `backend/app/engines/semantic.py`: PDF embedding + query engine
- `backend/app/memory/session.py`: in-memory session history and summaries
- `frontend/src/app/page.tsx`: chat UI, session panels, file upload, and message flow

## Key features

- Natural-language CSV analytics
- PDF semantic search + QA
- In-memory, session-aware chat history
- Multi-workspace chat sessions
- Vector search with Qdrant
- Google Gemini for embeddings and response generation
- Safe execution sandbox for model-generated Python code
- API rate-limit friendly error handling
