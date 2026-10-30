# Multi-tool Agentic RAG with MCP

Three capabilities, each backed by an MCP server:
1. ChromaDB retrieval
2. Natural-language-to-SQL over read-only SQLite
3. Web search via Tavily

NL-to-SQL uses two internal operations (schema discovery and query execution).

## Windows prerequisite

Install the latest **Microsoft Visual C++ Redistributable 2015-2022 (x64)**, version
14.40 or newer (`winget install Microsoft.VCRedist.2015+.x64`). With an older runtime,
ChromaDB and onnxruntime crash with an access violation. The RAG server then exits, and
the backend reports `McpError: Connection closed`. The RAG tool checks the runtime
version and returns a clear error instead of crashing.

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env
ollama pull llama3.2
python setup_demo.py
python ingest.py
uvicorn main:app --host 127.0.0.1 --port 8000
```
Set `AGENT_API_KEY` in `.env`. Set `TAVILY_API_KEY` to enable web search. The web tool returns a configuration message if no key is set. Swagger: http://127.0.0.1:8000/docs

Set `OLLAMA_MODEL=qwen3:8b` in `.env` to use the installed tool-calling model. Model calls, MCP requests, and external HTTP calls allow up to 300 seconds (five minutes) per operation. An agent response can take longer because it may require several model and tool calls.

Example request:
```json
{"question":"How many active employees are in Engineering?","top_k":4}
```
Send `X-API-Key` with the configured key.

## React chat frontend

The `frontend/` directory contains Nexus, a responsive React interface with Tailwind
CSS and a Three.js animated orb. Start the backend using the setup instructions
above, then run these commands in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite (normally `http://127.0.0.1:5173`). Click
**Connect API key** and enter the same `AGENT_API_KEY` configured in the backend
`.env`. The key and conversations are held only in browser memory and clear on
reload. The frontend never embeds your key in its source or build.

Vite proxies `/api/health` and `/api/agent/query` to `http://127.0.0.1:8000`, so no
CORS changes are needed. To use a different backend, set `AGENT_BACKEND_URL` in
the terminal before starting Vite. For example:

```powershell
$env:AGENT_BACKEND_URL = 'http://127.0.0.1:8001'
npm run dev
```

The chat shows real responses, Markdown, returned tool usage, and request timing.
Prompt cards fill the composer; Enter sends and Shift+Enter adds a line. Connection
settings also control retrieval `top_k`. The backend handles each question
independently; the conversation display does not add history to API requests.
Stopping a request stops waiting in the browser; it does not guarantee cancellation
of work already running on the backend. Service health is checked every 30 seconds
and does not verify individual external tools or the API key.

```powershell
npm run build
npm run preview
```

The production bundle is written to `frontend/dist/`. Vite preview includes the
same local API proxy. For production hosting, serve that directory and configure
a same-origin reverse proxy from `/api/*` to the FastAPI service (removing `/api`).
Use HTTPS for remote access. Three.js respects reduced-motion settings, pauses
animation when the tab is hidden, and has a CSS fallback when WebGL is unavailable.

## How questions are routed

`agent_routing.py` classifies each question before the LLM runs, because small local
models (such as llama3.2 3B) often pick the wrong tool:

| Route | Example | What happens |
|---|---|---|
| Knowledge | "What is the annual leave policy?" | The backend calls `search_knowledge_base` first. Passages with cosine distance above `RAG_MAX_DISTANCE` (default 0.70) are dropped. If none remain, the answer says the information is not in the knowledge base. Otherwise the LLM answers only from the passages and the source file is cited. Web search is never used. |
| Database | "How many active employees are in Engineering?" | The backend calls `get_database_schema`, the LLM writes one SQLite SELECT as text, the backend runs it with `query_database` (up to 3 corrected attempts), and the LLM answers only from the returned rows. |
| Web | "Search the web for ..." | Uses `web_search`. Without `TAVILY_API_KEY` the API returns HTTP 503 with a configuration message; RAG and SQL are unaffected. |
| General | mixed or unclear questions | The LLM chooses tools itself (`web_search` is hidden when Tavily is not configured). |

## Tests

```powershell
# Fast tests (no Ollama needed)
.\.venv\Scripts\python.exe -m unittest test_routing test_sql_regressions -v
.\.venv\Scripts\python.exe test_rag_mcp.py

# End-to-end agent tests (backend and Ollama running)
.\.venv\Scripts\python.exe test_agent_e2e.py                      # http://127.0.0.1:8000
```

`test_sql_regressions.py` runs the SQL MCP server over stdio and computes expected values
from `data/company.db` (employee counts, lists, departments, leave requests, read-only
rejections). `test_rag_mcp.py` reads `CHROMA_PATH`; point it at a copy of `chroma_db` if you
do not want to open the main database. `test_agent_e2e.py` checks routing, grounded RAG
answers with sources, SQL answers against the database, and web-search behavior.

## Important limitations
This is a learning baseline, not a certified production deployment. The database is synthetic. For real data, use a dedicated read-only database identity, restricted views/row-level permissions, auditing, and robust SQL validation. Use TLS, a real identity provider, rate limits, observability, secret management, and security testing before deployment.

The sample HR policy is generic and not an official policy of any employer. Web search results and retrieved documents are untrusted data. This project uses MCP Python SDK 1.x APIs (`FastMCP`, `ClientSession`, `stdio_client`).
