# Nexus — Agentic RAG Workspace

**An AI-powered workspace that intelligently routes user questions to a Knowledge Base, SQL Database, or Web Search using local LLMs and MCP tool integration.**

Nexus is an Agentic AI application built to demonstrate how Large Language Models can interact with multiple information sources, retrieve relevant context, execute database queries, and generate evidence-based responses through an interactive workspace.

**Developer:** Bhumi Tiwari

**Tech Stack:** Python · FastAPI · React · Vite · Ollama · Llama 3.2 · ChromaDB · SQLite · Model Context Protocol (MCP)

---

## Overview

Traditional chatbots primarily depend on an LLM to generate responses. Nexus takes a tool-oriented approach: it identifies the type of question and routes it to the appropriate information source.

The application supports:

- Knowledge retrieval using Retrieval-Augmented Generation (RAG).
- Natural-language questions converted into executable SQL queries.
- Web search integration through Tavily.
- Local LLM inference using Ollama.
- Tool communication through Model Context Protocol (MCP).
- Grounded responses based on retrieved documents and actual database results.

## Key Features

### 1. Agentic Question Routing

The backend classifies incoming questions into different routes:

| Route | Purpose |
|---|---|
| Knowledge | Search internal documents |
| Database | Retrieve information using SQL |
| Web | Search external information |
| General | Handle other questions using available tools |

### 2. Retrieval-Augmented Generation (RAG)

- Uses ChromaDB as the vector database.
- Retrieves relevant passages from the internal knowledge base.
- Uses retrieved context to generate answers.
- Includes source references in knowledge-based responses.
- Handles questions when relevant information is unavailable.

### 3. Natural Language to SQL

Users can ask questions in natural language instead of writing SQL manually.

Example:

**Question:** How many active employees are in Engineering?

**Process:**
1. Retrieve the database schema.
2. Generate an appropriate SQLite SELECT query.
3. Execute the query through the SQL MCP server.
4. Generate the final response using actual query results.

The database route requires successful query execution before returning a database answer, helping prevent fabricated database results.

### 4. MCP Tool Integration

The application uses separate MCP servers to expose tools:

| MCP Server | Tools |
|---|---|
| RAG Server | `search_knowledge_base` |
| SQL Server | `get_database_schema`, `query_database` |
| Web Search Server | `web_search` |

### 5. Local LLM Integration

- Ollama provides local model inference.
- Llama 3.2 is used for language understanding and response generation.
- Reduces dependency on hosted LLM inference for the configured local workflows.

### 6. Interactive Web Interface

A dark-themed workspace provides an interface for submitting questions and viewing AI-generated responses, retrieved information, and tool execution details.

---

## System Architecture

```text
                 USER QUESTION
                       |
                       v
                REACT + VITE UI
                       |
                       v
                 FASTAPI BACKEND
                       |
                       v
                  AGENT ROUTER
                       |
          +------------+------------+
          |            |            |
          v            v            v
      KNOWLEDGE     DATABASE       WEB
        ROUTE         ROUTE       ROUTE
          |            |            |
          v            v            v
       RAG MCP       SQL MCP      WEB MCP
          |            |            |
          v            v            v
       CHROMADB      SQLITE       TAVILY
          |            |            |
          +------------+------------+
                       |
                       v
                 OLLAMA LLM
                  LLAMA 3.2
                       |
                       v
                 FINAL RESPONSE
```

---

## Technology Stack

| Layer | Technologies |
|---|---|
| Programming Language | Python, JavaScript |
| Backend | FastAPI |
| Frontend | React, Vite |
| Language Model | Llama 3.2 |
| LLM Runtime | Ollama |
| Vector Database | ChromaDB |
| Relational Database | SQLite |
| Tool Orchestration | Model Context Protocol (MCP) |
| Web Search | Tavily API |
| Testing | Python unittest, HTTPX |

---

## Project Structure

```text
Agentic_RAG-main/
│
├── main.py
├── agent_routing.py
├── ingest.py
├── rag_store.py
│
├── mcp_servers/
│   ├── rag_server.py
│   ├── sql_server.py
│   └── web_search_server.py
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   ├── Orb.jsx
│   │   └── styles.css
│   ├── package.json
│   └── vite.config.js
│
├── data/
├── documents/
├── chroma_db/
│
├── test_routing.py
├── test_sql_regressions.py
├── test_rag_mcp.py
├── test_agent_e2e.py
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Getting Started

### Prerequisites

Install the following:

- Python 3.10 or later
- Node.js and npm
- Ollama
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/bhhumiii/nexus-agentic-rag.git

cd nexus-agentic-rag
```

### 2. Set Up Python Environment

```powershell
python -m venv .venv

.\.venv\Scripts\Activate.ps1
```

### 3. Install Backend Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure Ollama

Install Ollama and download Llama 3.2:

```bash
ollama pull llama3.2
```

Make sure the Ollama service is running.

### 5. Configure Environment Variables

Create a local `.env` file using `.env.example` as a reference.

Configure the required application authentication and database settings.

For external web search, configure:

```env
TAVILY_API_KEY=your_tavily_api_key
```

**Security:** Never commit your `.env` file, API keys, or other credentials to GitHub.

### 6. Start the Backend

From the project root:

```powershell
uvicorn main:app --host 127.0.0.1 --port 8000
```

### 7. Start the Frontend

Open another terminal:

```powershell
cd frontend

npm install

npm run dev
```

Open the local URL provided by Vite, usually:

```text
http://localhost:5173
```

---

## Example Queries

### Knowledge Base

```text
What is the annual leave policy of the company?
```

### Database

```text
How many active employees are in Engineering?
```

```text
List all employees in the Engineering department.
```

```text
Which employees have pending leave requests?
```

### Web Search

```text
Search the web for the latest Python release.
```

A valid Tavily API key is required for live web search.

---

## Testing and Validation

The development version has passed the following tests:

| Test Suite | Result |
|---|---|
| Unit and routing regression tests | 14 passed |
| RAG MCP tests | 4 passed |
| Agent end-to-end tests | 7 passed |
| Python syntax validation | 13 files passed |
| Python dependency check | No broken requirements |

### Run Tests

**Unit and SQL regression tests:**

```powershell
.\.venv\Scripts\python.exe -m unittest test_routing test_sql_regressions -v
```

**RAG MCP tests:**

```powershell
.\.venv\Scripts\python.exe test_rag_mcp.py
```

**End-to-end tests:**

Start the backend and Ollama first, then run:

```powershell
.\.venv\Scripts\python.exe test_agent_e2e.py
```

---

## Engineering Challenges Solved

Some of the important engineering problems addressed during development:

- **Tool Routing:** Implemented backend-side question classification to reduce incorrect tool selection by smaller local language models.
- **SQL Reliability:** Separated SQL generation from execution to improve reliability with Llama 3.2.
- **Grounded Database Answers:** Added execution checks to prevent database responses without actual query results.
- **SQL Validation:** Added read-only query restrictions and regression tests.
- **RAG Reliability:** Added retrieval relevance checks and handling for queries with no relevant internal information.
- **MCP Error Handling:** Improved tool error reporting and tested server behaviour after invalid tool requests.

---

## Future Improvements

- Add conversational memory and multi-turn context.
- Improve retrieval relevance and answer evaluation.
- Add document upload and ingestion management.
- Improve real-time response streaming.
- Expand observability and error monitoring.
- Deploy the application for public access.

---

## About the Developer

**Bhumi Tiwari**

M.Tech — Artificial Intelligence and Machine Learning

Interested in Software Engineering, Backend Development, Artificial Intelligence, and Agentic AI systems.

- **GitHub:** [bhhumiii](https://github.com/bhhumiii)
- **LinkedIn:** [Bhumi Tiwari](https://www.linkedin.com/in/bhumi-tiwari-4707a0282/)

---

**If you find this project interesting, consider giving the repository a star!**

Built with Python, local LLMs, and a focus on reliable AI tool orchestration.
