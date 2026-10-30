import logging
import sys

from mcp.server.fastmcp import FastMCP

# stdout carries the MCP stdio protocol; all diagnostics must go to stderr.
logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s %(levelname)s [rag_server] %(message)s")
log = logging.getLogger("rag_server")

from rag_store import search_knowledge  # noqa: E402

mcp = FastMCP("RAGKnowledge")


@mcp.tool()
def search_knowledge_base(query: str, top_k: int = 4) -> dict:
    """Search the internal company knowledge base: HR policies, annual/sick/casual leave,
    employee benefits, work-from-home rules, procedures and internal documentation.
    Use this FIRST for any question about company policies. Returns passages with their
    source filename and distance (lower is more relevant)."""
    try:
        return {"query": query, "results": search_knowledge(query, top_k)}
    except ValueError:
        raise
    except Exception as exc:
        # Log the full traceback; FastMCP returns the error to the client as a
        # tool error (isError) while this server keeps running.
        log.exception("search_knowledge_base failed")
        raise RuntimeError(f"Knowledge base search failed: {exc}") from exc


if __name__ == "__main__":
    log.info("Starting RAG MCP server")
    mcp.run(transport="stdio")
