"""Test search_knowledge_base through a real MCP stdio session.

Uses CHROMA_PATH from the environment (default: ./chroma_db). To avoid touching
the project database, point it at a copy:
    $env:CHROMA_PATH = "C:\\path\\to\\copy_of_chroma_db"
    .\\.venv\\Scripts\\python.exe test_rag_mcp.py
Exit code 0 = all passed, 1 = a check failed.
"""
import asyncio
import json
import os
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

BASE = Path(__file__).resolve().parent


async def run_checks(env: dict, checks: list) -> list[str]:
    params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_servers.rag_server"],
                                   cwd=str(BASE), env=env)
    failures = []
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=300)) as s:
            await s.initialize()
            tools = [t.name for t in (await s.list_tools()).tools]
            print("tools:", tools)
            if "search_knowledge_base" not in tools:
                return ["search_knowledge_base not registered"]
            for label, args, check in checks:
                result = await s.call_tool("search_knowledge_base", args)
                text = "\n".join(c.text for c in result.content if getattr(c, "text", None))
                ok, detail = check(result.isError, text)
                print(f"[{'PASS' if ok else 'FAIL'}] {label}: {detail}")
                if not ok:
                    failures.append(label)
    return failures


def results_ok(is_error, text):
    if is_error:
        return False, text[:300]
    rows = json.loads(text)["results"]
    good = rows and all({"text", "source", "distance"} <= r.keys() for r in rows)
    return bool(good), [(r["source"], round(r["distance"], 3)) for r in rows]


async def main() -> int:
    env = os.environ.copy()
    print("CHROMA_PATH:", env.get("CHROMA_PATH", str(BASE / "chroma_db")))
    failures = await run_checks(env, [
        ("annual leave search", {"query": "How many days of annual leave?", "top_k": 3}, results_ok),
        ("empty query is a tool error (server stays up)", {"query": "   "},
         lambda e, t: (e and "empty" in t.lower(), t[:120])),
        ("search still works after an error", {"query": "sick leave", "top_k": 1}, results_ok),
    ])
    with tempfile.TemporaryDirectory() as empty_db:
        failures += await run_checks({**env, "CHROMA_PATH": empty_db}, [
            ("empty collection returns no results", {"query": "anything"},
             lambda e, t: (not e and json.loads(t)["results"] == [], t[:120])),
        ])
    print("ALL MCP RAG TESTS PASSED" if not failures else f"FAILED: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
