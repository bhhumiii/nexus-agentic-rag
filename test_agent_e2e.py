"""End-to-end agent tests through the running FastAPI backend (needs Ollama).

Run (backend already started):
    .\\.venv\\Scripts\\python.exe test_agent_e2e.py            # http://127.0.0.1:8000
    .\\.venv\\Scripts\\python.exe test_agent_e2e.py http://127.0.0.1:8001
Reads AGENT_API_KEY and TAVILY_API_KEY from .env; never prints the key.
SQL expectations are computed from data/company.db (read-only), not hardcoded.
Exit code 0 = all passed, 1 = a check failed.
"""
import os
import sqlite3
import sys
import time
from pathlib import Path

import httpx
from dotenv import dotenv_values

BASE = Path(__file__).resolve().parent
ENV = dotenv_values(BASE / ".env")
URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
KEY = (ENV.get("AGENT_API_KEY") or "").strip()
TAVILY = bool((ENV.get("TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY") or "").strip())
DB = BASE / (ENV.get("SQLITE_PATH") or "data/company.db")


def db(sql):
    with sqlite3.connect(DB.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        return conn.execute(sql).fetchall()


def ask(question):
    start = time.perf_counter()
    r = httpx.post(f"{URL}/agent/query", json={"question": question, "top_k": 4},
                   headers={"X-API-Key": KEY}, timeout=900)
    return r.status_code, r.json(), time.perf_counter() - start


def contains_all(answer, items):
    missing = [str(i) for i in items if str(i).lower() not in answer.lower()]
    return not missing, f"missing {missing}" if missing else "all expected values present"


def sql_case(question, checker):
    def run():
        status, body, secs = ask(question)
        if status != 200:
            return False, f"HTTP {status}: {body.get('detail')}", secs
        tools = body["tools_used"]
        if "query_database" not in tools or "web_search" in tools:
            return False, f"wrong tools {tools}", secs
        ok, detail = checker(body["answer"])
        return ok, f"{detail}; tools={tools}; answer={body['answer'][:160]!r}", secs
    return question, run


def rag_case():
    question = "What is the annual leave policy of the company?"
    def run():
        status, body, secs = ask(question)
        if status != 200:
            return False, f"HTTP {status}: {body.get('detail')}", secs
        ok = (body["tools_used"] == ["search_knowledge_base"] and "18" in body["answer"]
              and "company_hr_policies.txt" in body["answer"])
        return ok, f"tools={body['tools_used']}; answer={body['answer'][:200]!r}", secs
    return question, run


def rag_not_found_case():
    question = "What is the company parking policy for electric vehicles?"
    def run():
        status, body, secs = ask(question)
        ok = status == 200 and "could not find" in body.get("answer", "").lower() \
            and "web_search" not in body.get("tools_used", [])
        return ok, f"HTTP {status}; {body}", secs
    return question, run


def web_case():
    question = "Search the web for the latest Python release."
    def run():
        status, body, secs = ask(question)
        if not TAVILY:
            ok = status == 503 and "TAVILY_API_KEY" in body.get("detail", "")
            return ok, f"(Tavily not configured) HTTP {status}: {body.get('detail')}", secs
        ok = status == 200 and body["tools_used"] == ["web_search"]
        return ok, f"(Tavily configured) HTTP {status}; {str(body)[:200]}", secs
    return question, run


def main():
    if not KEY:
        print("AGENT_API_KEY missing from .env")
        return 1
    active_eng = db("SELECT COUNT(*) FROM employees WHERE department='Engineering' AND status='Active'")[0][0]
    eng_names = [r[0] for r in db("SELECT name FROM employees WHERE department='Engineering'")]
    dept_counts = db("SELECT department, COUNT(*) FROM employees GROUP BY department")
    pending = [r[0] for r in db("SELECT e.name FROM leave_requests l JOIN employees e "
                                "ON e.employee_id=l.employee_id WHERE l.status='Pending'")]
    cases = [
        rag_case(),
        rag_not_found_case(),
        sql_case("How many active employees are in Engineering? Use the demo database.",
                 lambda a: contains_all(a, [active_eng])),
        sql_case("List all employees in the Engineering department.",
                 lambda a: contains_all(a, eng_names)),
        sql_case("How many employees are in each department?",
                 lambda a: contains_all(a, [x for row in dept_counts for x in row])),
        sql_case("Which employees have pending leave requests?",
                 lambda a: contains_all(a, pending)),
        web_case(),
    ]
    failures = 0
    for question, run in cases:
        ok, detail, secs = run()
        failures += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] ({secs:5.1f}s) {question}\n        {detail}")
    print("ALL AGENT E2E TESTS PASSED" if not failures else f"{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
