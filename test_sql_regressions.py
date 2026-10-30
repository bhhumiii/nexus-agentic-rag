"""Regression tests for the read-only SQL MCP server, through a real stdio session.

Run: .\\.venv\\Scripts\\python.exe -m unittest test_sql_regressions -v
Expected values are computed from data/company.db at test time (read-only).
"""
import asyncio
import json
import os
import sqlite3
import sys
import unittest
from datetime import timedelta
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

BASE = Path(__file__).resolve().parent
DB = Path(os.getenv("SQLITE_PATH", str(BASE / "data" / "company.db")))
if not DB.is_absolute():
    DB = BASE / DB


def expected(sql: str, params=()):
    with sqlite3.connect(DB.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        return conn.execute(sql, params).fetchall()


def with_session(test):
    """Run an async test inside one task that owns the MCP stdio session."""
    def wrapper(self):
        async def run():
            params = StdioServerParameters(command=sys.executable, cwd=str(BASE),
                                           args=["-m", "mcp_servers.sql_server"],
                                           env=os.environ.copy())
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write,
                                         read_timeout_seconds=timedelta(seconds=60)) as session:
                    await session.initialize()
                    self.session = session
                    await test(self)
        asyncio.run(run())
    wrapper.__name__ = test.__name__
    return wrapper


class SqlServerTests(unittest.TestCase):
    async def call(self, name, args=None, expect_error=False):
        result = await self.session.call_tool(name, args or {})
        text = "\n".join(c.text for c in result.content if getattr(c, "text", None))
        self.assertEqual(result.isError, expect_error, text)
        return text if expect_error else json.loads(text)

    async def query(self, sql, params=None):
        args = {"sql": sql} if params is None else {"sql": sql, "params": params}
        return await self.call("query_database", args)

    # --- schema ---------------------------------------------------------------
    @with_session
    async def test_tools_registered(self):
        names = {t.name for t in (await self.session.list_tools()).tools}
        self.assertEqual(names, {"get_database_schema", "query_database"})

    @with_session
    async def test_schema_lists_tables_and_real_values(self):
        schema = await self.call("get_database_schema")
        self.assertEqual(set(schema["tables"]), {"employees", "leave_requests"})
        departments = [r[0] for r in expected("SELECT DISTINCT department FROM employees ORDER BY 1")]
        self.assertEqual(schema["column_values"]["employees.department"], departments)
        self.assertIn("Active", schema["column_values"]["employees.status"])

    # --- employee counts --------------------------------------------------------
    @with_session
    async def test_active_engineering_count_literal(self):
        want = expected("SELECT COUNT(*) FROM employees WHERE department='Engineering' AND status='Active'")[0][0]
        out = await self.query("SELECT COUNT(*) AS n FROM employees WHERE department = 'engineering' "
                               "COLLATE NOCASE AND status = 'active' COLLATE NOCASE;")
        self.assertEqual(out["rows"][0]["n"], want)

    @with_session
    async def test_active_engineering_count_with_params(self):
        want = expected("SELECT COUNT(*) FROM employees WHERE department='Engineering' AND status='Active'")[0][0]
        out = await self.query("SELECT COUNT(*) AS n FROM employees WHERE department = ? "
                               "COLLATE NOCASE AND status = ? COLLATE NOCASE", ["Engineering", "Active"])
        self.assertEqual(out["rows"][0]["n"], want)

    @with_session
    async def test_placeholder_without_params_gives_clear_error(self):
        text = await self.call("query_database",
                               {"sql": "SELECT COUNT(*) FROM employees WHERE department = ?"},
                               expect_error=True)
        self.assertIn("bindings", text)
        self.assertIn("quoted literals", text)
        self.assertNotIn("approved tables", text)  # no misleading table hint

    # --- employee lists and departments -------------------------------------------
    @with_session
    async def test_employee_list(self):
        want = expected("SELECT employee_id, name FROM employees ORDER BY employee_id")
        out = await self.query("SELECT * FROM employees ORDER BY employee_id")
        self.assertEqual([(r["employee_id"], r["name"]) for r in out["rows"]], want)
        self.assertEqual(out["columns"], ["employee_id", "name", "department", "title", "location", "status"])

    @with_session
    async def test_department_counts(self):
        want = expected("SELECT department, COUNT(*) FROM employees GROUP BY department ORDER BY department")
        out = await self.query("SELECT department, COUNT(*) AS n FROM employees GROUP BY department ORDER BY department")
        self.assertEqual([(r["department"], r["n"]) for r in out["rows"]], want)

    # --- leave requests -----------------------------------------------------------
    @with_session
    async def test_pending_leave_requests_join(self):
        want = expected("SELECT e.name, l.leave_type FROM leave_requests l JOIN employees e "
                        "ON e.employee_id = l.employee_id WHERE l.status = 'Pending' ORDER BY l.request_id")
        out = await self.query("SELECT e.name, l.leave_type FROM leave_requests AS l JOIN employees AS e "
                               "ON e.employee_id = l.employee_id WHERE l.status = 'pending' COLLATE NOCASE "
                               "ORDER BY l.request_id")
        self.assertEqual([(r["name"], r["leave_type"]) for r in out["rows"]], want)

    @with_session
    async def test_cte_and_quoted_identifier(self):
        want = expected("SELECT COUNT(*) FROM leave_requests")[0][0]
        out = await self.query('WITH x AS (SELECT * FROM "leave_requests") SELECT COUNT(*) AS n FROM x')
        self.assertEqual(out["rows"][0]["n"], want)

    # --- read-only restrictions -----------------------------------------------------
    @with_session
    async def test_writes_and_unsafe_sql_rejected(self):
        bad = ["INSERT INTO employees VALUES (9,'x','y','z','w','Active')",
               "DELETE FROM employees",
               "DROP TABLE employees",
               "SELECT * FROM sqlite_master",
               "SELECT 1 FROM employees; DROP TABLE employees",
               "SELECT * FROM employees -- comment",
               "PRAGMA table_info(employees)"]
        for sql in bad:
            with self.subTest(sql=sql):
                await self.call("query_database", {"sql": sql}, expect_error=True)
        count = expected("SELECT COUNT(*) FROM employees")[0][0]
        self.assertGreater(count, 0)  # data untouched

    @with_session
    async def test_unapproved_table_rejected(self):
        text = await self.call("query_database", {"sql": "SELECT * FROM secrets"}, expect_error=True)
        self.assertIn("rejected", text)

    # --- formats small models actually produced (seen in backend logs) ---------------

    @with_session
    async def test_params_as_strings(self):
        want = expected("SELECT COUNT(*) FROM employees")[0][0]
        for params in ["{}", "[]", ""]:
            with self.subTest(params=params):
                out = await self.call("query_database",
                                      {"sql": "SELECT COUNT(*) AS n FROM employees", "params": params})
                self.assertEqual(out["rows"][0]["n"], want)
        want = expected("SELECT COUNT(*) FROM employees WHERE status='Active'")[0][0]
        out = await self.call("query_database", {"sql": "SELECT COUNT(*) AS n FROM employees "
                                                 "WHERE status = :status", "params": "{'status': 'Active'}"})
        self.assertEqual(out["rows"][0]["n"], want)
        await self.call("query_database", {"sql": "SELECT 1 FROM employees",
                                           "params": "__import__('os')"}, expect_error=True)

    def test_extract_sql(self):
        from main import extract_sql
        self.assertEqual(extract_sql("```sql\nSELECT * FROM employees;\n```"), "SELECT * FROM employees;")
        self.assertEqual(extract_sql("Here:\n```\nWITH x AS (SELECT 1) SELECT * FROM x\n```"),
                         "WITH x AS (SELECT 1) SELECT * FROM x")
        self.assertEqual(extract_sql("SELECT name FROM employees"), "SELECT name FROM employees")
        self.assertIsNone(extract_sql("```sql\nDELETE FROM employees\n```"))
        self.assertIsNone(extract_sql("There are 2 employees."))


if __name__ == "__main__":
    unittest.main(verbosity=2)
