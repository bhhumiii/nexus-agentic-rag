import ast, json, os, re, sqlite3
from pathlib import Path
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("ReadOnlySQL")
BASE = Path(__file__).resolve().parents[1]
DB = Path(os.getenv("SQLITE_PATH", str(BASE / "data" / "company.db")))
if not DB.is_absolute():
    DB = BASE / DB
ALLOWED = {"employees", "leave_requests"}

def connect_ro():
    if not DB.exists():
        raise FileNotFoundError("Demo DB missing; run python setup_demo.py")
    conn = sqlite3.connect(DB.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    denied = {
        sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_CREATE_INDEX, sqlite3.SQLITE_CREATE_TABLE,
        sqlite3.SQLITE_CREATE_TRIGGER, sqlite3.SQLITE_CREATE_VIEW,
        sqlite3.SQLITE_DROP_INDEX, sqlite3.SQLITE_DROP_TABLE,
        sqlite3.SQLITE_DROP_TRIGGER, sqlite3.SQLITE_DROP_VIEW,
        sqlite3.SQLITE_ALTER_TABLE, sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH,
        sqlite3.SQLITE_PRAGMA,
    }
    def authorize(action, table, column, database, trigger):
        if action in denied:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_READ and table.lower() not in ALLOWED:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    conn.set_authorizer(authorize)
    return conn

# Low-cardinality text columns whose real values help the model write correct filters.
VALUE_COLUMNS = {"employees": ["department", "title", "location", "status"],
                 "leave_requests": ["leave_type", "status"]}

@mcp.tool()
def get_database_schema() -> dict:
    """Return approved database schema to help plan natural-language SQL."""
    conn = connect_ro()
    try:
        values = {f"{table}.{col}": [r[0] for r in conn.execute(
                      f"SELECT DISTINCT {col} FROM {table} ORDER BY 1 LIMIT 25")]
                  for table, cols in VALUE_COLUMNS.items() for col in cols}
    finally:
        conn.close()
    return {"tables": {
        "employees": ["employee_id INTEGER", "name TEXT", "department TEXT",
                      "title TEXT", "location TEXT", "status TEXT"],
        "leave_requests": ["request_id INTEGER", "employee_id INTEGER",
                           "leave_type TEXT", "start_date TEXT", "end_date TEXT",
                           "status TEXT"]},
        "relationships": ["leave_requests.employee_id references employees.employee_id"],
        "column_values": values,
        "dates": "start_date and end_date are TEXT in YYYY-MM-DD format",
        "sql_rules": ["Write filter values as quoted literals, e.g. "
                      "WHERE department = 'Engineering' COLLATE NOCASE",
                      "Only filter on columns the question mentions"]}

def _normalize_params(params):
    """Accept params as a list, a dict (named :placeholders) or their JSON/Python text."""
    if isinstance(params, str):
        text = params.strip()
        if not text:
            return []
        try:
            params = json.loads(text)
        except json.JSONDecodeError:
            try:
                params = ast.literal_eval(text)  # literals only; never executes code
            except (ValueError, SyntaxError) as exc:
                raise ValueError("params must be a JSON list of values") from exc
    if params is None:
        return []
    if isinstance(params, (list, tuple)):
        values = list(params)
    elif isinstance(params, dict):
        values = params
    else:
        raise ValueError("params must be a list of values")
    items = values.values() if isinstance(values, dict) else values
    if len(items) > 20 or not all(v is None or isinstance(v, (str, int, float)) for v in items):
        raise ValueError("params must be at most 20 plain values (text or numbers)")
    return values

@mcp.tool()
def query_database(sql: str, params: list | dict | str | None = None) -> dict:
    """Execute one SQLite SELECT/CTE on employees or leave_requests; max 100 rows.

    Use get_database_schema first. Prefer quoted literal values in the SQL. If the SQL
    uses ? placeholders, pass their values in order in params. A final semicolon is
    accepted. Match text filters case-insensitively with COLLATE NOCASE. Never write
    or query unapproved tables.
    """
    params = _normalize_params(params)
    sql = sql.strip()
    if not sql or len(sql) > 4000:
        raise ValueError("SQL is empty or too long")
    # A single statement terminator is normal LLM output, not a second statement.
    sql = sql.removesuffix(";").rstrip()
    if any(x in sql for x in (";", "--", "/*", "*/")):
        raise ValueError("Semicolons between statements and SQL comments are forbidden")
    if not re.match(r"(?is)^(SELECT|WITH)\b", sql):
        raise ValueError("Only SELECT/CTE queries are allowed")
    if re.search(r"(?i)\b(sqlite_master|sqlite_schema|pragma|attach|detach|insert|update|delete|drop|alter|create|replace)\b", sql):
        raise ValueError("Disallowed SQL keyword")
    if not re.search(r"(?i)\bFROM\b", sql):
        raise ValueError("Use only approved tables: employees, leave_requests")
    conn = connect_ro()
    try:
        # SQLite resolves quoted identifiers, aliases and CTEs. Its authorizer,
        # rather than a FROM/JOIN regex, enforces the underlying table allowlist.
        try:
            cur = conn.execute(sql, params)
        except sqlite3.ProgrammingError as exc:  # e.g. placeholder/params count mismatch
            raise ValueError(f"SQLite query rejected: {exc}. Write values as quoted "
                             "literals instead of ? placeholders.") from exc
        except sqlite3.DatabaseError as exc:
            hint = (" Use only approved tables employees, leave_requests."
                    if "not authorized" in str(exc).lower() else
                    " Use only the tables and columns from get_database_schema.")
            raise ValueError(f"SQLite query rejected: {exc}.{hint}") from exc
        cols = [c[0] for c in cur.description or []]
        rows = cur.fetchmany(101)
        return {"columns": cols, "rows": [dict(r) for r in rows[:100]],
                "row_count": min(len(rows), 100), "truncated": len(rows) > 100}
    finally:
        conn.close()

if __name__ == "__main__":
    mcp.run(transport="stdio")
