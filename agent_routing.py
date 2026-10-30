"""Decide which tools a question should use before the LLM sees it.

Small local models (such as llama3.2 3B) often pick the wrong tool when every
tool is offered, so the backend routes clear-cut questions itself.
"""
import re

KNOWLEDGE, DATABASE, WEB, GENERAL = "knowledge", "database", "web", "general"

# The user explicitly asks for the public internet.
_WEB = re.compile(
    r"\b(web|internet|online|google|websites?|urls?|links?|news|latest|wikipedia|"
    r"official (site|website|documentation|docs))\b", re.I)

# Structured records in the demo SQLite database.
_DATABASE = re.compile(
    r"\b(database|sql|sqlite|tables?|records?|rows?|leave[- ]requests?|employee ids?|"
    r"departments?|(how many|list|show|which|count)( all| the| of the)?( active| inactive)? employees)\b",
    re.I)

# Internal company documents: policies, HR, leave, benefits, procedures.
_KNOWLEDGE = re.compile(
    r"\b(polic(y|ies)|hr|human resources|leaves?|vacations?|holidays?|benefits?|entitle\w*|"
    r"handbook|procedures?|guidelines?|internal|work from home|remote work|wfh|sick|"
    r"carr(y|ied) forward|notice period|reimburse\w*|code of conduct|onboarding|"
    r"probation|insurance|allowances?|company rules?)\b", re.I)


def classify_question(question: str) -> str:
    """Return KNOWLEDGE, DATABASE, WEB or GENERAL (let the model choose)."""
    if _WEB.search(question):
        return WEB  # An explicit request for the web wins.
    database = _DATABASE.search(question)
    # "leave requests" is a database table, not a policy topic.
    knowledge = _KNOWLEDGE.search(re.sub(r"(?i)\bleave[- ]requests?\b", " ", question))
    if database and knowledge:
        return GENERAL  # Mixed questions may need both tools.
    if database:
        return DATABASE
    if knowledge:
        return KNOWLEDGE
    return GENERAL
