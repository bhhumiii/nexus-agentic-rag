"""Unit tests for question routing. Run: .\\.venv\\Scripts\\python.exe -m unittest test_routing -v"""
import unittest

from agent_routing import DATABASE, GENERAL, KNOWLEDGE, WEB, classify_question

CASES = [
    ("What is the annual leave policy of the company?", KNOWLEDGE),
    ("How many days of annual leave do employees receive in the sample HR policy?", KNOWLEDGE),
    ("How many sick leave days do I get?", KNOWLEDGE),
    ("Can unused leave be carried forward?", KNOWLEDGE),
    ("What are the employee benefits?", KNOWLEDGE),
    ("Who is eligible to work from home?", KNOWLEDGE),
    ("What does the internal documentation say about onboarding?", KNOWLEDGE),
    ("How many active employees are in Engineering? Use the demo database.", DATABASE),
    ("List all employees", DATABASE),
    ("Which employees have pending leave requests?", DATABASE),
    ("Show the leave_requests table", DATABASE),
    ("Search the web for the official Python documentation and give me the link.", WEB),
    ("What is the latest news about OpenAI?", WEB),
    ("Search online for the company leave policy", WEB),
    ("Compare the HR leave policy with the leave requests in the database", GENERAL),
    ("What is 2 + 2?", GENERAL),
]


class RoutingTests(unittest.TestCase):
    def test_routes(self):
        for question, expected in CASES:
            with self.subTest(question=question):
                self.assertEqual(classify_question(question), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
