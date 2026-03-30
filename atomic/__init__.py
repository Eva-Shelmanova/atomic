"""
atomic — AI-агентная система с роутингом и цепочкой специализированных агентов.

Архитектура:
    User → RouterAgent → Chain of Responsibility (Retriever | Command | Analyst)
    → Memory Layer → Final Answer
"""

__version__ = "0.1.0"
