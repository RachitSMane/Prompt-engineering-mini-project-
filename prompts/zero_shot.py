"""Strategy A — Zero-shot: instructions + definitions + review. No examples."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from prompts.common import EXTRACTION_RULES, SYSTEM_ROLE

TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_ROLE + "\n\n" + EXTRACTION_RULES + "\n\n{taxonomy}\n\nOUTPUT FORMAT:\n{format_instructions}"),
        ("human", "Analyse this review and return the JSON object.\n\nREVIEW:\n\"\"\"\n{review_text}\n\"\"\""),
    ]
)
