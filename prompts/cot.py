"""Strategy C — Chain-of-Thought style structured reasoning prompt.

The model is guided through an explicit analysis procedure, but asked to carry
it out internally and return only the validated JSON plus a concise rationale.
We deliberately do NOT request or store step-by-step reasoning traces: the
output schema is identical to the other strategies.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from prompts.common import EXTRACTION_RULES, SYSTEM_ROLE

REASONING_PROCEDURE = """ANALYSIS PROCEDURE — work through these steps carefully before answering:
Step 1. Split the review into its separate statements.
Step 2. For each statement, decide whether it is about a product aspect. Ignore statements about the reviewer's personal situation that say nothing about the product.
Step 3. Map each aspect statement to exactly one aspect category using the definitions.
Step 4. Decide the aspect sentiment from the words in that statement only (watch for negation such as "not loud", and sarcasm).
Step 5. For every negative aspect, check whether it matches a complaint category definition. Some complaints (e.g. missing parts, not as described) may not map to an aspect; include them too.
Step 6. Weigh positive vs negative aspects to choose the overall sentiment: positive, negative, mixed (substantial both) or neutral (no opinion).
Step 7. Verify that every evidence string is copied exactly from the review and that no aspect is unsupported.

Perform these steps silently. Do NOT write out your step-by-step reasoning. Output only the final JSON object; you may put a one- or two-sentence justification in "rationale"."""

TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_ROLE + "\n\n" + EXTRACTION_RULES + "\n\n{taxonomy}\n\n" + REASONING_PROCEDURE + "\n\nOUTPUT FORMAT:\n{format_instructions}"),
        ("human", "Analyse this review following the procedure and return the JSON object.\n\nREVIEW:\n\"\"\"\n{review_text}\n\"\"\""),
    ]
)
