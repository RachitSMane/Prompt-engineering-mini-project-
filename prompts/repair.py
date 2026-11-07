"""Repair prompt: sent when an LLM output fails JSON parsing or schema validation."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

REPAIR_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You repair malformed or invalid JSON produced by another model. "
            "Return ONE corrected JSON object that satisfies the schema. "
            "Keep the original content and judgements unless they cause an error. "
            "Do not add new information that is not in the source material. "
            "No markdown fences, no explanations.",
        ),
        (
            "human",
            "TASK THE ORIGINAL OUTPUT WAS FOR:\n{task_description}\n\n"
            "REQUIRED SCHEMA:\n{format_instructions}\n\n"
            "{source_block}"
            "ORIGINAL OUTPUT:\n<<<\n{original_output}\n>>>\n\n"
            "VALIDATION ERRORS:\n{errors}\n\n"
            "Return the corrected JSON object only.",
        ),
    ]
)
