# src/services/template_utils.py
from __future__ import annotations
import re
from typing import Dict, Tuple

ISO_TS_FULL = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
ISO_TS_DATE = r"\d{4}-\d{2}-\d{2}"

def template_question(question: str) -> Tuple[str, Dict[str, str]]:
    """
    Produce a signature with <DATETIME_RANGE_i>/<STRING_i>/<DATE_i>/<NUMBER_i> placeholders + extracted params.
    Important: NUMBER matches must not replace digits inside existing <PLACEHOLDERS>.
    """
    extracted_params: Dict[str, str] = {}
    templated_q = question

    # 1) Date-time range: between 'A' and/to 'B'
    pattern_range = r"between\s+'([^']+)'\s+(?:to|and)\s+'([^']+)'"
    matches = re.findall(pattern_range, templated_q, flags=re.IGNORECASE)
    for i, match in enumerate(matches):
        placeholder = f"<DATETIME_RANGE_{i}>"
        original_text_and = f"between '{match[0]}' and '{match[1]}'"
        original_text_to  = f"between '{match[0]}' to '{match[1]}'"
        templated_q = re.sub(re.escape(original_text_and), placeholder, templated_q, count=1, flags=re.IGNORECASE)
        templated_q = re.sub(re.escape(original_text_to),  placeholder, templated_q, count=1, flags=re.IGNORECASE)
        extracted_params[placeholder] = original_text_and

    # 2) Quoted strings
    str_matches = re.findall(r"'([^']+)'", templated_q)
    for i, match in enumerate(str_matches):
        placeholder = f"<STRING_{i}>"
        templated_q = templated_q.replace(f"'{match}'", placeholder, 1)
        extracted_params[placeholder] = match

    # 3) ISO dates
    date_matches = re.findall(rf'\b({ISO_TS_FULL}|{ISO_TS_DATE})\b', templated_q)
    for i, match in enumerate(date_matches):
        placeholder = f"<DATE_{i}>"
        templated_q = re.sub(r'\b' + re.escape(match) + r'\b', placeholder, templated_q, count=1)
        extracted_params[placeholder] = match

    # 4) Numbers (avoid replacing digits inside <...>)
    placeholder_spans = [(m.start(), m.end()) for m in re.finditer(r"<[^>]+>", templated_q)]
    def in_placeholder(idx: int) -> bool:
        for a, b in placeholder_spans:
            if a <= idx < b:
                return True
        return False

    for m in list(re.finditer(r'\b(\d+)\b', templated_q)):
        if in_placeholder(m.start()):
            continue
        num = m.group(1)
        placeholder = f"<NUMBER_{len([k for k in extracted_params if k.startswith('<NUMBER_')])}>"
        templated_q = templated_q[:m.start()] + placeholder + templated_q[m.end():]
        extracted_params[placeholder] = num
        # update spans after replacement
        placeholder_spans = [(mm.start(), mm.end()) for mm in re.finditer(r"<[^>]+>", templated_q)]

    return templated_q, extracted_params


def template_sql(concrete_sql: str, params: Dict[str, str]) -> str:
    """
    Turn a concrete SQL into a reusable template by swapping back extracted values to placeholders.
    """
    templated_sql = concrete_sql
    # Replace longer values first to avoid partial overlaps
    for placeholder, value in sorted(params.items(), key=lambda x: len(x[1]), reverse=True):
        if placeholder.startswith('<NUMBER'):
            templated_sql = re.sub(r'\b' + re.escape(value) + r'\b', placeholder, templated_sql)
        else:
            templated_sql = templated_sql.replace(f"'{value}'", placeholder)
    return templated_sql.split(";")[0].strip() + ";"


def fill_sql_template(templated_sql: str, params: Dict[str, str]) -> str:
    """
    Fill a templated SQL (with <STRING_i>/<NUMBER_i>/...) with current question’s values.
    """
    filled_sql = templated_sql
    for placeholder, value in params.items():
        if placeholder.startswith('<NUMBER'):
            filled_sql = filled_sql.replace(placeholder, value)
        else:
            filled_sql = filled_sql.replace(placeholder, f"'{value}'")
    return filled_sql
