"""
Deterministic filename proposals.

The proposal is built from the classification values in schema order. It is a
suggestion only: the client remains responsible for the final storage name.
"""

import re
from typing import Dict, List, Optional

from .file_types import MIME_TO_EXTENSION

MAX_STEM_LENGTH = 180


def slugify_value(value: str) -> str:
    """
    'Annual Reports' -> 'annual_reports', '2023/2024' -> '2023_2024'.
    Values written entirely in capitals and digits (codes such as 'ABC 101')
    keep their case: 'ABC_101'.
    """
    text = value.strip()
    is_code = bool(re.search(r"[A-Z]", text)) and not re.search(r"[a-z]", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return text if is_code else text.lower()


def propose_filename_stem(classification: Dict[str, Optional[str]], schema: List[str]) -> Optional[str]:
    parts = [slugify_value(classification[f]) for f in schema if classification.get(f)]
    parts = [p for p in parts if p]
    if not parts:
        return None
    stem = "-".join(parts)
    return stem[:MAX_STEM_LENGTH].rstrip("-_")


def propose_filename(
    classification: Dict[str, Optional[str]],
    schema: List[str],
    source_mime_types: List[str],
) -> Optional[str]:
    """
    Extension: the source type when every source file has the same type;
    '.pdf' when a group combines several files of different types.
    """
    stem = propose_filename_stem(classification, schema)
    if not stem:
        return None
    unique = list(dict.fromkeys(source_mime_types))
    extension = MIME_TO_EXTENSION.get(unique[0], "") if len(unique) == 1 else ".pdf"
    return stem + extension
