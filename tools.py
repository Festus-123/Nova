from langchain_core.tools import tool
from ddgs import DDGS
from pathlib import Path
import wikipedia

from memory import (
    get_current_user_id,
    save_memory,
    get_memories,
    search_memories,
    delete_memory,
)


# ============================================================
# RESEARCH TOOLS
# ============================================================

@tool
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia for information about a topic.
    """
    try:
        return wikipedia.summary(query, sentences=5)
    except wikipedia.DisambiguationError as e:
        return f"Multiple matches found: {e.options[:5]}"
    except Exception as e:
        return str(e)


@tool
def web_search(query: str) -> str:
    """
    Search the web for recent information.
    """
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))

    if not results:
        return "No results found."

    output = []
    for result in results:
        output.append(
            f"""
Title: {result['title']}
URL: {result['href']}
Body: {result['body']}
"""
        )

    return "\n".join(output)


@tool
def save_report(filename: str, content: str) -> str:
    """
    Save research results to a text file.
    """
    Path(filename).write_text(content, encoding="utf-8")
    return f"Saved report to {filename}"


@tool
def read_report(filename: str) -> str:
    """
    Read a saved report.
    """
    return Path(filename).read_text(encoding="utf-8")


# ============================================================
# LONG-TERM MEMORY TOOLS
# ============================================================

@tool
def save_user_memory(category: str, key: str, value: str, importance: int = 8) -> str:
    """
    Save or update a piece of information to the user's persistent long-term memory.
    Use this whenever the user shares personal details (like their name, preferences,
    tech stack, ongoing projects, instructions) or explicitly asks you to save/remember
    something across conversations.

    Args:
        category: 'identity' (name, title, contact), 'preference' (coding languages, style),
                  'project' (current apps, repos), 'skill', 'instruction', or 'context'.
        key: A short identifier (e.g. 'name', 'favorite_language', 'coding_style', 'active_project').
        value: The exact fact or instruction to remember.
        importance: 1 to 10 scale (default 8). For user name or explicit instructions, use 9 or 10.
    """
    user_id = get_current_user_id()
    saved = save_memory(
        user_id=user_id,
        category=category,
        key=key,
        value=value,
        importance=importance,
    )
    return (
        f"Saved to persistent long-term memory for user '{user_id}': "
        f"[{saved['category']}] {saved['key']}: {saved['value']} (importance: {saved['importance']}/10)"
    )


@tool
def search_user_memory(query: str) -> str:
    """
    Search the user's persistent long-term memory across all past conversations.
    Use this if you need to recall specific stored facts, preferences, or details about the user.

    Args:
        query: The search term or topic (e.g. 'name', 'preferences', 'project').
    """
    user_id = get_current_user_id()
    memories = search_memories(user_id=user_id, query=query)
    if not memories:
        return f"No long-term memories found matching '{query}' for user '{user_id}'."

    results = [
        f"- [{m['category']}] {m['key']}: {m['value']} (importance: {m['importance']}/10)"
        for m in memories
    ]
    return f"Found {len(memories)} matching memory item(s):\n" + "\n".join(results)


@tool
def list_user_memories() -> str:
    """
    List all stored long-term memories for the current user.
    Use this when the user asks what you remember about them or asks for a summary of saved facts.
    """
    user_id = get_current_user_id()
    memories = get_memories(user_id=user_id, minimum_importance=1)
    if not memories:
        return f"No long-term memories currently stored for user '{user_id}'."

    results = [
        f"- [{m['category']}] {m['key']}: {m['value']} (importance: {m['importance']}/10)"
        for m in memories
    ]
    return f"Current stored long-term memories for user '{user_id}':\n" + "\n".join(results)


@tool
def delete_user_memory(category: str, key: str) -> str:
    """
    Delete a specific long-term memory item if the user asks to forget, delete, or remove something.

    Args:
        category: The category of the memory (e.g. 'identity', 'preference').
        key: The key of the memory (e.g. 'name', 'favorite_language').
    """
    user_id = get_current_user_id()
    deleted = delete_memory(user_id=user_id, category=category, key=key)
    if deleted:
        return f"Successfully deleted long-term memory '[{category}] {key}' for user '{user_id}'."
    return f"No memory found matching '[{category}] {key}' for user '{user_id}' to delete."


# ============================================================
# OCR TOOLS
# ============================================================

def _configure_tesseract() -> bool:
    """Ensure pytesseract knows the location of the tesseract executable."""
    import shutil
    import os

    if shutil.which("tesseract"):
        return True

    candidate_paths = [
        os.environ.get("TESSERACT_CMD", ""),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/opt/homebrew/bin/tesseract",
    ]
    for path in candidate_paths:
        if path and Path(path).is_file():
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = path
            return True
    return False


def _load_image(image_source: str):
    """Load a PIL Image from a local file path, URL, or base64 data URI."""
    import base64
    import io
    import requests
    from PIL import Image

    source_str = image_source.strip().strip("'\"")

    if source_str.startswith("data:image/") and ";base64," in source_str:
        _, base64_data = source_str.split(";base64,", 1)
        image_bytes = base64.b64decode(base64_data)
        return Image.open(io.BytesIO(image_bytes))

    if source_str.startswith("http://") or source_str.startswith("https://"):
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(source_str, headers=headers, timeout=20)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content))

    local_path = Path(source_str)
    if not local_path.is_file():
        raise FileNotFoundError(f"Image file not found at path: '{source_str}'")

    return Image.open(local_path)


@tool
def ocr_image_scan(image_source: str, target_data: str = "") -> str:
    """
    Perform Optical Character Recognition (OCR) to scan an image and identify certain data.
    Supports local image file paths, web image URLs, and base64 data URIs.

    Args:
        image_source: The path to a local image file (e.g. 'receipt.png', 'C:/path/image.jpg') or an image URL.
        target_data: Optional specific keywords, phrases, numbers, or regex patterns to identify and extract from the image.
                     If omitted or empty, all extracted text from the image is returned.
    """
    import pytesseract
    import re

    try:
        _configure_tesseract()
        image = _load_image(image_source)

        # Convert image to RGB mode if needed (handles RGBA, Palette, etc.)
        if image.mode not in ("L", "RGB"):
            image = image.convert("RGB")

        extracted_text = pytesseract.image_to_string(image).strip()

        if not extracted_text:
            return f"OCR scan completed for '{image_source}', but no readable text was detected in the image."

        # If no specific target data is requested, return the full extracted text
        if not target_data or not target_data.strip():
            return (
                f"=== OCR SCAN RESULTS ===\n"
                f"Source: {image_source}\n\n"
                f"--- EXTRACTED TEXT ---\n"
                f"{extracted_text}"
            )

        # Process target data identification
        targets = [t.strip() for t in re.split(r"[,;\n]+", target_data) if t.strip()]
        lines = extracted_text.splitlines()

        matched_findings = []
        for target in targets:
            term_matches = []
            try:
                pattern = re.compile(re.escape(target), re.IGNORECASE)
            except Exception:
                pattern = None

            for line_idx, line in enumerate(lines, start=1):
                clean_line = line.strip()
                if not clean_line:
                    continue

                is_match = False
                if pattern and pattern.search(clean_line):
                    is_match = True
                elif target.lower() in clean_line.lower():
                    is_match = True

                if is_match:
                    term_matches.append(f"  * Line {line_idx}: {clean_line}")

            if term_matches:
                matched_findings.append(f"- Matches for '{target}':\n" + "\n".join(term_matches))
            else:
                matched_findings.append(f"- Matches for '{target}': None found")

        matches_summary = "\n\n".join(matched_findings)

        return (
            f"=== OCR SCAN & DATA IDENTIFICATION RESULTS ===\n"
            f"Source: {image_source}\n\n"
            f"--- IDENTIFIED TARGET DATA ---\n"
            f"{matches_summary}\n\n"
            f"--- FULL EXTRACTED TEXT ---\n"
            f"{extracted_text}"
        )

    except pytesseract.TesseractNotFoundError:
        return (
            "Error: Tesseract OCR engine was not found on the system. "
            "Please ensure Tesseract OCR is installed and available in PATH or set TESSERACT_CMD."
        )
    except FileNotFoundError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error performing OCR scan on '{image_source}': {str(e)}"
