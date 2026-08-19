import sqlite3
import contextvars
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field


logger = logging.getLogger("horizon_memory")

# ============================================================
# CONTEXT VARIABLE FOR CURRENT USER
# ============================================================

current_user_id_context: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_user_id", default="default"
)

def get_current_user_id() -> str:
    """Return the current user ID from the context variable or default."""
    return current_user_id_context.get() or "default"

def set_current_user_id(user_id: str) -> None:
    """Set the current user ID in context."""
    clean_id = (user_id or "default").strip() or "default"
    current_user_id_context.set(clean_id)


# ============================================================
# DATABASE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "nova_memory.db"


# ============================================================
# MEMORY SCHEMAS
# ============================================================

class MemoryItem(BaseModel):
    """
    Represents one piece of long-term information
    that Nova may remember about a user.
    """

    category: str = Field(
        description=(
            "The category of memory: "
            "identity, preference, project, skill, goal, "
            "instruction, context."
        )
    )

    key: str = Field(
        description=(
            "A short identifier for the memory. "
            "Examples: name, nickname, favorite_language, current_project."
        )
    )

    value: str = Field(
        description="The actual information to remember."
    )

    importance: int = Field(
        default=8,
        ge=1,
        le=10,
        description=(
            "How useful this information is for future conversations. "
            "1 is least important and 10 is extremely important."
        ),
    )


class MemoryDecision(BaseModel):
    """
    Result produced by Nova's Memory Gate.
    """

    should_remember: bool = Field(
        description=(
            "Whether this interaction contains information "
            "worth storing as long-term memory."
        )
    )

    memories: list[MemoryItem] = Field(
        default_factory=list,
        description=(
            "The long-term memories extracted from the interaction."
        ),
    )


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    """
    Create a connection to Nova's long-term memory database.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def initialize_memory():
    """
    Create Nova's long-term memory tables if they do not exist.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # --------------------------------------------------------
    # LONG-TERM MEMORIES
    # --------------------------------------------------------
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            category TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            importance INTEGER NOT NULL DEFAULT 5,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(user_id, category, key)
        )
        """
    )

    # --------------------------------------------------------
    # AGENT ACTIONS
    # --------------------------------------------------------
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            action TEXT NOT NULL,
            result TEXT,
            created_at TEXT NOT NULL
        )
        """
    )

    # --------------------------------------------------------
    # INDEXES
    # --------------------------------------------------------
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_memories_user
        ON memories(user_id)
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_memories_importance
        ON memories(user_id, importance)
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_actions_user
        ON actions(user_id, created_at)
        """
    )

    conn.commit()
    conn.close()


# ============================================================
# CANONICAL CATEGORY & KEY NORMALIZATION
# ============================================================

CATEGORY_MAP = {
    "identities": "identity",
    "personal": "identity",
    "user": "identity",
    "user_info": "identity",
    "user_identity": "identity",
    "profile": "identity",
    "preferences": "preference",
    "user_preferences": "preference",
    "pref": "preference",
    "projects": "project",
    "current_project": "project",
    "skills": "skill",
    "expertise": "skill",
    "goals": "goal",
    "instructions": "instruction",
    "system_instruction": "instruction",
    "rule": "instruction",
    "rules": "instruction",
    "contexts": "context",
    "note": "context",
    "notes": "context",
}

KEY_MAP = {
    "user_name": "name",
    "fullname": "name",
    "full_name": "name",
    "first_name": "name",
    "preferred_name": "name",
    "username": "name",
    "user_preference": "preference",
    "preferred_language": "favorite_language",
}


def normalize_category(category: str) -> str:
    """Map categories to canonical names."""
    clean = category.strip().lower()
    return CATEGORY_MAP.get(clean, clean)


def normalize_key(category: str, key: str) -> str:
    """Map keys to canonical names based on category."""
    clean_key = key.strip().lower().replace(" ", "_").replace("-", "_")
    if category == "identity" and clean_key in KEY_MAP:
        return KEY_MAP[clean_key]
    return KEY_MAP.get(clean_key, clean_key)


# ============================================================
# SAVE / UPDATE LONG-TERM MEMORY
# ============================================================

def save_memory(
    user_id: str,
    category: str,
    key: str,
    value: str,
    importance: int = 8,
) -> dict:
    """
    Save or update a long-term memory item.

    Handles canonicalization, conflict resolution, and upserting.
    """
    if not user_id or not user_id.strip():
        user_id = "default"
    user_id = user_id.strip()

    if not category or not category.strip():
        category = "context"

    if not key or not key.strip():
        key = "general_note"

    if not value or not value.strip():
        raise ValueError("Memory value cannot be empty.")

    category = normalize_category(category)
    key = normalize_key(category, key)
    value = value.strip()
    importance = max(1, min(10, importance))

    conn = get_connection()
    now = datetime.now().isoformat()

    # If saving the user's primary name, remove conflicting legacy name keys
    if category == "identity" and key == "name":
        conn.execute(
            """
            DELETE FROM memories
            WHERE user_id = ?
            AND (
                (category IN ('identity', 'personal', 'user_preferences', 'preferences')
                 AND key IN ('name', 'user_name', 'preferred_name', 'fullname', 'full_name'))
            )
            """,
            (user_id,),
        )

    # Check if exact user_id, category, key exists
    existing = conn.execute(
        """
        SELECT id
        FROM memories
        WHERE user_id = ?
        AND category = ?
        AND key = ?
        """,
        (user_id, category, key),
    ).fetchone()

    if existing:
        conn.execute(
            """
            UPDATE memories
            SET value = ?,
                importance = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (value, importance, now, existing["id"]),
        )
        memory_id = existing["id"]
    else:
        cursor = conn.execute(
            """
            INSERT INTO memories (
                user_id,
                category,
                key,
                value,
                importance,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, category, key, value, importance, now, now),
        )
        memory_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return {
        "id": memory_id,
        "user_id": user_id,
        "category": category,
        "key": key,
        "value": value,
        "importance": importance,
        "updated_at": now,
    }


# ============================================================
# GET LONG-TERM MEMORIES
# ============================================================

def get_memories(
    user_id: str,
    minimum_importance: int = 1,
    limit: int = 50,
    category: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve stored long-term memories for a user.
    """
    if not user_id or not user_id.strip():
        user_id = "default"
    user_id = user_id.strip()

    conn = get_connection()

    if category:
        norm_cat = normalize_category(category)
        rows = conn.execute(
            """
            SELECT
                id,
                user_id,
                category,
                key,
                value,
                importance,
                created_at,
                updated_at
            FROM memories
            WHERE user_id = ?
            AND category = ?
            AND importance >= ?
            ORDER BY
                importance DESC,
                updated_at DESC
            LIMIT ?
            """,
            (user_id, norm_cat, minimum_importance, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT
                id,
                user_id,
                category,
                key,
                value,
                importance,
                created_at,
                updated_at
            FROM memories
            WHERE user_id = ?
            AND importance >= ?
            ORDER BY
                importance DESC,
                updated_at DESC
            LIMIT ?
            """,
            (user_id, minimum_importance, limit),
        ).fetchall()

    conn.close()
    return [dict(row) for row in rows]


# ============================================================
# GET ONE MEMORY
# ============================================================

def get_memory(
    user_id: str,
    category: str,
    key: str,
) -> Optional[Dict[str, Any]]:
    """
    Retrieve one specific memory item.
    """
    if not user_id or not user_id.strip():
        user_id = "default"
    user_id = user_id.strip()
    category = normalize_category(category)
    key = normalize_key(category, key)

    conn = get_connection()
    row = conn.execute(
        """
        SELECT
            id,
            user_id,
            category,
            key,
            value,
            importance,
            created_at,
            updated_at
        FROM memories
        WHERE user_id = ?
        AND category = ?
        AND key = ?
        """,
        (user_id, category, key),
    ).fetchone()
    conn.close()

    return dict(row) if row else None


# ============================================================
# SEARCH MEMORIES
# ============================================================

def search_memories(
    user_id: str,
    query: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Search long-term memories by keyword matching across category, key, and value.
    """
    if not user_id or not user_id.strip():
        user_id = "default"
    user_id = user_id.strip()

    search_pattern = f"%{query.strip()}%"

    conn = get_connection()
    rows = conn.execute(
        """
        SELECT
            id,
            user_id,
            category,
            key,
            value,
            importance,
            created_at,
            updated_at
        FROM memories
        WHERE user_id = ?
        AND (key LIKE ? OR value LIKE ? OR category LIKE ?)
        ORDER BY importance DESC, updated_at DESC
        LIMIT ?
        """,
        (user_id, search_pattern, search_pattern, search_pattern, limit),
    ).fetchall()
    conn.close()

    return [dict(row) for row in rows]


# ============================================================
# DELETE ONE MEMORY
# ============================================================

def delete_memory(
    user_id: str,
    category: str,
    key: str,
) -> bool:
    """
    Delete a memory by category and key.
    """
    if not user_id or not user_id.strip():
        user_id = "default"
    user_id = user_id.strip()
    category = normalize_category(category)
    key = normalize_key(category, key)

    conn = get_connection()
    cursor = conn.execute(
        """
        DELETE FROM memories
        WHERE user_id = ?
        AND category = ?
        AND key = ?
        """,
        (user_id, category, key),
    )
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted


def delete_memory_by_id(
    memory_id: int,
) -> bool:
    """
    Delete a specific memory by its primary ID.
    """
    conn = get_connection()
    cursor = conn.execute(
        """
        DELETE FROM memories
        WHERE id = ?
        """,
        (memory_id,),
    )
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted


# ============================================================
# DELETE ALL USER MEMORIES
# ============================================================

def delete_all_memories(
    user_id: str,
) -> int:
    """
    Delete all long-term memories belonging to a user.
    """
    if not user_id or not user_id.strip():
        user_id = "default"
    user_id = user_id.strip()

    conn = get_connection()
    cursor = conn.execute(
        """
        DELETE FROM memories
        WHERE user_id = ?
        """,
        (user_id,),
    )
    conn.commit()
    count = cursor.rowcount
    conn.close()
    return count


# ============================================================
# FORMAT MEMORIES FOR SYSTEM PROMPT
# ============================================================

def format_memories_for_prompt(memories: List[Dict[str, Any]]) -> str:
    """
    Format memories into clear, structured sections for the system prompt.
    """
    if not memories:
        return "No long-term memories stored yet."

    sections: Dict[str, List[str]] = {
        "User Identity": [],
        "User Preferences & Style": [],
        "Instructions & Rules": [],
        "Projects & Skills": [],
        "Other Details": [],
    }

    for mem in memories:
        cat = mem.get("category", "").lower()
        key = mem.get("key", "").replace("_", " ").title()
        val = mem.get("value", "")

        line = f"- {key}: {val}"

        if cat in ("identity", "profile", "personal"):
            sections["User Identity"].append(line)
        elif cat in ("preference", "style"):
            sections["User Preferences & Style"].append(line)
        elif cat in ("instruction", "rule"):
            sections["Instructions & Rules"].append(line)
        elif cat in ("project", "skill", "goal"):
            sections["Projects & Skills"].append(line)
        else:
            sections["Other Details"].append(line)

    output = []
    for header, items in sections.items():
        if items:
            output.append(f"### {header}\n" + "\n".join(items))

    return "\n\n".join(output) if output else "No long-term memories stored yet."


# ============================================================
# SAVE AGENT ACTION
# ============================================================

def save_action(
    user_id: str,
    action: str,
    result: str,
):
    """
    Record an important agent action.
    """
    if not user_id or not user_id.strip():
        user_id = "default"
    user_id = user_id.strip()

    if not action:
        return

    conn = get_connection()
    conn.execute(
        """
        INSERT INTO actions (
            user_id,
            action,
            result,
            created_at
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            user_id,
            action,
            result[:1000] if result else "",
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


# ============================================================
# GET RECENT AGENT ACTIONS
# ============================================================

def get_recent_actions(
    user_id: str,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    Retrieve recent agent actions for context.
    """
    if not user_id or not user_id.strip():
        user_id = "default"
    user_id = user_id.strip()

    conn = get_connection()
    rows = conn.execute(
        """
        SELECT
            id,
            action,
            result,
            created_at
        FROM actions
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (user_id, limit),
    ).fetchall()
    conn.close()

    return [dict(row) for row in rows]


# ============================================================
# MEMORY GATE (AUTOMATIC BACKGROUND EXTRACTION)
# ============================================================

def process_memory(
    llm,
    user_id: str,
    user_message: str,
    assistant_response: str,
) -> List[MemoryItem]:
    """
    Evaluate interaction and automatically extract persistent long-term memories.
    """
    if not user_message or not user_message.strip():
        return []

    if not user_id or not user_id.strip():
        user_id = "default"
    user_id = user_id.strip()

    prompt = f"""You are Horizon's Long-Term Memory Extractor.

Your task is to analyze the conversation turn and decide if the user shared persistent facts that should be remembered ACROSS DIFFERENT CONVERSATIONS.

============================================================
CRITICAL EXTRACTION RULES:
============================================================

1. USER IDENTITY:
   - User's name or what they like to be called (e.g. "My name is Festus", "I'm Alex", "Call me Sam").
   - Extract with: category="identity", key="name", value="<User Name>", importance=10.

2. EXPLICIT SAVE REQUESTS:
   - If the user explicitly asks to save or remember something (e.g., "remember that...", "save this: ...", "don't forget...").
   - Extract with: category="instruction" or appropriate category, importance=10.

3. PERSISTENT USER PREFERENCES:
   - Coding languages, frameworks, formatting preferences, communication styles (e.g. "I prefer Python over JavaScript", "Always give concise answers").
   - Extract with: category="preference", importance=8-9.

4. USER PROJECTS & SKILLS:
   - Ongoing projects or stated professional skills (e.g. "I'm building an e-commerce agent", "I am a frontend engineer").
   - Extract with: category="project" or "skill", importance=7-8.

============================================================
DO NOT STORE:
============================================================
- Greetings, small talk, casual jokes.
- One-time temporary requests (e.g., "Summarize this article", "What is 2+2?").
- Facts generated by the assistant that the user did not say or confirm.
- Transitory queries.

============================================================
CURRENT USER MESSAGE:
============================================================
{user_message}

============================================================
ASSISTANT'S RESPONSE:
============================================================
{assistant_response}
"""

    try:
        decision = llm.with_structured_output(MemoryDecision).invoke(prompt)

        if not decision or not decision.should_remember or not decision.memories:
            return []

        saved_memories = []
        for memory in decision.memories:
            if not memory.category or not memory.key or not memory.value:
                continue

            # Ensure minimum importance threshold
            if memory.importance < 5:
                continue

            save_memory(
                user_id=user_id,
                category=memory.category,
                key=memory.key,
                value=memory.value,
                importance=memory.importance,
            )
            saved_memories.append(memory)

        return saved_memories

    except Exception as e:
        logger.warning(f"Error in process_memory: {e}")
        return []


# ============================================================
# INITIALIZE DATABASE
# ============================================================

initialize_memory()