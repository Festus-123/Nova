import sqlite3
from pathlib import Path
from datetime import datetime

from pydantic import BaseModel, Field


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
            "The type of memory. Examples: "
            "identity, preference, project, skill, goal, "
            "instruction, context."
        )
    )

    key: str = Field(
        description=(
            "A short identifier for the memory. "
            "Examples: name, favorite_language, current_project."
        )
    )

    value: str = Field(
        description="The actual information to remember."
    )

    importance: int = Field(
        default=5,
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
# SAVE / UPDATE LONG-TERM MEMORY
# ============================================================

def save_memory(
    user_id: str,
    category: str,
    key: str,
    value: str,
    importance: int = 5,
):
    """
    Save a long-term memory.

    If the same user/category/key already exists,
    the existing memory is updated instead of creating
    a duplicate.
    """

    if not user_id:
        raise ValueError("user_id is required.")

    if not category:
        raise ValueError("Memory category is required.")

    if not key:
        raise ValueError("Memory key is required.")

    if not value:
        raise ValueError("Memory value is required.")

    # Keep importance inside the valid range.
    importance = max(
        1,
        min(10, importance),
    )

    conn = get_connection()

    now = datetime.now().isoformat()

    existing = conn.execute(
        """
        SELECT id

        FROM memories

        WHERE user_id = ?
        AND category = ?
        AND key = ?
        """,
        (
            user_id,
            category,
            key,
        ),
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
            (
                value,
                importance,
                now,
                existing["id"],
            ),
        )

    else:

        conn.execute(
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
            (
                user_id,
                category,
                key,
                value,
                importance,
                now,
                now,
            ),
        )

    conn.commit()

    conn.close()


# ============================================================
# GET LONG-TERM MEMORIES
# ============================================================

def get_memories(
    user_id: str,
    minimum_importance: int = 6,
    limit: int = 20,
):
    """
    Retrieve important long-term memories belonging to a user.

    These are the memories that Nova is allowed to use
    when building its context.
    """

    conn = get_connection()

    rows = conn.execute(
        """
        SELECT
            id,
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
        (
            user_id,
            minimum_importance,
            limit,
        ),
    ).fetchall()

    conn.close()

    return [
        dict(row)
        for row in rows
    ]


# ============================================================
# GET ONE MEMORY
# ============================================================

def get_memory(
    user_id: str,
    category: str,
    key: str,
):
    """
    Retrieve one specific memory.
    """

    conn = get_connection()

    row = conn.execute(
        """
        SELECT
            id,
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
        (
            user_id,
            category,
            key,
        ),
    ).fetchone()

    conn.close()

    if not row:
        return None

    return dict(row)


# ============================================================
# DELETE ONE MEMORY
# ============================================================

def delete_memory(
    user_id: str,
    category: str,
    key: str,
):
    """
    Delete one long-term memory.
    """

    conn = get_connection()

    conn.execute(
        """
        DELETE FROM memories

        WHERE user_id = ?
        AND category = ?
        AND key = ?
        """,
        (
            user_id,
            category,
            key,
        ),
    )

    conn.commit()

    conn.close()


# ============================================================
# DELETE ALL USER MEMORIES
# ============================================================

def delete_all_memories(
    user_id: str,
):
    """
    Delete all long-term memories belonging to a user.

    This does NOT delete conversation history.
    """

    conn = get_connection()

    conn.execute(
        """
        DELETE FROM memories

        WHERE user_id = ?
        """,
        (
            user_id,
        ),
    )

    conn.commit()

    conn.close()


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

    Examples:

        web_search
        wikipedia_search
        report_saved
        report_read
        agent_response
    """

    if not user_id:
        raise ValueError("user_id is required.")

    if not action:
        raise ValueError("action is required.")

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
            result,
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
    limit: int = 10,
):
    """
    Retrieve recent agent actions for the user.

    Actions are NOT conversation history.

    They simply provide Nova with lightweight awareness
    of what the agent has recently done.
    """

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
        (
            user_id,
            limit,
        ),
    ).fetchall()

    conn.close()

    return [
        dict(row)
        for row in rows
    ]


# ============================================================
# MEMORY GATE
# ============================================================

def process_memory(
    llm,
    user_id: str,
    user_message: str,
    assistant_response: str,
):
    """
    Decide whether anything from the current interaction
    deserves long-term memory.

    This is the Memory Gate.

    IMPORTANT:

    This function does NOT save conversation history.

    history.py handles conversation history.

    This function only decides whether important information
    should become long-term memory.
    """

    if not user_message:
        return []

    # --------------------------------------------------------
    # MEMORY GATE PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are Nova's Memory Manager.

Your job is to decide whether the CURRENT USER MESSAGE
contains information that should become long-term memory.

Long-term memory should contain information that is likely
to remain useful across future conversations.

The memory system is intentionally selective.

Do NOT save everything.

============================================================
GOOD LONG-TERM MEMORY
============================================================

Examples include:

- The user's name
- A persistent nickname the user explicitly chooses
- Stable user preferences
- Long-term goals
- Important ongoing projects
- Important skills
- Persistent work routines
- Persistent instructions
- Important context that will likely remain useful
- Explicit requests to remember something

============================================================
DO NOT STORE
============================================================

Do NOT store:

- Casual conversation
- Greetings
- Jokes
- Temporary questions
- One-time requests
- General knowledge
- Facts about unrelated people
- Facts about unrelated topics
- The assistant's own statements
- Temporary conversation state
- Every question the user asks
- Every answer Nova gives
- Information that is only useful inside the current conversation

============================================================
VERY IMPORTANT
============================================================

The USER MESSAGE is the primary source of truth.

Do not create memories from information that Nova invented,
guessed, assumed, or merely mentioned in its response.

Only store information that the user actually provided
or explicitly requested Nova to remember.

If the user explicitly says something like:

"Remember my name is John."

"Save this: I prefer TypeScript."

"Remember that Nova's name is Nova Flash 1."

then this is a strong signal that the information should
be remembered.

============================================================
IMPORTANCE
============================================================

Rate each possible memory from 1 to 10.

1-5:
Not important enough for long-term memory.

6-7:
Useful and reasonably persistent.

8-9:
Very useful and likely to matter frequently.

10:
Extremely important identity, preference, instruction,
or other persistent information.

Only memories with importance >= 6 should normally be saved.

============================================================
CURRENT USER MESSAGE
============================================================

{user_message}

============================================================
NOVA'S RESPONSE
============================================================

{assistant_response}

Remember:

The assistant response is NOT a source of user facts.

Only extract memories that are supported by the user's
message.
"""

    # --------------------------------------------------------
    # ASK THE MEMORY GATE
    # --------------------------------------------------------

    decision = llm.with_structured_output(
        MemoryDecision
    ).invoke(prompt)

    # --------------------------------------------------------
    # MEMORY GATE REJECTED EVERYTHING
    # --------------------------------------------------------

    if not decision.should_remember:
        return []

    # --------------------------------------------------------
    # SAVE APPROVED MEMORIES
    # --------------------------------------------------------

    saved_memories = []

    for memory in decision.memories:

        # -----------------------------------------------
        # Validate memory
        # -----------------------------------------------

        if not memory.category:
            continue

        if not memory.key:
            continue

        if not memory.value:
            continue

        # -----------------------------------------------
        # Importance gate
        # -----------------------------------------------

        if memory.importance < 6:
            continue

        # -----------------------------------------------
        # Normalize text
        # -----------------------------------------------

        category = memory.category.strip().lower()

        key = memory.key.strip().lower()

        value = memory.value.strip()

        if not category:
            continue

        if not key:
            continue

        if not value:
            continue

        # -----------------------------------------------
        # Save / update memory
        # -----------------------------------------------

        save_memory(
            user_id=user_id,
            category=category,
            key=key,
            value=value,
            importance=memory.importance,
        )

        saved_memories.append(
            memory
        )

    return saved_memories


# ============================================================
# INITIALIZE DATABASE
# ============================================================

initialize_memory()