import os
import sqlite3
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any


logger = logging.getLogger("nebula_history")

# ============================================================
# DATABASE PATH & MIGRATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
# See memory.py: NEBULA_DATA_DIR points the databases at a persistent disk.
DATA_DIR = Path(os.environ.get("NEBULA_DATA_DIR") or BASE_DIR)
LEGACY_DB_PATH = DATA_DIR / "nova_history.db"
DB_PATH = DATA_DIR / "nebula_history.db"


def _migrate_legacy_db_if_needed():
    """Safely migrate legacy nova_history.db to nebula_history.db if needed."""
    if not DB_PATH.exists() and LEGACY_DB_PATH.exists():
        try:
            shutil.copy2(LEGACY_DB_PATH, DB_PATH)
            logger.info("Migrated legacy history database from %s to %s", LEGACY_DB_PATH, DB_PATH)
        except Exception as e:
            logger.warning("Could not auto-copy legacy history database: %s", e)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    """
    Create a robust connection to Nebula's conversation history database.
    Enables foreign keys, WAL mode, and busy timeouts for concurrent stability.
    """
    _migrate_legacy_db_if_needed()
    target_path = DB_PATH if DB_PATH.exists() else (LEGACY_DB_PATH if LEGACY_DB_PATH.exists() else DB_PATH)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(target_path, timeout=30.0)
    conn.row_factory = sqlite3.Row

    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
    except Exception:
        pass

    return conn


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def initialize_history():
    """
    Create the conversation history database and tables
    if they do not already exist.
    """
    _migrate_legacy_db_if_needed()
    conn = get_connection()
    cursor = conn.cursor()

    # --------------------------------------------------------
    # Conversations
    # --------------------------------------------------------
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL UNIQUE,
            user_id TEXT NOT NULL,
            title TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    # --------------------------------------------------------
    # Messages
    # --------------------------------------------------------
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id)
                REFERENCES conversations(session_id)
                ON DELETE CASCADE
        )
        """
    )

    # --------------------------------------------------------
    # Indexes
    # --------------------------------------------------------
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_messages_session
        ON messages(session_id)
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_messages_created
        ON messages(session_id, created_at)
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_conversations_user
        ON conversations(user_id)
        """
    )

    conn.commit()
    conn.close()


# ============================================================
# CREATE / GET CONVERSATION
# ============================================================

def create_conversation(
    user_id: str,
    session_id: str,
    title: Optional[str] = None,
):
    """
    Create a conversation if it does not already exist.
    A session_id uniquely identifies one conversation.
    """
    clean_user = (user_id or "default").strip() or "default"
    clean_session = (session_id or "default").strip() or "default"

    conn = get_connection()
    now = datetime.now().isoformat()

    conn.execute(
        """
        INSERT OR IGNORE INTO conversations (
            session_id,
            user_id,
            title,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            clean_session,
            clean_user,
            title,
            now,
            now,
        ),
    )

    conn.commit()
    conn.close()


# ============================================================
# UPDATE CONVERSATION TIMESTAMP
# ============================================================

def touch_conversation(session_id: str):
    """
    Update the last activity time of a conversation.
    """
    clean_session = (session_id or "default").strip() or "default"
    conn = get_connection()
    now = datetime.now().isoformat()

    conn.execute(
        """
        UPDATE conversations
        SET updated_at = ?
        WHERE session_id = ?
        """,
        (
            now,
            clean_session,
        ),
    )

    conn.commit()
    conn.close()


# ============================================================
# UPDATE CONVERSATION TITLE
# ============================================================

def update_conversation_title(
    session_id: str,
    title: str,
):
    """
    Update the title of a conversation.
    """
    clean_session = (session_id or "default").strip() or "default"
    conn = get_connection()

    conn.execute(
        """
        UPDATE conversations
        SET title = ?
        WHERE session_id = ?
        """,
        (
            title,
            clean_session,
        ),
    )

    conn.commit()
    conn.close()


# ============================================================
# SAVE MESSAGE
# ============================================================

def save_message(
    user_id: str,
    session_id: str,
    role: str,
    content: str,
):
    """
    Save a message to a conversation.
    role should normally be: user, assistant, system, tool.
    """
    allowed_roles = {
        "user",
        "assistant",
        "system",
        "tool",
    }

    if role not in allowed_roles:
        raise ValueError(
            f"Invalid message role: {role}"
        )

    if not content or not content.strip():
        return

    clean_user = (user_id or "default").strip() or "default"
    clean_session = (session_id or "default").strip() or "default"

    # Make sure the conversation exists
    create_conversation(
        user_id=clean_user,
        session_id=clean_session,
    )

    conn = get_connection()
    now = datetime.now().isoformat()

    conn.execute(
        """
        INSERT INTO messages (
            session_id,
            role,
            content,
            created_at
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            clean_session,
            role,
            content.strip(),
            now,
        ),
    )

    conn.commit()
    conn.close()

    # Update conversation activity
    touch_conversation(clean_session)


# ============================================================
# GET CONVERSATION HISTORY
# ============================================================

def get_conversation_history(
    user_id: str,
    session_id: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Retrieve the most recent messages from one conversation.
    Messages are returned in chronological order.
    """
    clean_user = (user_id or "default").strip() or "default"
    clean_session = (session_id or "default").strip() or "default"

    conn = get_connection()

    conversation = conn.execute(
        """
        SELECT session_id
        FROM conversations
        WHERE session_id = ?
        AND user_id = ?
        """,
        (
            clean_session,
            clean_user,
        ),
    ).fetchone()

    if not conversation:
        conn.close()
        return []

    rows = conn.execute(
        """
        SELECT
            id,
            session_id,
            role,
            content,
            created_at
        FROM (
            SELECT *
            FROM messages
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        )
        ORDER BY created_at ASC
        """,
        (
            clean_session,
            limit,
        ),
    ).fetchall()

    conn.close()
    return [dict(row) for row in rows]


# ============================================================
# GET RECENT MESSAGES
# ============================================================

def get_recent_messages(
    user_id: str,
    session_id: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Alias for get_conversation_history().
    """
    return get_conversation_history(
        user_id=user_id,
        session_id=session_id,
        limit=limit,
    )


# ============================================================
# GET SINGLE CONVERSATION
# ============================================================

def get_conversation(
    user_id: str,
    session_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Retrieve information about one conversation.
    """
    clean_user = (user_id or "default").strip() or "default"
    clean_session = (session_id or "default").strip() or "default"

    conn = get_connection()
    row = conn.execute(
        """
        SELECT
            id,
            session_id,
            user_id,
            title,
            created_at,
            updated_at
        FROM conversations
        WHERE session_id = ?
        AND user_id = ?
        """,
        (
            clean_session,
            clean_user,
        ),
    ).fetchone()

    conn.close()
    return dict(row) if row else None


# ============================================================
# GET USER CONVERSATIONS
# ============================================================

def get_user_conversations(
    user_id: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Get conversations belonging to a user.
    """
    clean_user = (user_id or "default").strip() or "default"

    conn = get_connection()
    rows = conn.execute(
        """
        SELECT
            id,
            session_id,
            user_id,
            title,
            created_at,
            updated_at
        FROM conversations
        WHERE user_id = ?
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        (
            clean_user,
            limit,
        ),
    ).fetchall()

    conn.close()
    return [dict(row) for row in rows]


# ============================================================
# DELETE CONVERSATION
# ============================================================

def delete_conversation(
    user_id: str,
    session_id: str,
):
    """
    Delete an entire conversation and all its messages.
    """
    clean_user = (user_id or "default").strip() or "default"
    clean_session = (session_id or "default").strip() or "default"

    conn = get_connection()
    conn.execute(
        """
        DELETE FROM conversations
        WHERE session_id = ?
        AND user_id = ?
        """,
        (
            clean_session,
            clean_user,
        ),
    )

    conn.commit()
    conn.close()


# ============================================================
# DELETE ALL USER CONVERSATIONS
# ============================================================

def delete_user_conversations(
    user_id: str,
):
    """
    Delete every conversation belonging to a user.
    """
    clean_user = (user_id or "default").strip() or "default"

    conn = get_connection()
    conn.execute(
        """
        DELETE FROM conversations
        WHERE user_id = ?
        """,
        (
            clean_user,
        ),
    )

    conn.commit()
    conn.close()


# ============================================================
# COUNT MESSAGES
# ============================================================

def count_messages(
    user_id: str,
    session_id: str,
) -> int:
    """
    Return the number of messages in a conversation.
    """
    clean_user = (user_id or "default").strip() or "default"
    clean_session = (session_id or "default").strip() or "default"

    conn = get_connection()
    row = conn.execute(
        """
        SELECT COUNT(*) AS count
        FROM messages m
        JOIN conversations c
            ON m.session_id = c.session_id
        WHERE m.session_id = ?
        AND c.user_id = ?
        """,
        (
            clean_session,
            clean_user,
        ),
    ).fetchone()

    conn.close()
    return row["count"] if row else 0


# ============================================================
# INITIALIZE WHEN MODULE IS USED
# ============================================================

initialize_history()