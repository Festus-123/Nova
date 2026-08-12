import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional


# ============================================================
# DATABASE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "nova_history.db"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    """
    Create a connection to Nova's conversation history database.
    """

    conn = sqlite3.connect(DB_PATH)

    conn.row_factory = sqlite3.Row

    # Enable foreign key support
    conn.execute("PRAGMA foreign_keys = ON")

    return conn


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def initialize_history():
    """
    Create the conversation history database and tables
    if they do not already exist.
    """

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
            session_id,
            user_id,
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
            session_id,
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

    conn = get_connection()

    conn.execute(
        """
        UPDATE conversations

        SET title = ?

        WHERE session_id = ?
        """,
        (
            title,
            session_id,
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

    role should normally be:

        user
        assistant
        system
        tool
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

    # Make sure the conversation exists
    create_conversation(
        user_id=user_id,
        session_id=session_id,
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
            session_id,
            role,
            content,
            now,
        ),
    )

    conn.commit()

    conn.close()

    # Update conversation activity
    touch_conversation(session_id)


# ============================================================
# GET CONVERSATION HISTORY
# ============================================================

def get_conversation_history(
    user_id: str,
    session_id: str,
    limit: int = 20,
):
    """
    Retrieve the most recent messages from one conversation.

    Messages are returned in chronological order so they can
    be passed directly to Nova.
    """

    conn = get_connection()

    # First make sure the conversation belongs to this user.
    conversation = conn.execute(
        """
        SELECT session_id

        FROM conversations

        WHERE session_id = ?
        AND user_id = ?
        """,
        (
            session_id,
            user_id,
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
            session_id,
            limit,
        ),
    ).fetchall()

    conn.close()

    return [
        dict(row)
        for row in rows
    ]


# ============================================================
# GET RECENT MESSAGES
# ============================================================

def get_recent_messages(
    user_id: str,
    session_id: str,
    limit: int = 20,
):
    """
    Alias for get_conversation_history().

    This makes the intent clearer when used by main.py.
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
):
    """
    Retrieve information about one conversation.
    """

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
            session_id,
            user_id,
        ),
    ).fetchone()

    conn.close()

    if not row:
        return None

    return dict(row)


# ============================================================
# GET USER CONVERSATIONS
# ============================================================

def get_user_conversations(
    user_id: str,
    limit: int = 50,
):
    """
    Get conversations belonging to a user.

    Useful later for building the chat history sidebar
    in the frontend.
    """

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
# DELETE CONVERSATION
# ============================================================

def delete_conversation(
    user_id: str,
    session_id: str,
):
    """
    Delete an entire conversation and all its messages.

    Because messages use ON DELETE CASCADE, deleting the
    conversation automatically deletes its messages.
    """

    conn = get_connection()

    conn.execute(
        """
        DELETE FROM conversations

        WHERE session_id = ?
        AND user_id = ?
        """,
        (
            session_id,
            user_id,
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

    This does NOT affect long-term memory.
    """

    conn = get_connection()

    conn.execute(
        """
        DELETE FROM conversations

        WHERE user_id = ?
        """,
        (
            user_id,
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
):
    """
    Return the number of messages in a conversation.
    """

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
            session_id,
            user_id,
        ),
    ).fetchone()

    conn.close()

    return row["count"] if row else 0


# ============================================================
# INITIALIZE WHEN MODULE IS USED
# ============================================================

initialize_history()