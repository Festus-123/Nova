from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from main import run_research
from memory import (
    get_memories,
    save_memory,
    delete_memory_by_id,
    delete_all_memories,
    search_memories,
)
from history import (
    get_user_conversations,
    get_conversation_history,
    delete_conversation,
    delete_user_conversations,
)


app = FastAPI(
    title="Horizon API",
    description="Research & Long-Term Memory Agent API for Horizon",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://nova-flash-psi.vercel.app",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST & RESPONSE SCHEMAS
# ============================================================

class ResearchRequest(BaseModel):
    query: str
    user_id: str = "default"
    session_id: str = "default"


class ResearchResponse(BaseModel):
    response: str
    user_id: str
    session_id: str


class MemoryCreateRequest(BaseModel):
    user_id: str = "default"
    category: str = Field(..., description="e.g. identity, preference, project, skill, instruction, context")
    key: str = Field(..., description="e.g. name, favorite_language, current_project")
    value: str = Field(..., description="The information to remember")
    importance: int = Field(default=8, ge=1, le=10)


# ============================================================
# SYSTEM HEALTH ENDPOINTS
# ============================================================

@app.get("/")
def home():
    return {
        "status": "Horizon Agent Running",
        "version": "1.1.0",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
    }


# ============================================================
# RESEARCH / AGENT INTERACTION ENDPOINT
# ============================================================

@app.post(
    "/research",
    response_model=ResearchResponse,
)
def research(request: ResearchRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Research query cannot be empty.",
        )

    clean_user_id = (request.user_id or "default").strip() or "default"
    clean_session_id = (request.session_id or "default").strip() or "default"

    try:
        response = run_research(
            query=request.query.strip(),
            user_id=clean_user_id,
            session_id=clean_session_id,
        )

        return {
            "response": response,
            "user_id": clean_user_id,
            "session_id": clean_session_id,
        }

    except Exception as e:
        print("Horizon Error:", e)
        raise HTTPException(
            status_code=500,
            detail=f"Horizon encountered an error: {str(e)}",
        )


# ============================================================
# LONG-TERM MEMORY ENDPOINTS
# ============================================================

@app.get("/memories")
def list_memories(
    user_id: str = Query("default", description="User identifier"),
    category: Optional[str] = Query(None, description="Filter by category"),
    q: Optional[str] = Query(None, description="Search query"),
):
    """Retrieve all persistent long-term memories for a user."""
    clean_user = (user_id or "default").strip() or "default"

    if q and q.strip():
        memories = search_memories(user_id=clean_user, query=q.strip())
    else:
        memories = get_memories(user_id=clean_user, minimum_importance=1, category=category)

    return {
        "user_id": clean_user,
        "count": len(memories),
        "memories": memories,
    }


@app.post("/memories")
def create_memory(item: MemoryCreateRequest):
    """Explicitly add or update a long-term memory item."""
    clean_user = (item.user_id or "default").strip() or "default"

    saved = save_memory(
        user_id=clean_user,
        category=item.category,
        key=item.key,
        value=item.value,
        importance=item.importance,
    )

    return {
        "status": "success",
        "memory": saved,
    }


@app.delete("/memories/{memory_id}")
def delete_single_memory(memory_id: int):
    """Delete a specific long-term memory by ID."""
    deleted = delete_memory_by_id(memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found.")
    return {"status": "success", "deleted_id": memory_id}


@app.delete("/memories")
def clear_all_memories(user_id: str = Query("default")):
    """Delete all long-term memories for a user."""
    clean_user = (user_id or "default").strip() or "default"
    count = delete_all_memories(clean_user)
    return {"status": "success", "deleted_count": count}


# ============================================================
# CONVERSATION HISTORY ENDPOINTS
# ============================================================

@app.get("/conversations")
def list_conversations(user_id: str = Query("default")):
    """List all conversation sessions for a user."""
    clean_user = (user_id or "default").strip() or "default"
    conversations = get_user_conversations(clean_user)
    return {
        "user_id": clean_user,
        "count": len(conversations),
        "conversations": conversations,
    }


@app.get("/conversations/{session_id}/messages")
def get_session_messages(session_id: str, user_id: str = Query("default")):
    """Get message history for a specific conversation session."""
    clean_user = (user_id or "default").strip() or "default"
    messages = get_conversation_history(user_id=clean_user, session_id=session_id, limit=50)
    return {
        "user_id": clean_user,
        "session_id": session_id,
        "count": len(messages),
        "messages": messages,
    }


@app.delete("/conversations/{session_id}")
def delete_session(session_id: str, user_id: str = Query("default")):
    """Delete a conversation session and its messages."""
    clean_user = (user_id or "default").strip() or "default"
    delete_conversation(user_id=clean_user, session_id=session_id)
    return {
        "status": "success",
        "deleted_session_id": session_id,
    }