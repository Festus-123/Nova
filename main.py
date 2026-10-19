import logging
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from tools import (
    wikipedia_search,
    web_search,
    save_report,
    read_report,
    ocr_image_scan,
    save_user_memory,
    search_user_memory,
    list_user_memories,
    delete_user_memory,
)
from document_sorter.tool import sort_documents

from memory import (
    initialize_memory,
    get_memories,
    format_memories_for_prompt,
    get_recent_actions,
    save_action,
    process_memory,
    set_current_user_id,
)

from history import (
    initialize_history,
    save_message,
    get_recent_messages,
)

logger = logging.getLogger("nebula_main")

# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# INITIALIZE DATABASES
# ============================================================

initialize_memory()
initialize_history()


# ============================================================
# TOOLS
# ============================================================

tools = [
    wikipedia_search,
    web_search,
    save_report,
    read_report,
    ocr_image_scan,
    save_user_memory,
    search_user_memory,
    list_user_memories,
    delete_user_memory,
    sort_documents,
]


# ============================================================
# LLM
# ============================================================

# Defaults preserve the existing deployment; override through env if needed.
NEBULA_MODEL = os.environ.get("NEBULA_MODEL", "gpt-5.4-nano")
NEBULA_LLM_BASE_URL = os.environ.get("NEBULA_LLM_BASE_URL", "https://www.routeapi.ai/v1")

llm = ChatOpenAI(
    model=NEBULA_MODEL,
    base_url=NEBULA_LLM_BASE_URL,
    temperature=0.3,
)


# ============================================================
# AGENT
# ============================================================

agent = create_agent(
    model=llm,
    tools=tools,
)


# ============================================================
# BUILD LONG-TERM MEMORY CONTEXT
# ============================================================

def build_memory_context(user_id: str) -> str:
    """
    Retrieve and format long-term memories for this user
    so they persist across every conversation session.
    """
    try:
        memories = get_memories(
            user_id=user_id,
            minimum_importance=1,
            limit=50,
        )
        actions = get_recent_actions(
            user_id=user_id,
            limit=3,
        )
    except Exception as e:
        # Memory must never take the agent down; answer without it.
        logger.warning(f"Long-term memory unavailable: {e}")
        return "PERSISTENT USER MEMORY: unavailable for this request."

    formatted_memories = format_memories_for_prompt(memories)

    if actions:
        action_text = "\n".join(
            f"- {action['action']}: {(action['result'] or '')[:200]}"
            for action in actions
        )
    else:
        action_text = "None."

    return f"""PERSISTENT USER MEMORY (Available across all conversations):
{formatted_memories}

RECENT AGENT ACTIONS:
{action_text}"""


# ============================================================
# SYSTEM PROMPT
# ============================================================

def build_system_prompt(memory_context: str) -> str:
    return f"""You are Nebula, an intelligent and reliable AI research assistant.

CORE CAPABILITIES & RULES:

1. PERSISTENT LONG-TERM MEMORY:
   - You have persistent memory about this user that remains intact ACROSS DIFFERENT CONVERSATIONS.
   - When a new conversation starts, you ALREADY know the user's name, preferences, project details, and past instructions if they are listed in the PERSISTENT USER MEMORY section below.
   - If the user asks "What is my name?", "Who am I?", "Do you remember me?", or asks about their preferences, check the PERSISTENT USER MEMORY section and answer accurately and directly.
   - If the user provides personal facts (e.g. "My name is Festus", "I prefer Python", "I'm working on Nebula") or explicitly asks you to remember/save something, use the `save_user_memory` tool or confirm it is saved.
   - If the user asks to forget or delete saved info, use the `delete_user_memory` tool.
   - Address the user naturally by name if their name is known from memory.

2. RESEARCH & TOOLS:
   - Use `web_search` or `wikipedia_search` when fresh information or factual verification is needed.
   - Use `save_report` / `read_report` for saving long markdown reports to files.
   - Use `ocr_image_scan` to read text or extract target data from images.
   - Use `sort_documents` to group and classify local document files by a metadata schema.
   - Never fabricate tool results or claims of tool usage.

3. ACCURACY & CONVERSATION STYLE:
   - Be clear, structured, and helpful.
   - Maintain context across conversation history for the current session, while leveraging long-term memory for cross-session continuity.

============================================================
CURRENT USER'S PERSISTENT CONTEXT:
============================================================
{memory_context}
""".strip()


# ============================================================
# RUN NEBULA EXECUTION PIPELINE
# ============================================================

def run_research(
    query: str,
    user_id: str = "default",
    session_id: str = "default",
) -> str:
    """
    Main execution pipeline for Nebula.

    Ensures seamless memory persistence across sessions.
    """
    clean_user_id = (user_id or "default").strip() or "default"
    clean_session_id = (session_id or "default").strip() or "default"

    # Set context variable for tools
    set_current_user_id(clean_user_id)

    # 1. SAVE USER MESSAGE TO HISTORY
    save_message(
        user_id=clean_user_id,
        session_id=clean_session_id,
        role="user",
        content=query,
    )

    # 2. GET CONVERSATION HISTORY (Current session)
    history = get_recent_messages(
        user_id=clean_user_id,
        session_id=clean_session_id,
        limit=20,
    )

    # 3. GET PERSISTENT LONG-TERM MEMORY (Across all sessions)
    memory_context = build_memory_context(
        user_id=clean_user_id,
    )

    # 4. BUILD MESSAGE CONTEXT
    messages = [
        {
            "role": "system",
            "content": build_system_prompt(memory_context),
        }
    ]

    for message in history:
        messages.append(
            {
                "role": message["role"],
                "content": message["content"],
            }
        )

    # 5. RUN AGENT
    result = agent.invoke({"messages": messages})

    # 6. EXTRACT RESPONSE
    last_message = result["messages"][-1]
    response = last_message.content

    # 7. SAVE RESPONSE TO HISTORY
    save_message(
        user_id=clean_user_id,
        session_id=clean_session_id,
        role="assistant",
        content=response,
    )

    # 8. RECORD ACTION
    try:
        save_action(
            user_id=clean_user_id,
            action="agent_response",
            result=response,
        )
    except Exception as e:
        logger.warning(f"Could not record agent action: {e}")

    # 9. BACKGROUND MEMORY EXTRACTION (Memory Gate)
    try:
        process_memory(
            llm=llm,
            user_id=clean_user_id,
            user_message=query,
            assistant_response=response,
        )
    except Exception as e:
        logger.warning(f"Background memory extraction error: {e}")

    # 10. RETURN RESPONSE
    return response