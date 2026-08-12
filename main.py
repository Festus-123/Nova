from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from tools import (
    wikipedia_search,
    web_search,
    save_report,
    read_report,
)

from memory import (
    initialize_memory,
    get_memories,
    get_recent_actions,
    save_action,
    process_memory,
)

from history import (
    initialize_history,
    save_message,
    get_recent_messages,
)


# ENVIRONMENT

load_dotenv()


# INITIALIZE DATABASES

initialize_memory()
initialize_history()


# TOOLS

tools = [
    wikipedia_search,
    web_search,
    save_report,
    read_report,
]


# LLM

llm = ChatOpenAI(
    model="gpt-5.4-nano",
    base_url="https://www.routeapi.ai/v1",
    temperature=0.3,
)


# AGENT

agent = create_agent(
    model=llm,
    tools=tools,
)


# BUILD LONG-TERM MEMORY CONTEXT

def build_memory_context(user_id: str):
    """
    Retrieve only the information Nova should know
    from long-term memory.

    Conversation history is handled separately by history.py.
    """

    memories = get_memories(
        user_id=user_id,
        minimum_importance=6,
        limit=20,
    )

    actions = get_recent_actions(
        user_id=user_id,
        limit=10,
    )

    # --------------------------------------------------------
    # Long-term memories
    # --------------------------------------------------------

    if memories:

        memory_text = "\n".join(
            (
                f"- [{memory['category']}] "
                f"{memory['key']}: "
                f"{memory['value']} "
                f"(importance: {memory['importance']}/10)"
            )
            for memory in memories
        )

    else:

        memory_text = "No relevant long-term memories."


    # --------------------------------------------------------
    # Recent agent actions
    # --------------------------------------------------------

    if actions:

        action_text = "\n".join(
            (
                f"- {action['action']}: "
                f"{action['result']}"
            )
            for action in actions
        )

    else:

        action_text = "No recent actions."


    # --------------------------------------------------------
    # Final context
    # --------------------------------------------------------

    return f"""
LONG-TERM MEMORY:

{memory_text}


RECENT AGENT ACTIONS:

{action_text}
""".strip()


# SYSTEM PROMPT

def build_system_prompt(memory_context: str):

    return f"""
You are Nova Flash 1, a reliable AI research assistant.

Your job is to help the user research topics, find information,
use available tools, and provide clear and useful answers.

CORE RULES:

1. Be accurate and honest.

2. Use tools when the user's request requires them.

3. Never claim that you used a tool if you did not actually use it.

4. If a tool returns no useful information, say so honestly.

5. Use long-term memory only when it is relevant to the
   current request.

6. Do not mention the memory system to the user.

7. Do not repeat stored memories unnecessarily.

8. Do not assume that every previous interaction is relevant.

9. The user's current request always takes priority.

10. Conversation history provides context for the current
    conversation. Long-term memory provides persistent
    information about the user.

11. Keep your responses natural and conversational.

12. When performing research, structure the answer clearly
    using headings, bullet points, paragraphs, and lists where
    appropriate.

13. Do not invent sources, tool results, or facts.

AVAILABLE LONG-TERM CONTEXT:

{memory_context}

Use this context silently and only when it helps answer
the user's current request.
""".strip()


# RUN NOVA

def run_research(
    query: str,
    user_id: str = "default",
    session_id: str = "default",
):
    """
    Main execution pipeline for Nova Flash 1.

    Flow:

        User request
              ↓
        Save to history
              ↓
        Retrieve conversation history
              ↓
        Retrieve long-term memory
              ↓
        Build Nova context
              ↓
        Run agent
              ↓
        Get response
              ↓
        Save response to history
              ↓
        Record agent action
              ↓
        Memory Gate evaluates interaction
              ↓
        Return response
    """

    # 1. SAVE USER MESSAGE

    save_message(
        user_id=user_id,
        session_id=session_id,
        role="user",
        content=query,
    )


    # 2. GET CONVERSATION HISTORY

    history = get_recent_messages(
        user_id=user_id,
        session_id=session_id,
        limit=20,
    )


    # 3. GET LONG-TERM MEMORY

    memory_context = build_memory_context(
        user_id=user_id,
    )


    # 4. BUILD NOVA'S MESSAGE CONTEXT

    messages = []


    # --------------------------------------------------------
    # System message
    # --------------------------------------------------------

    messages.append(
        {
            "role": "system",
            "content": build_system_prompt(
                memory_context
            ),
        }
    )


    # --------------------------------------------------------
    # Previous conversation
    # --------------------------------------------------------

    for message in history:

        messages.append(
            {
                "role": message["role"],
                "content": message["content"],
            }
        )


    # 5. RUN AGENT

    result = agent.invoke(
        {
            "messages": messages
        }
    )


    # 6. GET FINAL RESPONSE

    last_message = result["messages"][-1]

    response = last_message.content


    # 7. SAVE NOVA'S RESPONSE TO HISTORY

    save_message(
        user_id=user_id,
        session_id=session_id,
        role="assistant",
        content=response,
    )


    # 8. RECORD AGENT ACTION

    save_action(
        user_id=user_id,
        action="agent_response",
        result=response,
    )


    # 9. MEMORY GATE

    process_memory(
        llm=llm,
        user_id=user_id,
        user_message=query,
        assistant_response=response,
    )


    # 10. RETURN RESPONSE

    return response