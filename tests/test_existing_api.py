"""Existing Nebula functionality: research, memory and conversation endpoints."""

import uuid

import pytest


def _user():
    return f"user_{uuid.uuid4().hex[:8]}"


# ============================================================
# 20  /research
# ============================================================

def test_research_endpoint_contract(client, monkeypatch):
    import server

    calls = {}

    def fake_run_research(query, user_id, session_id):
        calls.update(query=query, user_id=user_id, session_id=session_id)
        return "answer"

    monkeypatch.setattr(server, "run_research", fake_run_research)
    resp = client.post("/research", json={"query": "  what is RAG? ", "user_id": " u1 ", "session_id": "s1"})
    assert resp.status_code == 200
    assert resp.json() == {"response": "answer", "user_id": "u1", "session_id": "s1"}
    assert calls == {"query": "what is RAG?", "user_id": "u1", "session_id": "s1"}


def test_research_rejects_empty_query(client):
    assert client.post("/research", json={"query": "   "}).status_code == 400


def test_research_error_does_not_leak_details(client, monkeypatch):
    import server

    def boom(**kwargs):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(server, "run_research", boom)
    resp = client.post("/research", json={"query": "hi"})
    assert resp.status_code == 500
    assert "secret internal detail" not in resp.text
    assert "Nebula" in resp.json()["detail"]


def test_run_research_pipeline_uses_history_and_memory(monkeypatch):
    import main
    from history import get_conversation_history
    from langchain_core.messages import AIMessage
    from memory import save_memory

    user, session = _user(), "sess-1"
    save_memory(user, "identity", "name", "Festus", 10)
    captured = {}

    class FakeAgent:
        def invoke(self, payload):
            captured["messages"] = payload["messages"]
            return {"messages": [AIMessage(content="Hello Festus")]}

    monkeypatch.setattr(main, "agent", FakeAgent())
    monkeypatch.setattr(main, "process_memory", lambda **kwargs: [])

    assert main.run_research("hi", user_id=user, session_id=session) == "Hello Festus"
    system_prompt = captured["messages"][0]["content"]
    assert "Nebula" in system_prompt and "Festus" in system_prompt
    roles = [m["role"] for m in get_conversation_history(user, session)]
    assert roles == ["user", "assistant"]


def test_run_research_survives_memory_outage(monkeypatch):
    import main
    from langchain_core.messages import AIMessage

    def broken(**kwargs):
        raise RuntimeError("database is locked")

    class FakeAgent:
        def invoke(self, payload):
            return {"messages": [AIMessage(content="ok")]}

    monkeypatch.setattr(main, "get_memories", broken)
    monkeypatch.setattr(main, "agent", FakeAgent())
    monkeypatch.setattr(main, "process_memory", lambda **kwargs: [])
    assert main.run_research("hi", user_id=_user(), session_id="s") == "ok"


# ============================================================
# 21  MEMORY ENDPOINTS
# ============================================================

def test_memory_create_list_search_delete(client):
    user = _user()
    created = client.post(
        "/memories",
        json={"user_id": user, "category": "preference", "key": "favorite_language", "value": "Python"},
    )
    assert created.status_code == 200
    memory_id = created.json()["memory"]["id"]

    listed = client.get("/memories", params={"user_id": user}).json()
    assert listed["count"] == 1 and listed["memories"][0]["value"] == "Python"

    found = client.get("/memories", params={"user_id": user, "q": "what is my favorite language"}).json()
    assert found["count"] == 1

    by_category = client.get("/memories", params={"user_id": user, "category": "identity"}).json()
    assert by_category["count"] == 0

    assert client.delete(f"/memories/{memory_id}").json() == {"status": "success", "deleted_id": memory_id}
    assert client.delete(f"/memories/{memory_id}").status_code == 404


def test_memory_update_replaces_instead_of_duplicating(client):
    user = _user()
    client.post("/memories", json={"user_id": user, "category": "preference", "key": "favorite_language", "value": "Python"})
    client.post("/memories", json={"user_id": user, "category": "preferences", "key": "coding_language", "value": "C++"})
    memories = client.get("/memories", params={"user_id": user}).json()["memories"]
    assert [(m["category"], m["key"], m["value"]) for m in memories] == [("preference", "favorite_language", "C++")]


def test_memory_rejects_secrets(client):
    user = _user()
    for key, value in [
        ("note", "my api_key = sk-abcdefghijklmnopqrstuvwxyz123456"),
        ("wifi_password", "hunter22"),
        ("note", "my password is hunter22"),
    ]:
        resp = client.post("/memories", json={"user_id": user, "category": "context", "key": key, "value": value})
        assert resp.status_code == 400, (key, value)
    assert client.get("/memories", params={"user_id": user}).json()["count"] == 0


def test_delete_all_memories_is_scoped_to_user(client):
    a, b = _user(), _user()
    client.post("/memories", json={"user_id": a, "category": "identity", "key": "name", "value": "Ada"})
    client.post("/memories", json={"user_id": b, "category": "identity", "key": "name", "value": "Bola"})
    assert client.delete("/memories", params={"user_id": a}).json()["deleted_count"] == 1
    assert client.get("/memories", params={"user_id": b}).json()["count"] == 1


# ============================================================
# 22  CONVERSATION ENDPOINTS
# ============================================================

def test_conversation_list_retrieve_delete(client):
    from history import save_message

    user = _user()
    save_message(user, "s-1", "user", "hello")
    save_message(user, "s-1", "assistant", "hi there")

    listed = client.get("/conversations", params={"user_id": user}).json()
    assert listed["count"] == 1 and listed["conversations"][0]["session_id"] == "s-1"

    messages = client.get("/conversations/s-1/messages", params={"user_id": user}).json()
    assert [m["content"] for m in messages["messages"]] == ["hello", "hi there"]

    other = client.get("/conversations/s-1/messages", params={"user_id": _user()}).json()
    assert other["count"] == 0

    assert client.delete("/conversations/s-1", params={"user_id": user}).json()["status"] == "success"
    assert client.get("/conversations", params={"user_id": user}).json()["count"] == 0


def test_cors_no_longer_wildcard(client):
    resp = client.options(
        "/health",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"},
    )
    assert resp.headers.get("access-control-allow-origin") != "https://evil.example"
    ok = client.options(
        "/health",
        headers={"Origin": "https://nova-flash-psi.vercel.app", "Access-Control-Request-Method": "GET"},
    )
    assert ok.headers.get("access-control-allow-origin") == "https://nova-flash-psi.vercel.app"
