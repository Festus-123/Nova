"""Long-term memory reliability: dedup, consolidation, caps, gate filtering."""

import uuid
from datetime import datetime

import memory


def _user():
    return f"mem_{uuid.uuid4().hex[:8]}"


def test_same_value_under_different_key_is_not_duplicated():
    user = _user()
    memory.save_memory(user, "project", "current_project", "Nebula agent")
    memory.save_memory(user, "project", "side_project", "nebula agent.")
    assert len(memory.get_memories(user)) == 1


def test_unknown_category_falls_back_to_context():
    user = _user()
    saved = memory.save_memory(user, "random stuff", "fact", "likes tea")
    assert saved["category"] == "context"


def test_returned_importance_matches_stored_value():
    user = _user()
    memory.save_memory(user, "identity", "name", "Festus", importance=10)
    saved = memory.save_memory(user, "identity", "name", "Festus P", importance=6)
    stored = memory.get_memory(user, "identity", "name")
    assert saved["importance"] == stored["importance"] == 10
    assert stored["value"] == "Festus P"


def test_memory_cap_evicts_least_important(monkeypatch):
    user = _user()
    monkeypatch.setattr(memory, "MAX_MEMORIES_PER_USER", 3)
    memory.save_memory(user, "context", "a", "fact a", importance=2)
    memory.save_memory(user, "context", "b", "fact b", importance=9)
    memory.save_memory(user, "context", "c", "fact c", importance=9)
    memory.save_memory(user, "context", "d", "fact d", importance=9)
    keys = sorted(m["key"] for m in memory.get_memories(user))
    assert keys == ["b", "c", "d"]


def test_legacy_synonym_rows_are_consolidated():
    user = _user()
    conn = memory.get_connection()
    now = datetime.now().isoformat()
    rows = [
        (user, "preference", "coding_language", "Python", 9, "2026-01-01T00:00:00"),
        (user, "preference", "favorite_language", "Go", 8, "2026-02-01T00:00:00"),
        (user, "project", "active_project", "Nebula", 8, "2026-01-01T00:00:00"),
        (user, "user_preferences", "style", "concise", 7, "2026-01-01T00:00:00"),
    ]
    for r in rows:
        conn.execute(
            "INSERT INTO memories (user_id, category, key, value, importance, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (*r[:5], now, r[5]),
        )
    conn.commit()
    conn.close()

    memory._consolidate_legacy_duplicates()
    memory._consolidate_legacy_duplicates()  # idempotent

    got = {(m["category"], m["key"]): m["value"] for m in memory.get_memories(user)}
    assert got == {
        ("preference", "favorite_language"): "Go",
        ("project", "current_project"): "Nebula",
        ("preference", "style"): "concise",
    }


def test_actions_table_is_bounded(monkeypatch):
    user = _user()
    monkeypatch.setattr(memory, "MAX_ACTIONS_PER_USER", 5)
    for i in range(12):
        memory.save_action(user, "agent_response", f"r{i}")
    assert len(memory.get_recent_actions(user, limit=100)) == 5


class _FakeStructuredLLM:
    def __init__(self, decision):
        self.decision = decision

    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt):
        return self.decision


def test_memory_gate_filters_transient_secret_and_low_importance():
    user = _user()
    decision = memory.MemoryDecision(
        should_remember=True,
        memories=[
            memory.MemoryItem(category="identity", key="name", value="Festus", importance=10),
            memory.MemoryItem(category="instruction", key="task", value="save a report on AI to file", importance=8),
            memory.MemoryItem(category="context", key="api_token", value="abc", importance=9),
            memory.MemoryItem(category="context", key="mood", value="tired", importance=2),
        ],
    )
    saved = memory.process_memory(_FakeStructuredLLM(decision), user, "My name is Festus", "Nice to meet you")
    assert [m.key for m in saved] == ["name"]
    assert [m["key"] for m in memory.get_memories(user)] == ["name"]


def test_memory_gate_skips_secret_messages():
    user = _user()
    decision = memory.MemoryDecision(
        should_remember=True,
        memories=[memory.MemoryItem(category="context", key="x", value="y", importance=9)],
    )
    assert memory.process_memory(_FakeStructuredLLM(decision), user, "my password is hunter22", "ok") == []
