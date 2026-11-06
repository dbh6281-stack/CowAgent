"""Delegated work survives reload, scoped to the exact original run."""
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.memory.conversation_store import ConversationStore
from agent.tools.agent_delegate.agent_delegate import AgentDelegateTool
from channel.web.api import sessions as api


def handoff(source="primary", target="reviewer", run="run1", root="chat"):
    return {"type": "tool", "name": "agent_delegate", "result": json.dumps({
        "delegated_by": source, "agent_id": target, "run_id": run,
        "session_id": AgentDelegateTool._session_id(source, target, root),
        "content": "answer",
    })}


def seed(store, session, run, text="thinking"):
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "task"}]},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": text},
            {"type": "tool_use", "id": "call", "name": "read", "input": {"path": "a.py"}},
        ]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call", "content": "file contents"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
    ]
    for msg in messages:
        msg["run_id"] = run
    assert store.append_messages(session, messages)


@pytest.fixture
def stored(tmp_path, monkeypatch):
    monkeypatch.setattr("config.conf", lambda: {"enable_thinking": True})
    store = ConversationStore(tmp_path / "history.db", agent_id="reviewer")
    session = AgentDelegateTool._session_id("primary", "reviewer", "chat")
    seed(store, session, "run1", "first task reasoning")
    seed(store, session, "run2", "later task reasoning")
    registry = SimpleNamespace(get=lambda target, **kw: SimpleNamespace(workspace="reviewer"))
    monkeypatch.setattr("agent.registry.get_agent_registry", lambda: registry)
    monkeypatch.setattr("agent.memory.get_conversation_store", lambda workspace: store)
    return store, session


def test_restores_thinking_tools_and_results_without_changing_model_result(stored):
    step = handoff()
    original = step["result"]
    api._add_delegate_histories([step], "chat")
    trace = step["delegation_steps"]
    assert [s["type"] for s in trace] == ["thinking", "tool", "content"]
    assert trace[0]["content"] == "first task reasoning"
    assert trace[1]["result"] == "file contents"
    assert step["result"] == original


def test_exact_agent_session_and_run_are_required(stored):
    store, session = stored
    assert store.load_run_steps(session, "missing") == []
    assert store.load_run_steps("other-chat", "run1") == []
    other = ConversationStore(store._db_path, agent_id="other")
    assert other.load_run_steps(session, "run1") == []
    step = handoff(root="private-chat")
    api._add_delegate_histories([step], "chat")
    assert "delegation_steps" not in step


def test_thinking_toggle_preserves_tools(stored, monkeypatch):
    monkeypatch.setattr("config.conf", lambda: {"enable_thinking": False})
    step = handoff()
    api._add_delegate_histories([step], "chat")
    assert [s["type"] for s in step["delegation_steps"]] == ["tool", "content"]


def test_history_usage_uses_exact_run_latest_snapshot(stored):
    store, session = stored
    store.append_messages(session, [{
        'role': 'assistant', 'content': [{'type': 'text', 'text': 'answer'}],
        'run_id': 'run1', 'extras': {'usage': {'prompt_tokens': 100, 'completion_tokens': 10}},
    }])
    store.append_messages(session, [{
        'role': 'assistant', 'content': [{'type': 'text', 'text': 'answer'}],
        'run_id': 'run2', 'extras': {'usage': {'prompt_tokens': 999}},
    }])
    step = handoff()
    api._add_delegate_histories([step], 'chat')
    assert step['delegation_usage']['total_tokens'] == 110
    assert store.load_run_usage(session, 'missing') is None
    assert store.load_run_usage('other', 'run1') is None


def test_run_usage_is_latest_snapshot_and_isolated_by_agent(stored):
    store, session = stored
    for usage in ({'prompt_tokens': 40}, {'prompt_tokens': 100}, None):
        store.append_messages(session, [{
            'role': 'assistant', 'content': [{'type': 'text', 'text': 'answer'}],
            'run_id': 'run1', 'extras': {'usage': usage},
        }])
    # A later non-assistant record must not override provider accounting.
    store.append_messages(session, [{
        'role': 'user', 'content': [{'type': 'text', 'text': 'task'}],
        'run_id': 'run1', 'extras': {'usage': {'prompt_tokens': 999}},
    }])
    assert store.load_run_usage(session, 'run1') == {'prompt_tokens': 100}
    other = ConversationStore(store._db_path, agent_id='other')
    assert other.load_run_usage(session, 'run1') is None
    assert store.load_run_usage(session, '') is None
    assert store.load_run_usage('', 'run1') is None


def test_missing_or_legacy_trace_leaves_answer_available(stored):
    for result in ("not json", "[]", json.dumps({"content": "legacy answer"}), handoff(run="gone")["result"]):
        step = {"name": "agent_delegate", "result": result}
        api._add_delegate_histories([step], "chat")
        assert "delegation_steps" not in step


def test_nested_handoffs_are_hydrated(stored, monkeypatch):
    inner = handoff(source="reviewer", target="writer", run="nested")
    monkeypatch.setattr("agent.memory.get_conversation_store", lambda workspace: SimpleNamespace(
        load_run_steps=lambda session, run: [inner] if run == "run1" else [{"type": "thinking", "content": "nested thought"}]
    ))
    step = handoff()
    api._add_delegate_histories([step], "chat")
    assert step["delegation_steps"][0]["delegation_steps"][0]["content"] == "nested thought"


def test_web_history_keeps_peer_steps_and_nested_authorship(stored):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for the browser history regression")
    step = handoff()
    api._add_delegate_histories([step], "chat")
    inner = handoff(source="reviewer", target="writer", run="nested")
    inner["delegation_steps"] = [{"type": "thinking", "content": "writer thought"}]
    step["delegation_steps"].insert(1, inner)
    step["delegation_usage"] = {"prompt_tokens": 100, "completion_tokens": 10}
    script = r'''
const fs = require('fs'), vm = require('vm'), assert = require('assert/strict');
const source = fs.readFileSync('channel/web/static/js/chat/render.js', 'utf8');
const ctx = {}; vm.createContext(ctx);
vm.runInContext(source.slice(source.indexOf('function handoffPayload('), source.indexOf('// With keepContent')), ctx);
const step = JSON.parse(fs.readFileSync(0, 'utf8'));
const bubbles = ctx.splitAssistantTurn({content:'main answer', steps:[step], usage:{prompt_tokens:200}});
assert(bubbles.some(b => b.peer?.id === 'reviewer' && b.msg.steps.some(s => s.type === 'thinking')));
assert(bubbles.some(b => b.peer?.id === 'reviewer' && b.msg.steps.some(s => s.name === 'read')));
assert(bubbles.some(b => b.peer?.id === 'writer' && b.msg.steps[0].content === 'writer thought'));
assert.equal(bubbles.at(-1).msg.content, 'main answer');
assert.equal(bubbles.at(-1).peer, null);
assert.equal(bubbles.at(-1).msg.usage.prompt_tokens, 200);
assert.equal(bubbles.filter(b => b.peer?.id === 'reviewer' && b.msg.usage).length, 1);
assert.equal(bubbles.find(b => b.peer?.id === 'reviewer' && b.msg.usage).msg.usage.prompt_tokens, 100);
assert.equal(bubbles[0].msg.usage, null);
'''
    subprocess.run([node, "-e", script], input=json.dumps(step), text=True, check=True,
                   cwd=Path(__file__).resolve().parents[1], capture_output=True)


def test_web_member_badge_displays_numbers_and_prefers_hydrated_team_total():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for the browser history regression")
    script = r'''
const fs = require('fs'), vm = require('vm'), assert = require('assert/strict');
const source = fs.readFileSync('channel/web/static/js/chat/render.js', 'utf8');
const ctx = {
 document: {createElement() {
   const badge = {style:{display:'none'}}, answer = {dataset:{}};
   return {dataset:{}, badge, querySelector(s) {return s === '.turn-tokens' ? badge : answer;}};
 }},
 localizeCancelMarker: s=>s, isCancelMarker: ()=>false,
 renderThinkingHtml: ()=>'', renderToolCallsHtml: ()=>'',
 findAgent: ()=>null, botSpeakerAgent: ()=>null, sharedConversation: ()=>false,
 agentAvatarHTML: ()=>'', escapeHtml:s=>s, renderMarkdown:s=>s,
 formatTime: ()=>'', t:s=>s, renderMessageUsageHtml:()=>'',
 renderBotSpeakerButton:()=>{}, applyHighlighting:()=>{},
 activeAgentId:'primary', currentLang:'zh'
};
vm.createContext(ctx);
vm.runInContext(source.slice(source.indexOf('function createBotMessageEl('),
 source.indexOf('// Append (or replace) a small audio player')), ctx);
const peer = ctx.createBotMessageEl('answer',0,null,
 {usage:{prompt_tokens:100,completion_tokens:10,prompt_cache_hit_tokens:60}}, {id:'reviewer'});
assert.equal(peer.badge.style.display, '');
assert.equal(peer.badge.textContent, '↑100 ↓10 ⚡60');
const root = ctx.createBotMessageEl('answer',0,null,
 {usage:{input:300,output:30},extras:{usage:{input:20,output:2}}}, null);
assert.equal(root.badge.textContent, '↑300 ↓30');
const legacy = ctx.createBotMessageEl('answer',0,null,
 {extras:{usage:{input:20,output:2}}}, null);
assert.equal(legacy.badge.textContent, '↑20 ↓2');
'''
    subprocess.run([node, '-e', script], check=True, capture_output=True,
                   cwd=Path(__file__).resolve().parents[1])
