# encoding:utf-8
"""Provider-reported usage is persisted and shown per assistant reply.

The context donut answers "how full is the window right now?" from the latest
request. These tests pin the separate, per-message view: a reply's complete
provider usage is persisted with the reply, survives a history reload, and is
exposed to the web console.
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.memory.conversation_store import ConversationStore
from agent.protocol.agent_stream import (
    AgentStreamExecutor,
    _cache_hit_tokens,
    _nonnegative_int,
)
from agent.protocol.step_writer import StepWriter


def _user(text):
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def _assistant(text, usage=None):
    message = {"role": "assistant", "content": [{"type": "text", "text": text}]}
    if usage is not None:
        message["extras"] = {"usage": usage}
    return message


def test_history_turn_exposes_persisted_assistant_usage():
    usage = {
        "prompt_tokens": 1200,
        "completion_tokens": 240,
        "total_tokens": 1440,
        "prompt_cache_hit_tokens": 800,
    }
    with tempfile.TemporaryDirectory() as tmp:
        store = ConversationStore(Path(tmp) / "index.db")
        store.append_messages("s", [_user("question"), _assistant("answer", usage)])

        page = store.load_history_page("s")

        assistant = [m for m in page["messages"] if m["role"] == "assistant"][0]
        assert assistant["usage"] == usage


def test_latest_pair_seqs_returns_latest_assistant_usage_for_live_done():
    usage = {
        "prompt_tokens": 20,
        "completion_tokens": 5,
        "total_tokens": 25,
        "prompt_cache_hit_tokens": 12,
    }
    with tempfile.TemporaryDirectory() as tmp:
        store = ConversationStore(Path(tmp) / "index.db")
        store.append_messages("s", [_user("question"), _assistant("answer", usage)])

        latest = store.get_latest_pair_seqs("s")

        assert latest["usage"] == usage


@pytest.mark.parametrize("value, expected", [
    (12, 12), ("12", 12), (0, 0), (-3, 0), ("-3", 0),
    (None, 0), ("", 0), ("invalid", 0), ({}, 0),
    (float("nan"), 0), (float("inf"), 0),
])
def test_usage_counts_are_nonnegative_integers(value, expected):
    assert _nonnegative_int(value) == expected


def test_stream_usage_is_normalized_and_accumulated_across_model_calls():
    first = {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100,
             "prompt_tokens_details": {"cached_tokens": 600}}
    second = {"prompt_tokens": 1400, "completion_tokens": 200, "total_tokens": 1600,
              "prompt_cache_hit_tokens": 1200}

    normalized = AgentStreamExecutor._normalize_stream_usage(first)
    combined = AgentStreamExecutor._merge_usage(
        normalized, AgentStreamExecutor._normalize_stream_usage(second)
    )

    assert normalized["prompt_cache_miss_tokens"] == 400
    assert normalized["calls"] == 1
    assert combined == {
        "prompt_tokens": 2400,
        "completion_tokens": 300,
        "total_tokens": 2700,
        "prompt_cache_hit_tokens": 1800,
        "prompt_cache_miss_tokens": 600,
        "calls": 2,
    }


def test_step_writer_stamps_run_usage_on_a_copy_of_the_answer():
    usage = {"prompt_tokens": 300, "completion_tokens": 40, "total_tokens": 340, "calls": 2}
    tool_step = {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "x", "input": {}}]}
    answer = _assistant("final answer")
    written = []
    writer = StepWriter(written.extend)
    writer.bind(SimpleNamespace(run_usage=usage))

    writer.finish([tool_step, _user("tool result"), answer])

    assert "extras" not in answer
    assert "extras" not in written[0]
    assert written[-1]["extras"]["usage"] == {**usage, "scope": "own"}


def test_team_usage_supports_old_snapshots_without_double_counting_new_totals():
    from agent.protocol.usage import team_usage
    leaf = {"prompt_tokens": 100, "completion_tokens": 10}
    child = team_usage({"input": 200, "output": 20}, [leaf])
    root = team_usage({"input": 300, "output": 30}, [child])
    assert root['input'] == root['prompt_tokens'] == 600
    assert root['total_tokens'] == 660
    assert team_usage(root, [child]) == root


def test_executor_keeps_own_usage_separate_from_delegates():
    executor = object.__new__(AgentStreamExecutor)
    executor.agent = SimpleNamespace(run_usage_total=None)
    executor.on_event = None
    executor.run_usage = AgentStreamExecutor._normalize_stream_usage({
        'prompt_tokens': 100, 'completion_tokens': 10,
    })
    child = AgentStreamExecutor._normalize_stream_usage({
        'prompt_tokens': 200, 'completion_tokens': 20,
    })
    executor._emit_event('peer_message_end', {'usage': child})
    assert executor.run_usage['prompt_tokens'] == 100
    executor._emit_event('delegated_usage', {'usage': child})
    assert executor.run_usage['prompt_tokens'] == 100
    assert executor._ui_run_usage()['total_tokens'] == 110


def test_own_usage_recovers_old_team_snapshot_without_subtracting_from_own_snapshot():
    from agent.protocol.usage import own_usage, team_usage
    child = team_usage({'input': 200, 'output': 20}, [{'input': 50, 'output': 5}])
    root = team_usage({'input': 100, 'output': 10}, [child])
    recovered = own_usage(root, [child])
    assert recovered['prompt_tokens'] == 100
    assert recovered['total_tokens'] == 110
    assert own_usage(recovered, [child]) == recovered


def test_cache_token_helper_rejects_malformed_provider_values():
    # A provider that sends an infinite/negative cache count must not leak a
    # bogus number into the usage figure, and must not raise out of the helper.
    assert _cache_hit_tokens({"prompt_cache_hit_tokens": float("inf")}) == 0
    assert _cache_hit_tokens({"prompt_cache_hit_tokens": -20}) == 0
    # Well-formed values keep working.
    assert _cache_hit_tokens({"prompt_tokens_details": {"cached_tokens": 600}}) == 600


class _UnreportedUsageModel:
    """A provider stream that never reports usage (e.g. a stripping proxy)."""

    model = "no-usage-model"
    context_window = 32000
    max_tokens = 1000
    client = None

    def call_stream(self, request):
        yield {"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}


def test_run_stream_does_not_inherit_previous_turns_usage():
    """A usage-silent run must not report the PREVIOUS turn's token counts.

    Consumers fall back to ``last_usage`` when the run accumulator is empty, so
    a run that never receives usage would otherwise present an old question's
    cost as this turn's.
    """
    from agent.protocol.agent import Agent

    agent = Agent(
        system_prompt="You are a test agent.",
        model=_UnreportedUsageModel(),
        enable_skills=False,
    )
    agent.last_usage = {
        "prompt_tokens": 1234,
        "completion_tokens": 56,
        "total_tokens": 1290,
    }
    agent.run_usage_total = None

    agent.run_stream("hello", clear_history=False)

    assert agent.last_usage is None
    assert agent.run_usage_total is None
