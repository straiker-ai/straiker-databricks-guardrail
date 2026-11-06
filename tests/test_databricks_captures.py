"""Envelopes captured from a live Databricks Unity Gateway (2026-11-05), not from its reference.

What the captures show that the reference does not: `target` is the service's Unity Catalog ID,
`config` also carries `timeout_ms`, `on_error` and `dry_run`, every call had a `trace_id`, and a
streamed answer arrives in `model_result.data` as one assembled Messages object."""
import json
from pathlib import Path

import pytest

from straiker_databricks.relay import answer_as_sse, build_payload, parse_event, route, session_of

FIXTURES = Path(__file__).parent / "fixtures" / "databricks"
UUID_TARGET = "5a3d699d-3bd1-4d96-b33d-b7b12be12f36"


def envelope(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())["envelope"]


@pytest.mark.parametrize("name", sorted(p.name for p in FIXTURES.glob("*.json")))
def test_every_captured_envelope_parses(name):
    event = parse_event(envelope(name))
    assert event.target == UUID_TARGET and event.trace_id
    assert event.api_type == "anthropic/v1/messages"
    assert set(envelope(name)["config"]) >= {"phases", "timeout_ms", "on_error", "dry_run"}


def test_streamed_claude_code_answer_is_one_assembled_message():
    event = parse_event(envelope("model_result_claude_code_streamed.json"))
    assert isinstance(event.data, dict) and event.data["role"] == "assistant" and "stop_reason" in event.data
    payload = build_payload(event, session_of(event, route(event)))
    assert payload["straiker_phase"] == "response-sync"
    assert json.loads(answer_as_sse(event, "m"))["role"] == "assistant"


def test_claude_code_through_databricks_is_recognised_and_keeps_its_session():
    event = parse_event(envelope("model_call_claude_code.json"))
    r = route(event)
    assert r.client == "claude" and r.agent is None
    session = session_of(event, r)
    assert session == json.loads(event.data["metadata"]["user_id"])["session_id"]
