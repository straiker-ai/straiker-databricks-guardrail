"""HTTP-level tests with a mocked Straiker /api/v3/detect.

Databricks envelopes follow the external service policy API reference; the model bodies inside
them are captured client traffic (tests/fixtures/clients)."""
import base64
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from straiker_databricks import app as app_module
from straiker_databricks.relay import ReplayMemory

KEY = "sk_agt_test.abc.def"
AUTH = {"x-forwarded-email": "caller-sp@databricks"}
FIXTURES = Path(__file__).parent / "fixtures" / "clients"

ALLOW = (200, {"hookSpecificOutput": {"permissionDecision": "allow"},
               "straiker": {"turn_id": "t1", "action": "allow", "controls": [], "blocked_by": [], "events_scored": 1}})
DETECT = (200, {"hookSpecificOutput": {"permissionDecision": "allow"},
                "straiker": {"turn_id": "t3", "action": "detect", "controls": ["custom_commands"], "blocked_by": [],
                             "events_scored": 1}})
DENY = (200, {"hookSpecificOutput": {"permissionDecision": "deny"}, "stopReason": "Blocked by policy.",
              "straiker": {"turn_id": "t2", "action": "block", "controls": ["llm_evasion"],
                           "blocked_by": ["llm_evasion"], "events_scored": 1}})


def captured(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())["body"]


def envelope(kind="model_call", data=None, request_data=None, api_type="openai/v1/chat/completions",
             target="main.agents.support_assistant", policy_config=None, trace_id="4bf92f3577b34da6a3ce929d0e0e4736"):
    event = {"type": kind, "target": target, "data": data if data is not None else
             {"model": "m", "messages": [{"role": "user", "content": "hello"}]}}
    if request_data is not None:
        event["request_data"] = request_data
    if api_type and kind.startswith("model"):
        event["api_type"] = api_type
    if trace_id:
        event["trace_id"] = trace_id
    config = {"phases": "pre_call"}
    if policy_config is not None:
        config["policy_config"] = json.dumps(policy_config) if not isinstance(policy_config, str) else policy_config
    return {"event": event, "config": config}


class Recorder:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def __call__(self, request: httpx.Request):
        self.calls.append(request)
        if isinstance(self.reply, Exception):
            raise self.reply
        status, body = self.reply
        return httpx.Response(status, json=body) if not isinstance(body, str) else httpx.Response(status, text=body)

    def payload(self, i=-1) -> dict:
        return json.loads(self.calls[i].content)


@pytest.fixture
def make_client(monkeypatch):
    def make(reply):
        monkeypatch.setattr(app_module, "REPLAYS", ReplayMemory())
        rec = Recorder(reply)
        state = app_module.app.state
        state.client = httpx.AsyncClient(transport=httpx.MockTransport(rec))
        state.detect_url = app_module.DEFAULT_DETECT_URL
        state.api_key = KEY
        return TestClient(app_module.app), rec
    return make


def post(client, body, headers=AUTH, path="/v1/databricks"):
    return client.post(path, json=body, headers=headers)


# ---------------------------------------------------------------------------------------------
# the Databricks contract


def test_allow_is_200_and_relays_the_provider_body(make_client):
    client, rec = make_client(ALLOW)
    resp = post(client, envelope(data={"model": "m", "user": "alice@example.com",
                                       "messages": [{"role": "user", "content": "hello"}]}))
    assert resp.status_code == 200 and resp.json() == {"result": "ALLOW"}
    sent = rec.payload()
    assert sent["messages"] == [{"role": "user", "content": "hello"}]
    assert sent["session_id"].startswith("databricks-")
    assert sent["original"] == {"processed": {"Meta": {"user": "alice@example.com"}}}
    h = rec.calls[0].headers
    assert h["authorization"] == f"Bearer {KEY}"
    assert h["x-s6r-agent"] == "main.agents.support_assistant"   # one agent per governed service
    assert "x-s6r-client" not in h
    assert h["x-s6r-format"] == "openai.chat"


def test_a_service_id_target_is_labelled(make_client):
    # Live Databricks sends the model service's ID as `target`, not its Unity Catalog name.
    client, rec = make_client(ALLOW)
    post(client, envelope(target="5a3d699d-3bd1-4d96-b33d-b7b12be12f36"))
    assert rec.calls[0].headers["x-s6r-agent"] == "Databricks service 5a3d699d-3bd1-4d96-b33d-b7b12be12f36"


def test_root_path_is_the_policy_endpoint_too(make_client):
    # Databricks posts to "/" when the connection has no API path.
    client, rec = make_client(ALLOW)
    assert post(client, envelope(), path="/").json() == {"result": "ALLOW"}
    assert len(rec.calls) == 1


@pytest.mark.parametrize("path", ["/v1/databricks/", "/v1/databricks", "/"])
def test_databricks_path_is_answered_without_a_redirect(make_client, path):
    # Databricks posts to `<API path>/` and treats a redirect as a failed evaluation (deny).
    client, rec = make_client(ALLOW)
    resp = client.post(path, json=envelope(), headers=AUTH, follow_redirects=False)
    assert resp.status_code == 200 and resp.json() == {"result": "ALLOW"}
    assert len(rec.calls) == 1


def test_deny_carries_the_reason_and_a_resend_is_blocked_again(make_client):
    client, rec = make_client(DENY)
    body = envelope()
    assert post(client, body).json() == {"result": "DENY", "reason": "Blocked by policy."}
    # The platform would answer the resend as a replay (allow); the server blocks it itself.
    rec.reply = ALLOW
    assert post(client, body).json() == {"result": "DENY", "reason": "Blocked by policy."}
    assert len(rec.calls) == 1


def test_model_result_relays_the_answer_with_its_request(make_client):
    client, rec = make_client(ALLOW)
    request = {"model": "m", "messages": [{"role": "user", "content": "Please refund me."}], "metadata": {"x": 1}}
    answer = {"model": "m", "choices": [{"message": {"role": "assistant", "content": "Here is your refund link."}}]}
    assert post(client, envelope("model_result", data=answer, request_data=request)).json() == {"result": "ALLOW"}
    sent = rec.payload()
    assert sent["straiker_phase"] == "response-sync"
    assert sent["request"] == {"model": "m", "messages": request["messages"]}   # allowlisted
    assert json.loads(sent["sse"]) == answer


def test_model_result_deny_blocks_the_answer(make_client):
    client, _ = make_client(DENY)
    resp = post(client, envelope("model_result", data={"choices": []}, request_data={"messages": []}))
    assert resp.json()["result"] == "DENY"


def test_streamed_anthropic_answer_given_as_chunks_is_written_as_sse():
    from straiker_databricks.relay import answer_as_sse, parse_event
    chunks = [{"type": "message_start", "message": {"id": "m1"}}, {"type": "message_stop"}]
    event = parse_event(envelope("model_result", data=chunks, request_data={"messages": []}, api_type="anthropic/v1/messages"))
    assert answer_as_sse(event, "m") == ('event: message_start\ndata: {"type": "message_start", "message": {"id": "m1"}}\n\n'
                                         'event: message_stop\ndata: {"type": "message_stop"}\n')


# ---------------------------------------------------------------------------------------------
# coding agents, from captured bodies


def test_claude_code_follows_the_custom_gateway_contract_by_default(make_client):
    # Coding clients: no X-S6r-Agent; the platform files Claude Code under its own agent.
    client, rec = make_client(ALLOW)
    body = captured("claude_code_sdk_cli_toolloop")
    post(client, envelope(data=body, api_type="anthropic/v1/messages", target="system.ai.databricks-claude-sonnet-4-6"))
    h = rec.calls[0].headers
    session = json.loads(body["metadata"]["user_id"])["session_id"]
    assert "x-s6r-agent" not in h
    assert h["x-s6r-client"] == "claude"
    assert h["x-s6r-format"] == "anthropic.messages"
    assert h["x-claude-code-session-id"] == session
    sent = rec.payload()
    assert sent["session_id"] == session
    assert "metadata" not in sent   # carries a device id
    assert "original" not in sent   # the device is not the user


def test_coding_agents_named_for_the_gateway_on_request(make_client):
    client, rec = make_client(ALLOW)
    body = captured("codex_exec_toolloop")
    post(client, envelope(data=body, api_type="codex/v1/responses", policy_config={"name_coding_agents": True}))
    assert rec.calls[0].headers["x-s6r-agent"] == "Codex (Databricks)"
    assert rec.payload()["session_id"] == body["prompt_cache_key"]
    assert "client_metadata" not in rec.payload() and "prompt_cache_key" not in rec.payload()


def test_opencode_title_call_and_main_call_share_a_session_without_a_stated_one(make_client):
    # Anthropic-format OpenCode states no session in the body (only in a header Databricks drops).
    client, rec = make_client(ALLOW)
    for name in ("opencode_anthropic_first", "opencode_anthropic_title"):
        post(client, envelope(data=captured(name), api_type="anthropic/v1/messages"))
    assert [c.headers["x-s6r-client"] for c in rec.calls] == ["opencode"] * 2
    assert rec.payload(0)["session_id"] == rec.payload(1)["session_id"]


def test_operator_pin_outranks_a_coding_agent_claim(make_client):
    # A body marker is the caller's claim; the operator's pin decides which controls apply.
    client, rec = make_client(ALLOW)
    post(client, envelope(data=captured("claude_code_sdk_cli_first"), api_type="anthropic/v1/messages",
                          policy_config={"agent": "Support Assistant"}))
    h = rec.calls[0].headers
    assert h["x-s6r-agent"] == "Support Assistant" and "x-s6r-client" not in h


def test_codex_follows_the_custom_gateway_rule(make_client):
    # No X-S6r-Agent for a coding client; X-S6r-Client and the session header carry it.
    client, rec = make_client(ALLOW)
    body = captured("codex_exec_first")
    post(client, envelope(data=body, api_type="openai/v1/responses"))
    h = rec.calls[0].headers
    assert "x-s6r-agent" not in h
    assert h["x-s6r-client"] == "codex" and h["x-claude-code-session-id"] == body["prompt_cache_key"]


@pytest.mark.parametrize("name", ["opencode_anthropic_first", "opencode_anthropic_title", "codex_exec_first"])
def test_every_coding_client_carries_its_session_header(make_client, name):
    client, rec = make_client(ALLOW)
    api = "codex/v1/responses" if name.startswith("codex") else "anthropic/v1/messages"
    post(client, envelope(data=captured(name), api_type=api))
    assert rec.calls[0].headers["x-claude-code-session-id"] == rec.payload()["session_id"]


def test_user_is_sent_as_user_name(make_client):
    client, rec = make_client(ALLOW)
    post(client, envelope(policy_config={"user": "svc-support-bot"}))
    assert rec.payload()["user_name"] == "svc-support-bot"


# ---------------------------------------------------------------------------------------------
# MCP


def test_tool_call_is_relayed_as_a_named_tool_call(make_client):
    client, rec = make_client(ALLOW)
    data = {"name": "create_issue", "arguments": {"repo": "acme/website", "title": "Bug"}}
    assert post(client, envelope("tool_call", data=data, target="main.default.github_mcp")).json() == {"result": "ALLOW"}
    sent = rec.payload()
    call = json.loads(sent["sse"])["choices"][0]["message"]["tool_calls"][0]
    assert call["function"] == {"name": "create_issue", "arguments": json.dumps(data["arguments"], sort_keys=True)}
    assert sent["session_id"] == "databricks-trace-4bf92f3577b34da6a3ce929d0e0e4736"
    assert rec.calls[0].headers["x-s6r-agent"] == "main.default.github_mcp"
    assert "x-s6r-format" not in rec.calls[0].headers


def test_tool_result_answers_its_named_call(make_client):
    client, rec = make_client(ALLOW)
    post(client, envelope("tool_result", data={"content": [{"type": "text", "text": "Issue #4213 created"}]},
                          request_data={"name": "create_issue", "arguments": {"repo": "acme/website"}},
                          target="main.default.github_mcp"))
    call, result = rec.payload()["messages"]
    assert call["tool_calls"][0]["function"]["name"] == "create_issue"
    assert result == {"role": "tool", "tool_call_id": call["tool_calls"][0]["id"], "content": "Issue #4213 created"}


@pytest.mark.parametrize("kind, ask_on_detect, reply, expected", [
    ("tool_call", True, DETECT, "ASK"),
    ("tool_call", False, DETECT, "ALLOW"),
    ("tool_call", True, ALLOW, "ALLOW"),       # nothing fired
    ("tool_call", True, DENY, "DENY"),         # a block stays a block
    ("tool_result", True, DETECT, "ALLOW"),    # ASK on a result would not hold anything
    ("model_call", True, DETECT, "ALLOW"),     # ASK on a model event would deny
])
def test_ask_only_on_a_tool_call_that_fired_a_detect_control(make_client, kind, ask_on_detect, reply, expected):
    client, _ = make_client(reply)
    data = {"name": "run_sql", "arguments": {"q": "DROP TABLE x"}} if kind == "tool_call" else \
        {"content": [{"type": "text", "text": "ok"}]} if kind == "tool_result" else None
    body = envelope(kind, data=data, request_data={"name": "run_sql", "arguments": {}} if kind == "tool_result" else None,
                    policy_config={"ask_on_detect": ask_on_detect})
    assert post(client, body).json()["result"] == expected


# ---------------------------------------------------------------------------------------------
# failure: the policy's fail mode, applied by this server, always as HTTP 200


@pytest.mark.parametrize("reply", [
    httpx.ReadTimeout("slow"),
    httpx.ConnectError("down"),
    (500, {"detail": "boom"}),
    (401, {"detail": "Authentication failed"}),
    (200, "not json"),
])
@pytest.mark.parametrize("fail_open, expected", [(False, "DENY"), (True, "ALLOW"), (None, "DENY")])
def test_fail_mode_matrix(make_client, reply, fail_open, expected):
    client, _ = make_client(reply)
    cfg = None if fail_open is None else {"fail_open": fail_open}
    resp = post(client, envelope(policy_config=cfg))
    assert resp.status_code == 200
    assert resp.json()["result"] == expected
    if expected == "DENY":
        assert "Straiker guardrail could not decide" in resp.json()["reason"]


def test_unreadable_policy_config_fails_closed(make_client):
    client, _ = make_client(httpx.ReadTimeout("slow"))
    assert post(client, envelope(policy_config='{"fail_open": tru')).json()["result"] == "DENY"


@pytest.mark.parametrize("fail_open, expected", [(False, "DENY"), (True, "ALLOW")])
def test_unreadable_event_follows_the_fail_mode(make_client, fail_open, expected):
    client, rec = make_client(ALLOW)
    body = {"event": {"type": "something_new", "target": "x", "data": {}},
            "config": {"policy_config": json.dumps({"fail_open": fail_open})}}
    assert post(client, body).json()["result"] == expected
    assert rec.calls == []


@pytest.mark.parametrize("fail_open, expected", [(False, "DENY"), (True, "ALLOW")])
def test_oversize_payload_follows_the_fail_mode(make_client, monkeypatch, fail_open, expected):
    client, rec = make_client(ALLOW)
    monkeypatch.setattr(app_module, "MAX_BODY", 500)
    body = envelope(data={"messages": [{"role": "user", "content": "x" * 1000}]}, policy_config={"fail_open": fail_open})
    assert post(client, body).json()["result"] == expected
    assert rec.calls == []


@pytest.mark.parametrize("action, expected, calls", [("relay", "ALLOW", 1), ("allow", "ALLOW", 0), ("deny", "DENY", 0)])
def test_unknown_api_type(make_client, action, expected, calls):
    client, rec = make_client(ALLOW)
    body = envelope(api_type="typesafe/v1/systemone", policy_config={"unknown_api_type": action})
    assert post(client, body).json()["result"] == expected
    assert len(rec.calls) == calls


def test_embedding_vectors_are_not_sent(make_client):
    client, rec = make_client(DENY)
    body = envelope("model_result", data={"data": [{"embedding": [0.1, 0.2]}]}, request_data={"input": "hi"},
                    api_type="openai/v1/embeddings")
    assert post(client, body).json() == {"result": "ALLOW"}
    assert rec.calls == []


def test_reason_is_bounded(make_client):
    long = (200, {"hookSpecificOutput": {"permissionDecision": "deny"}, "stopReason": "x" * 100_000,
                  "straiker": {"blocked_by": ["llm_evasion"]}})
    client, _ = make_client(long)
    resp = post(client, envelope())
    assert len(resp.json()["reason"]) == 1000 and len(resp.content) < 64 * 1024


def test_tool_credentials_are_redacted(make_client):
    client, rec = make_client(ALLOW)
    data = {"model": "m", "messages": [{"role": "user", "content": "hi"}],
            "mcp_servers": [{"type": "url", "url": "https://mcp.example", "authorization_token": "SECRET"}]}
    post(client, envelope(data=data, api_type="anthropic/v1/messages"))
    assert rec.payload()["mcp_servers"][0]["authorization_token"] == "[redacted]"
    assert "SECRET" not in rec.calls[0].content.decode()


# ---------------------------------------------------------------------------------------------
# Databricks App: the key is the app's secret resource; the Apps proxy authenticates callers


def test_the_key_comes_from_the_secret_resource_not_the_caller(make_client):
    # Databricks' own OAuth token reaches the app; it must never be relayed to Straiker.
    client, rec = make_client(ALLOW)
    resp = client.post("/v1/databricks/", json=envelope(),
                       headers={"x-forwarded-email": "caller-sp@databricks", "authorization": "Bearer databricks-oauth"})
    assert resp.json() == {"result": "ALLOW"}
    assert rec.calls[0].headers["authorization"] == f"Bearer {KEY}"
    assert "databricks-oauth" not in rec.calls[0].content.decode()


def test_no_token_endpoint(make_client):
    client, _ = make_client(ALLOW)
    assert client.post("/oauth/token", data={"grant_type": "client_credentials"}).status_code in (404, 405)


def test_startup_refuses_without_a_key(monkeypatch):
    monkeypatch.delenv("STRAIKER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="STRAIKER_API_KEY"):
        app_module.api_key()


@pytest.mark.parametrize("body, user", [
    ({"model": "m", "input": "hi", "safety_identifier": "u-7f3a"}, "u-7f3a"),
    ({"model": "m", "input": "hi", "user": "u-7f3a"}, "u-7f3a"),
    ({"model": "m", "metadata": {"user_id": "8d1c0a6e"}, "messages": [{"role": "user", "content": "hi"}]}, "8d1c0a6e"),
])
def test_end_user_from_fields_unity_gateway_accepts(make_client, body, user):
    client, rec = make_client(ALLOW)
    api = "openai/v1/responses" if "input" in body else "anthropic/v1/messages"
    post(client, envelope(data=body, api_type=api))
    assert rec.payload()["user_name"] == user


def test_claude_code_device_metadata_is_not_a_user(make_client):
    client, rec = make_client(ALLOW)
    post(client, envelope(data=captured("claude_code_sdk_cli_first"), api_type="anthropic/v1/messages"))
    assert "user_name" not in rec.payload()

