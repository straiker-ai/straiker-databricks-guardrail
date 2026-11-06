"""Map a Databricks external service policy call onto Straiker's v3 detect contract, and back.

The Straiker side is what the Kong plugin, the LiteLLM guardrail and the TrueFoundry service send
to ``POST /api/v3/detect``: the provider body on the request phase,
``{straiker_phase: "response-sync", sse, model, request}`` on the answer, ``session_id`` and the
end user in ``original.processed.Meta.user``, and the ``x-s6r-*`` routing headers.

The Databricks side is the external service policy API (Beta, reference read 2026-11-05):

* body: ``{"event": {type, target, data, request_data, api_type, trace_id}, "config": {policy_config, phases}}``
* ``type``: ``model_call`` / ``model_result`` (Model and Model Provider Services), ``tool_call`` /
  ``tool_result`` (MCP Services)
* ``data`` is the caller's body verbatim in the format ``api_type`` names; ``request_data`` is the
  originating request, on results only
* ``target`` is the governed service's Unity Catalog name; ``trace_id`` is optional
* ``config.policy_config`` is a JSON object in a string, typed by the customer per policy and
  never read by Databricks: it is this server's settings channel
* the event carries no identity and no client headers
* the reply is HTTP 200 ``{"result": "ALLOW" | "DENY" | "ASK", "reason"}``, at most 64 KB, within
  5 s; anything else and Databricks denies
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Final, Mapping

from . import clients

GATEWAY_NAME: Final = "Databricks"
RESPONSE_PHASE: Final = "response-sync"
BLOCK_DECISIONS: Final = frozenset({"block", "deny"})
DERIVED_SESSION_PREFIX: Final = "databricks-"
MAX_REASON_CHARS: Final = 1000

AGENT_HEADER: Final = "x-s6r-agent"
CLIENT_HEADER: Final = "x-s6r-client"
FORMAT_HEADER: Final = "x-s6r-format"
CLAUDE_SESSION_HEADER: Final = "x-claude-code-session-id"

MODEL_CALL: Final = "model_call"
MODEL_RESULT: Final = "model_result"
TOOL_CALL: Final = "tool_call"
TOOL_RESULT: Final = "tool_result"
EVENT_TYPES: Final = frozenset({MODEL_CALL, MODEL_RESULT, TOOL_CALL, TOOL_RESULT})

#: `api_type` → the format family this server relays it as. From the reference's table.
API_FORMATS: Final[Mapping[str, str]] = {
    "mlflow/v1/chat/completions": "openai.chat",
    "openai/v1/chat/completions": "openai.chat",
    "cursor/v1/chat/completions": "openai.chat",
    "mlflow/v1/completions": "openai.completions",
    "openai/v1/completions": "openai.completions",
    "mlflow/v1/embeddings": "embeddings",
    "openai/v1/embeddings": "embeddings",
    "mlflow/v1/responses": "responses",
    "openai/v1/responses": "responses",
    "codex/v1/responses": "responses",
    "anthropic/v1/messages": "anthropic.messages",
    "gemini/v1/generateContent": "gemini",
    "gemini/v1/streamGenerateContent": "gemini",
}
#: `x-s6r-format` breaks the one tie the body cannot: chat completions and Messages both carry
#: `messages`. Gemini is relayed translated to chat completions, so it takes that hint.
FORMAT_HINTS: Final[Mapping[str, str]] = {
    "openai.chat": "openai.chat",
    "anthropic.messages": "anthropic.messages",
    "gemini": "openai.chat",
}
UNKNOWN_API_TYPE_ACTIONS: Final = frozenset({"relay", "allow", "deny"})

# Provider body fields relayed to Straiker: the LiteLLM and TrueFoundry allowlist. An allowlist so
# a field a client adds later is not relayed by accident. `metadata`, `client_metadata` and
# `prompt_cache_key` are read for the session here and not relayed: they carry device and
# installation ids.
PROVIDER_BODY_KEYS: Final = frozenset({
    # OpenAI chat completions
    "model", "messages", "tools", "tool_choice", "functions", "function_call", "temperature",
    "top_p", "n", "stream", "stream_options", "stop", "max_tokens", "max_completion_tokens",
    "presence_penalty", "frequency_penalty", "logit_bias", "user", "response_format", "seed",
    "logprobs", "top_logprobs", "parallel_tool_calls", "reasoning_effort", "modalities", "audio",
    "prediction", "store", "service_tier", "web_search_options",
    # OpenAI text completions
    "prompt", "suffix", "echo", "best_of",
    # Anthropic messages
    "system", "stop_sequences", "top_k", "thinking", "container", "mcp_servers",
    "context_management", "output_format",
    # OpenAI responses
    "input", "instructions", "previous_response_id", "truncation", "text", "include",
    "reasoning", "max_output_tokens", "background", "conversation",
    # embeddings
    "encoding_format", "dimensions",
    # Conversation grouping a client may state itself
    "session_id",
})
CREDENTIAL_FIELDS: Final = frozenset({"authorization_token", "authorization", "headers"})
REDACTED_KEYS: Final = frozenset({"tools", "mcp_servers"})
REDACTED_VALUE: Final = "[redacted]"


class GuardrailError(Exception):
    """A call this server cannot read. Answered as the policy's failure verdict."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


# ---------------------------------------------------------------------------------------------
# settings and envelope


def _flag(cfg: Mapping[str, Any], key: str, default: bool) -> bool:
    value = cfg.get(key)
    return value if isinstance(value, bool) else default


def _text(cfg: Mapping[str, Any], key: str) -> str | None:
    value = cfg.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else None


@dataclass(frozen=True)
class Settings:
    """Per-policy settings: the customer's `policy_config`, a JSON object in a string."""

    agent: str | None = None            # names ONE application; outranks everything else
    user: str | None = None             # a fixed user, for a service principal's application
    fail_open: bool = False             # Straiker unreachable or late: ALLOW instead of DENY
    ask_on_detect: bool = False         # MCP tool call that fires a detect-mode control: ASK
    name_coding_agents: bool = False    # True: "Claude Code (Databricks)", … under this integration
    unknown_api_type: str = "relay"     # relay | allow | deny, for an api_type not in the table
    format_hint: str | None = None      # x-s6r-format override
    error: str | None = None            # why policy_config could not be read, for the log

    @classmethod
    def from_policy_config(cls, raw: object) -> "Settings":
        if raw is None or raw == "":
            return cls()
        try:
            cfg = json.loads(raw) if isinstance(raw, str) else raw
        except ValueError:
            return cls(error="policy_config is not JSON; using defaults")
        if not isinstance(cfg, Mapping):
            return cls(error="policy_config is not a JSON object; using defaults")
        unknown = _text(cfg, "unknown_api_type") or "relay"
        return cls(
            agent=_text(cfg, "agent"),
            user=_text(cfg, "user"),
            fail_open=_flag(cfg, "fail_open", False),
            ask_on_detect=_flag(cfg, "ask_on_detect", False),
            name_coding_agents=_flag(cfg, "name_coding_agents", False),
            unknown_api_type=unknown if unknown in UNKNOWN_API_TYPE_ACTIONS else "relay",
            format_hint=_text(cfg, "format_hint"),
        )


@dataclass(frozen=True)
class Event:
    type: str
    target: str
    data: Any
    request_data: Any
    api_type: str | None
    trace_id: str | None
    phases: str | None
    settings: Settings

    @property
    def family(self) -> str | None:
        return API_FORMATS.get(self.api_type or "")

    @property
    def is_mcp(self) -> bool:
        return self.type in (TOOL_CALL, TOOL_RESULT)

    @property
    def request(self) -> Mapping[str, Any]:
        """The model request this event is about: `data` on a call, `request_data` on a result."""
        body = self.data if self.type == MODEL_CALL else self.request_data
        return body if isinstance(body, Mapping) else {}


def settings_of(body: object) -> Settings:
    """The policy's settings, read even when the event is unusable, so a failure is answered
    with the policy's own fail mode."""
    config = body.get("config") if isinstance(body, Mapping) else None
    raw = config.get("policy_config") if isinstance(config, Mapping) else None
    return Settings.from_policy_config(raw)


def parse_event(body: object) -> Event:
    if not isinstance(body, Mapping):
        raise GuardrailError("body must be a JSON object")
    event = body.get("event")
    if not isinstance(event, Mapping):
        raise GuardrailError("event is missing or not an object")
    kind = event.get("type")
    if kind not in EVENT_TYPES:
        raise GuardrailError(f"event.type {kind!r} is not one of {sorted(EVENT_TYPES)}")
    target = event.get("target")
    if not isinstance(target, str) or not target.strip():
        raise GuardrailError("event.target is missing")
    if "data" not in event:
        raise GuardrailError("event.data is missing")
    config = body.get("config") if isinstance(body.get("config"), Mapping) else {}
    api_type = event.get("api_type")
    trace_id = event.get("trace_id")
    phases = config.get("phases")
    return Event(
        type=kind,
        target=target.strip(),
        data=event.get("data"),
        request_data=event.get("request_data"),
        api_type=api_type if isinstance(api_type, str) and api_type else None,
        trace_id=trace_id if isinstance(trace_id, str) and trace_id else None,
        phases=phases if isinstance(phases, str) else None,
        settings=settings_of(body),
    )


# ---------------------------------------------------------------------------------------------
# provider bodies


def _without_credentials(entries: object) -> object:
    if not isinstance(entries, (list, tuple)):
        return entries
    return [
        {str(k): (REDACTED_VALUE if str(k).lower() in CREDENTIAL_FIELDS else v) for k, v in e.items()}
        if isinstance(e, Mapping) else e
        for e in entries
    ]


def provider_body(body: Mapping[str, Any]) -> dict[str, Any]:
    """The client's request as Straiker should see it: allowlisted, credentials redacted."""
    out = {
        key: (_without_credentials(value) if key in REDACTED_KEYS else value)
        for key, value in body.items()
        if key in PROVIDER_BODY_KEYS
    }
    # The Responses API takes `input` as a string or a list; Straiker scores only the list form
    # (measured 2026-10-05 for TrueFoundry: string 0 events, list 1).
    if isinstance(out.get("input"), str):
        out["input"] = [{"role": "user", "content": out["input"]}]
    return out


# Gemini `generateContent` has no reader on the platform, so it is relayed as the chat completion
# it means: `contents` → `messages`, `systemInstruction` → a system message, `functionCall` →
# `tool_calls`, `functionResponse` → a `tool` message. From Google's published format; no
# Gemini client traffic has been captured through this server yet.


def _gemini_parts(parts: object) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    texts: list[str] = []
    calls: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for part in parts if isinstance(parts, list) else []:
        if not isinstance(part, Mapping):
            continue
        if isinstance(part.get("text"), str):
            texts.append(part["text"])
        call = part.get("functionCall")
        if isinstance(call, Mapping):
            name = str(call.get("name") or "tool")
            calls.append({"id": str(call.get("id") or f"gemini_{name}_{len(calls)}"), "type": "function",
                          "function": {"name": name, "arguments": json.dumps(call.get("args") or {}, sort_keys=True)}})
        result = part.get("functionResponse")
        if isinstance(result, Mapping):
            name = str(result.get("name") or "tool")
            results.append({"role": "tool", "tool_call_id": str(result.get("id") or f"gemini_{name}"),
                            "content": json.dumps(result.get("response"), sort_keys=True, default=str)})
    return "\n".join(texts), calls, results


def gemini_to_chat(body: Mapping[str, Any]) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    system = clients.system_prompt(body)
    if system:
        messages.append({"role": "system", "content": system})
    for content in body.get("contents") or []:
        if not isinstance(content, Mapping):
            continue
        text, calls, results = _gemini_parts(content.get("parts"))
        if content.get("role") == "model":
            message: dict[str, Any] = {"role": "assistant", "content": text or None}
            if calls:
                message["tool_calls"] = calls
            messages.append(message)
        else:
            messages.extend(results)
            if text:
                messages.append({"role": "user", "content": text})
    out: dict[str, Any] = {"messages": messages}
    if isinstance(body.get("model"), str):
        out["model"] = body["model"]
    return out


def gemini_answer_to_chat(body: object, model: str | None) -> dict[str, Any]:
    candidates = body.get("candidates") if isinstance(body, Mapping) else None
    first = candidates[0] if isinstance(candidates, list) and candidates and isinstance(candidates[0], Mapping) else {}
    content = first.get("content") if isinstance(first.get("content"), Mapping) else {}
    text, calls, _ = _gemini_parts(content.get("parts"))
    message: dict[str, Any] = {"role": "assistant", "content": text or None}
    if calls:
        message["tool_calls"] = calls
    return {"object": "chat.completion", "model": model, "choices": [
        {"index": 0, "message": message, "finish_reason": "tool_calls" if calls else "stop"}]}


def request_for_straiker(event: Event) -> dict[str, Any]:
    body = event.request
    return gemini_to_chat(body) if event.family == "gemini" else provider_body(body)


def answer_as_sse(event: Event, model: str | None) -> str:
    """`data` of a `model_result`, as the `sse` string Straiker parses.

    The reference does not say what Databricks sends for a streamed answer. An object is a
    whole answer and a string is passed through as received; a list is taken as stream chunks
    and written as server-sent events (Anthropic chunks keep their `event:` line)."""
    data = event.data
    if event.family == "gemini":
        if isinstance(data, list):   # streamGenerateContent: merge the chunks' parts
            parts = [p for c in data if isinstance(c, Mapping)
                     for cand in (c.get("candidates") or [])[:1] if isinstance(cand, Mapping)
                     for p in ((cand.get("content") or {}).get("parts") or [])]
            data = {"candidates": [{"content": {"role": "model", "parts": parts}}]}
        return json.dumps(gemini_answer_to_chat(data, model))
    if isinstance(data, str):
        return data
    if isinstance(data, list):
        lines = []
        for chunk in data:
            if isinstance(chunk, Mapping) and event.family == "anthropic.messages" and isinstance(chunk.get("type"), str):
                lines.append(f"event: {chunk['type']}")
            lines.append("data: " + json.dumps(chunk, default=str))
            lines.append("")
        return "\n".join(lines)
    return json.dumps(data, default=str)


# ---------------------------------------------------------------------------------------------
# agent, user, session


@dataclass(frozen=True)
class Route:
    agent: str | None
    client: str | None
    source: str               # policy_config | client:<evidence> | target
    match: clients.ClientMatch | None


def route(event: Event) -> Route:
    """Who the traffic belongs to, highest precedence first.

    1. `policy_config.agent`: the operator names one application. Outranks the body, because a
       body marker is the caller's own claim.
    2. A coding agent recognised from `api_type` or the body. By default it follows the Custom
       Gateway contract: no X-S6r-Agent, X-S6r-Client names the tool, and the platform files the
       traffic under that tool's own agent, as it does every other path the tool arrives by. With
       `name_coding_agents` it is named for this gateway instead (`Claude Code (Databricks)`).
    3. The governed service, `event.target`: one agent per Model Service or MCP Service, the way
       Kong names one agent per route.
    """
    match = None if event.is_mcp else clients.identify(event.request, api_type=event.api_type)
    s = event.settings
    if s.agent:
        return Route(s.agent, None, "policy_config", match)
    if match and s.name_coding_agents:
        return Route(match.agent_name(GATEWAY_NAME), match.vendor, f"client:{match.evidence}", match)
    if match:
        # The Custom Gateway page's coding-client rule: no X-S6r-Agent, so the traffic files under
        # the tenant's own agent for that tool rather than one enumerated under this integration.
        return Route(None, match.vendor, f"client-only:{match.evidence}", match)
    return Route(service_label(event.target), None, "target", match)


#: Measured 2026-11-05 on a live workspace: `event.target` is the service's Unity Catalog ID
#: (`5a3d699d-…`), not the three-level name the reference shows. An ID alone names nothing a
#: person recognises, so it is labelled; `policy_config.agent` is how an operator names the app.
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)


def service_label(target: str) -> str:
    return f"{GATEWAY_NAME} service {target}" if _UUID.match(target) else target


def user_of(event: Event) -> str | None:
    """The end user, from what the body states, because Databricks sends no caller identity.

    What Unity Gateway accepts (measured 2026-11-05): OpenAI Responses `user` and
    `safety_identifier`; Anthropic Messages `metadata.user_id`, but only a UUID or hash (an email
    is rejected); nothing on Chat Completions. Claude Code's `metadata.user_id` is a JSON object
    naming a device, not a person, so it is never read as a user."""
    if event.settings.user:
        return event.settings.user
    body = event.request
    for key in ("user", "safety_identifier"):
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    meta = body.get("metadata")
    stated = meta.get("user_id") if isinstance(meta, Mapping) else None
    if isinstance(stated, str) and stated.strip() and not stated.lstrip().startswith("{"):
        return stated.strip()
    return None


def session_of(event: Event, r: Route) -> str | None:
    """The client's own session, then one the body states, then a derived one.

    The derived seed holds the service, the agent and the user: two of any of those opening with
    the same words are different conversations, and Straiker skips turns it has already scored
    for a session. For a coding agent the system prompt is left out of the seed, because its side
    calls (titles, summaries) carry their own system prompt but belong to the same conversation.
    """
    if event.is_mcp:
        if event.trace_id:
            return f"{DERIVED_SESSION_PREFIX}trace-{event.trace_id}"
        call = event.data if event.type == TOOL_CALL else event.request_data
        seed = f"{event.target}\0{json.dumps(call, sort_keys=True, default=str)}"
        return DERIVED_SESSION_PREFIX + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]
    body = event.request
    stated = clients.stated_session(body, r.match)
    if stated:
        return stated
    explicit = body.get("session_id")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    opening = clients.conversation_opening(body, r.match)
    system = "" if r.match else clients.system_prompt(body)
    if not opening and not system:
        return None
    seed = f"{event.target}\0{r.agent or ''}\0{user_of(event) or ''}\0{system}\0{opening}"
    return DERIVED_SESSION_PREFIX + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]


# ---------------------------------------------------------------------------------------------
# outbound call


def _identity(payload: dict[str, Any], session: str | None, user: str | None) -> dict[str, Any]:
    """`session_id` and `user_name` at the top level, as the Custom Gateway page specifies. The user
    is also sent where the Kong and LiteLLM integrations put it, which the platform reads too."""
    if session:
        payload["session_id"] = session
    if user:
        payload["user_name"] = user
        payload["original"] = {"processed": {"Meta": {"user": user}}}
    return payload


def build_payload(event: Event, session: str | None) -> dict[str, Any]:
    if event.is_mcp:
        return _identity(build_mcp_payload(event), session, user_of(event))
    request = request_for_straiker(event)
    if event.type == MODEL_CALL:
        return _identity(dict(request), session, user_of(event))
    model = None
    if isinstance(event.data, Mapping) and isinstance(event.data.get("model"), str):
        model = event.data["model"]
    model = model or request.get("model") or event.target
    payload = {"straiker_phase": RESPONSE_PHASE, "model": model, "request": request,
               "sse": answer_as_sse(event, model)}
    return _identity(payload, session, user_of(event))


def build_headers(api_key: str, event: Event, r: Route, session: str | None = None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if r.agent:
        headers[AGENT_HEADER] = r.agent
    if r.client:
        headers[CLIENT_HEADER] = r.client
    fmt = event.settings.format_hint or FORMAT_HINTS.get(event.family or "")
    if fmt and not event.is_mcp:
        headers[FORMAT_HEADER] = fmt
    # Every coding client's session rides in this header, whichever client it is: without it the
    # platform scores nothing for a coding-agent body (Custom Gateway page, "Carry the session").
    if r.client and session:
        headers[CLAUDE_SESSION_HEADER] = session
    return headers


# MCP. Databricks names the tool and the service, so a tool call reaches Straiker as a named tool
# call, the way Kong and LiteLLM present one before it runs: the model's answer on the response
# phase. A result is the `tool` message answering that call.

def _tool_call_of(event: Event) -> dict[str, Any]:
    call = event.data if event.type == TOOL_CALL else event.request_data
    call = call if isinstance(call, Mapping) else {}
    name = str(call.get("name") or "tool")
    args = json.dumps(call.get("arguments") or {}, sort_keys=True, default=str)
    call_id = "dbx_mcp_" + hashlib.sha256(f"{event.target}\0{name}\0{args}".encode("utf-8")).hexdigest()[:16]
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}


def tool_result_text(result: object) -> str:
    if isinstance(result, Mapping):
        parts = [str(c.get("text")) for c in result.get("content") or []
                 if isinstance(c, Mapping) and c.get("type") == "text" and c.get("text") is not None]
        if parts:
            return "\n".join(parts)
        if result.get("structuredContent") is not None:
            return json.dumps(result["structuredContent"], sort_keys=True, default=str)
    return result if isinstance(result, str) else json.dumps(result, sort_keys=True, default=str)


def build_mcp_payload(event: Event) -> dict[str, Any]:
    call = _tool_call_of(event)
    if event.type == TOOL_CALL:
        completion = {"object": "chat.completion", "model": event.target, "choices": [{
            "index": 0, "finish_reason": "tool_calls",
            "message": {"role": "assistant", "content": None, "tool_calls": [call]}}]}
        # A second tool-call-only turn in one session is read as a replay unless its input is new
        # (measured on TrueFoundry, 2026-10-05), so the turn carries one labelled marker.
        marker = {"role": "user", "content": f"[Databricks MCP tool call {event.target} {call['id']}]"}
        return {"straiker_phase": RESPONSE_PHASE, "model": event.target,
                "request": {"model": event.target, "messages": [marker]}, "sse": json.dumps(completion)}
    return {"model": event.target, "messages": [
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": call["id"], "content": tool_result_text(event.data)}]}


# ---------------------------------------------------------------------------------------------
# verdict


@dataclass(frozen=True)
class Verdict:
    blocked: bool
    decision: str | None
    reason: str | None
    blocked_by: tuple[str, ...]
    controls: tuple[str, ...]
    turn_id: str | None
    events_scored: int | None


def parse_verdict(body: Mapping[str, Any]) -> Verdict:
    """Decide on `permissionDecision` / `blocked_by`, never on `action` alone: a detect-mode
    control fires into `controls` without changing the decision."""
    nested = body.get("straiker")
    verdict = nested if isinstance(nested, Mapping) else body
    decision = None
    hook = body.get("hookSpecificOutput")
    if isinstance(hook, Mapping) and isinstance(hook.get("permissionDecision"), str):
        decision = hook["permissionDecision"].lower() or None
    if decision is None and isinstance(verdict.get("action"), str):
        decision = verdict["action"].lower() or None
    raw_blocked = verdict.get("blocked_by")
    blocked_by = tuple(sorted(str(c) for c in raw_blocked)) if isinstance(raw_blocked, list) else ()
    raw_controls = verdict.get("controls")
    controls = tuple(sorted(str(c) for c in raw_controls)) if isinstance(raw_controls, list) else ()
    blocked = decision in BLOCK_DECISIONS or bool(blocked_by)
    reason = None
    if blocked:
        hook_reason = hook.get("permissionDecisionReason") if isinstance(hook, Mapping) else None
        for candidate in (verdict.get("block_message"), hook_reason, verdict.get("deny_reason"), body.get("stopReason")):
            if isinstance(candidate, str) and candidate.strip():
                reason = candidate.strip()
                break
        reason = reason or f"Straiker blocked this request: {', '.join(blocked_by) or 'policy'}"
    scored = verdict.get("events_scored")
    return Verdict(
        blocked=blocked,
        decision=decision,
        reason=reason,
        blocked_by=blocked_by,
        controls=controls,
        turn_id=verdict.get("turn_id") if isinstance(verdict.get("turn_id"), str) else None,
        events_scored=scored if isinstance(scored, int) else None,
    )


def _reason(text: str) -> str:
    return text if len(text) <= MAX_REASON_CHARS else text[: MAX_REASON_CHARS - 1] + "…"


def databricks_reply(verdict: Verdict, event: Event) -> dict[str, str]:
    """The external policy verdict. `ASK` only on an MCP tool call, where Databricks can hold the
    call for the user's approval; on any other event it would be a block."""
    if verdict.blocked:
        return {"result": "DENY", "reason": _reason(verdict.reason or "Blocked by Straiker")}
    if event.type == TOOL_CALL and event.settings.ask_on_detect and verdict.controls:
        return {"result": "ASK", "reason": _reason(f"Straiker flagged this tool call: {', '.join(verdict.controls)}")}
    return {"result": "ALLOW"}


def failure_reply(settings: Settings, why: str) -> dict[str, str]:
    """The answer when Straiker could not decide. Databricks would deny any error without telling
    the caller why, so the policy's own fail mode is applied here, explicitly, with a reason."""
    if settings.fail_open:
        return {"result": "ALLOW"}
    return {"result": "DENY", "reason": _reason(f"Straiker guardrail could not decide ({why}). "
                                                 "This policy fails closed.")}


# ---------------------------------------------------------------------------------------------
# replay memory (the TrueFoundry service's)


def conversation_prefixes(body: Mapping[str, Any]) -> tuple[str, ...]:
    """A fingerprint of the conversation after each message, first to last."""
    messages = [m for m in (body.get("messages") if isinstance(body.get("messages"), list)
                            else body.get("input") if isinstance(body.get("input"), list) else [])
                if isinstance(m, Mapping)]
    if messages:
        digest = hashlib.sha256()
        out = []
        for message in messages:
            digest.update(json.dumps(message, sort_keys=True, default=str).encode("utf-8"))
            digest.update(b"\x1e")
            out.append(digest.copy().hexdigest())
        return tuple(out)
    plain = body.get("input") if "input" in body else body.get("prompt") if "prompt" in body else body.get("contents")
    if plain is None:
        return ()
    return (hashlib.sha256(json.dumps(plain, sort_keys=True, default=str).encode("utf-8")).hexdigest(),)


class ReplayMemory:
    """Remembers blocked conversations per session so a resend is blocked again.

    Straiker answers a turn it has already scored for a session with `allow` (`events_replayed`),
    so without this a client could resend a blocked prompt and get it through. Per process."""

    def __init__(self, size: int = 10_000, ttl_s: float = 24 * 3600) -> None:
        self._size, self._ttl = size, ttl_s
        self._store: "OrderedDict[tuple[str, str], tuple[float, str]]" = OrderedDict()

    def remember(self, session: str, body: Mapping[str, Any], reason: str) -> None:
        prefixes = conversation_prefixes(body)
        if not session or not prefixes:
            return
        key = (session, prefixes[-1])
        self._store[key] = (time.monotonic() + self._ttl, reason)
        self._store.move_to_end(key)
        while len(self._store) > self._size:
            self._store.popitem(last=False)

    def blocked_reason(self, session: str | None, body: Mapping[str, Any]) -> str | None:
        if not session:
            return None
        now = time.monotonic()
        for prefix in conversation_prefixes(body):
            hit = self._store.get((session, prefix))
            if hit and hit[0] > now:
                return hit[1]
        return None
