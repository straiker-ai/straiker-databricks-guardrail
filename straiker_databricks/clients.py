"""Which coding agent sent a request: one catalog for every gateway Straiker sits behind.

A gateway that forwards client headers (Kong, LiteLLM) can read the User-Agent. Databricks Unity
Gateway does not: an external service policy receives the body and `api_type` only. So the
catalog reads every signal that can arrive, strongest first, and reports which one decided:

1. headers, when present: User-Agent, Codex's `originator`, Databricks' own launcher (`ug`,
   which sends `User-Agent: ucode/<v> <agent>/<v> (...)`)
2. `api_type`, when the gateway states it: Databricks routes Codex to `codex/v1/responses` and
   Cursor to `cursor/v1/chat/completions`
3. the body: Claude Code's billing line, the system-prompt markers below, OpenCode's title
   side-call

Every marker here was captured from the client itself. Vendors are the platform's vocabulary
(`claude`, `codex`, `opencode`, `cursor`, `cowork`), the values `x-s6r-client` takes.

Captures (`tests/fixtures/clients/`, recorded 2026-11-05 through `tools/record_proxy.py`):
  Claude Code 2.1.318 `claude -p`  UA `claude-cli/2.1.318 (external, sdk-cli)`;
      system opens `x-anthropic-billing-header: cc_version=…; cc_entrypoint=sdk-cli;` then
      "You are a Claude agent, built on Anthropic's Claude Agent SDK."; `metadata.user_id` is a
      JSON string holding `session_id`
  Codex 0.157.1 `codex exec`  UA `codex_exec/0.157.1 (…)`, `originator: codex_exec`;
      `instructions` opens "You are a coding agent running in the Codex CLI";
      `prompt_cache_key` and `client_metadata.session_id` are the session
  OpenCode 1.17.11  UA `opencode/1.17.11 …`; OpenAI-format system "You are OpenCode, You and
      the user share…", Anthropic-format "You are OpenCode, the best coding agent on the
      planet."; `prompt_cache_key` = its `ses_…` id on OpenAI-format main calls only; the title
      side-call opens "You are a title generator. You output ONLY a thread title."
Earlier captures (argus `ingress/hints.py`, 2026-09/10): entrypoints `cli`, `claude-vscode`,
`claude-desktop-3p` are Claude Code, `local-agent` is Cowork, `sdk-py`/`sdk-ts` are customers'
own Agent SDK applications and are deliberately NOT named.

⚠️ A body marker is a claim the caller makes. Naming a request as a coding agent changes which
controls Straiker applies, so an operator's pin (`policy_config.agent`) always outranks this
module, and nothing here ever decides whether a request is inspected.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Final, Mapping

#: How far into the system prompt a marker may sit. Claude Code's billing line comes first and
#: is short; every product sentence below was captured at offset 0 of its own block.
PREAMBLE_WINDOW: Final = 600

PRODUCTS: Final[Mapping[str, str]] = {
    "claude": "Claude Code",
    "codex": "Codex",
    "opencode": "OpenCode",
    "cursor": "Cursor",
    "cowork": "Claude Cowork",
}

#: `cc_entrypoint` values that are Claude Code itself. `sdk-py` and `sdk-ts` are applications a
#: customer built on the Agent SDK: they are agents of their own, not Claude Code.
CLAUDE_CODE_ENTRYPOINTS: Final = frozenset({"cli", "sdk-cli", "claude-vscode", "claude-desktop-3p"})
COWORK_ENTRYPOINTS: Final = frozenset({"local-agent"})

#: Lowercase substrings of the leading system prompt.
PREAMBLE_MARKERS: Final[tuple[tuple[str, str], ...]] = (
    ("you are claude code", "claude"),
    ("claude is operating as an agent inside the claude desktop app", "cowork"),
    ("you are a coding agent running in the codex cli", "codex"),
    ("you are codex, an agent based on", "codex"),
    ("you are opencode", "opencode"),
    ("you are an ai agent running in opencode", "opencode"),
)

#: Side calls a client makes besides the conversation itself. Named for their client so they
#: land on its agent, never used to skip inspection: the caller writes the system prompt.
SIDECAR_MARKERS: Final[tuple[tuple[str, str], ...]] = (
    ("you are a title generator. you output only a thread title.", "opencode"),
)
#: OpenCode's title side-call wraps the conversation's first prompt in this line.
TITLE_REQUEST_PREFIX: Final = "Generate a title for this conversation:"

#: `api_type` values that name their client, as Databricks Unity Gateway sends them.
API_TYPE_CLIENTS: Final[Mapping[str, str]] = {
    "codex/v1/responses": "codex",
    "cursor/v1/chat/completions": "cursor",
}

_ENTRYPOINT = re.compile(r"cc_entrypoint=([a-z0-9-]+)")
_CLAUDE_UA = re.compile(r"^claude-cli/[^\s]+ \(external, ([a-z0-9-]+)")
_UCODE_AGENT = re.compile(r"^ucode/\S+ ([a-z-]+)/")
_UCODE_AGENTS: Final[Mapping[str, str]] = {"claude": "claude", "codex": "codex", "opencode": "opencode"}


@dataclass(frozen=True)
class ClientMatch:
    vendor: str            # x-s6r-client value
    product: str           # display name, "Claude Code"
    evidence: str          # which signal decided, for logs: "user-agent", "api_type", "billing-line", …
    surface: str | None = None   # Claude Code entrypoint or launcher, when known
    sidecar: bool = False

    def agent_name(self, gateway: str) -> str:
        return f"{self.product} ({gateway})"


def _match(vendor: str, evidence: str, surface: str | None = None, sidecar: bool = False) -> ClientMatch:
    return ClientMatch(vendor, PRODUCTS[vendor], evidence, surface, sidecar)


# ---------------------------------------------------------------------------------------------
# body text


def _block_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_block_text(v) for v in value)
    if isinstance(value, Mapping):
        for key in ("text", "content"):
            if key in value:
                return _block_text(value[key])
    return ""


def _conversation(body: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("messages", "input"):
        items = body.get(key)
        if isinstance(items, list):
            return [m for m in items if isinstance(m, Mapping)]
    return []


def system_prompt(body: Mapping[str, Any]) -> str:
    """The leading system text in any of the formats a gateway relays: Anthropic `system`,
    Responses `instructions`, a leading system/developer message, Gemini `systemInstruction`."""
    for key in ("system", "instructions"):
        if body.get(key):
            return _block_text(body[key])
    gemini = body.get("systemInstruction") or body.get("system_instruction")
    if isinstance(gemini, Mapping):
        return "\n".join(_block_text(p) for p in gemini.get("parts") or [])
    lead = next(iter(_conversation(body)), None)
    if lead is not None and lead.get("role") in ("system", "developer"):
        return _block_text(lead.get("content"))
    return ""


def user_texts(body: Mapping[str, Any]) -> list[str]:
    return [_block_text(m.get("content")) for m in _conversation(body) if m.get("role") == "user"]


def conversation_opening(body: Mapping[str, Any], match: ClientMatch | None) -> str:
    """The text that opens the conversation, for deriving a session when the client states none.

    OpenCode's title side-call carries the first prompt wrapped in a request for a title; it is
    unwrapped here so the side-call and the conversation it titles derive the same session."""
    texts = user_texts(body)
    if match and match.sidecar and match.vendor == "opencode":
        joined = "\n".join(texts)
        if joined.startswith(TITLE_REQUEST_PREFIX):
            return joined[len(TITLE_REQUEST_PREFIX):].strip()
    return texts[0].strip() if texts else ""


# ---------------------------------------------------------------------------------------------
# identification


def from_headers(headers: Mapping[str, str] | None) -> ClientMatch | None:
    """For gateways that forward client headers. Keys are compared case-insensitively."""
    if not headers:
        return None
    h = {str(k).lower(): str(v) for k, v in headers.items()}
    ua = h.get("user-agent", "")
    if m := _UCODE_AGENT.match(ua):
        vendor = _UCODE_AGENTS.get(m.group(1))
        if vendor:
            return _match(vendor, "user-agent", surface="ug")
    if m := _CLAUDE_UA.match(ua):
        entrypoint = m.group(1)
        if entrypoint in CLAUDE_CODE_ENTRYPOINTS:
            return _match("claude", "user-agent", surface=entrypoint)
        if entrypoint in COWORK_ENTRYPOINTS:
            return _match("cowork", "user-agent", surface=entrypoint)
        return None   # an Agent SDK application: the customer's own agent
    originator = h.get("originator", "")
    if ua.startswith("codex_") or originator.startswith("codex_"):
        return _match("codex", "user-agent", surface=(originator or ua.split("/", 1)[0]))
    if ua.startswith("opencode/") or originator == "opencode":
        return _match("opencode", "user-agent")
    return None


def from_body(body: Mapping[str, Any], api_type: str | None = None) -> ClientMatch | None:
    if api_type and (vendor := API_TYPE_CLIENTS.get(api_type)):
        return _match(vendor, "api_type")
    head = system_prompt(body)[:PREAMBLE_WINDOW]
    lowered = head.lower()
    if m := _ENTRYPOINT.search(lowered):
        entrypoint = m.group(1)
        if entrypoint in CLAUDE_CODE_ENTRYPOINTS:
            return _match("claude", "billing-line", surface=entrypoint)
        if entrypoint in COWORK_ENTRYPOINTS:
            return _match("cowork", "billing-line", surface=entrypoint)
    for marker, vendor in PREAMBLE_MARKERS:
        if marker in lowered:
            return _match(vendor, "preamble")
    for marker, vendor in SIDECAR_MARKERS:
        if marker in lowered:
            return _match(vendor, "sidecar", sidecar=True)
    return None


def identify(body: Mapping[str, Any], *, headers: Mapping[str, str] | None = None,
             api_type: str | None = None) -> ClientMatch | None:
    """The client, strongest signal first. None for anything that is not a known coding agent,
    which is every application a customer built."""
    return from_headers(headers) or from_body(body, api_type)


# ---------------------------------------------------------------------------------------------
# sessions a client states in its own body


#: Clients whose `prompt_cache_key` is their conversation id. An application usually sets one
#: fixed key for everyone to share a cache, so for anything else it would merge every user's
#: conversations into one session.
CACHE_KEY_SESSIONS: Final = frozenset({"codex", "opencode"})


def stated_session(body: Mapping[str, Any], match: ClientMatch | None) -> str | None:
    """A conversation id the client put in the body.

    Claude Code: `metadata.user_id`, a JSON string with `session_id`. Codex:
    `client_metadata.session_id`. Codex and OpenCode only: `prompt_cache_key` (OpenCode's is its
    `ses_…` id; Codex's equals its session id)."""
    meta = body.get("metadata")
    raw = meta.get("user_id") if isinstance(meta, Mapping) else None
    if isinstance(raw, str) and raw.startswith("{"):
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = None
        session = parsed.get("session_id") if isinstance(parsed, Mapping) else None
        if isinstance(session, str) and session:
            return session
    client_meta = body.get("client_metadata")
    if isinstance(client_meta, Mapping) and isinstance(client_meta.get("session_id"), str) and client_meta["session_id"]:
        return client_meta["session_id"]
    key = body.get("prompt_cache_key")
    if match and match.vendor in CACHE_KEY_SESSIONS and isinstance(key, str) and key.strip():
        return key.strip()
    return None
