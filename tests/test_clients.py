"""Client recognition and stated sessions, against bodies captured from the real clients."""
import json
from pathlib import Path

import pytest

from straiker_databricks import clients

FIXTURES = Path(__file__).parent / "fixtures" / "clients"


def load(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


@pytest.mark.parametrize("name, vendor, evidence, sidecar", [
    ("claude_code_sdk_cli_first", "claude", "billing-line", False),
    ("claude_code_sdk_cli_toolloop", "claude", "billing-line", False),
    ("codex_exec_first", "codex", "preamble", False),
    ("codex_exec_toolloop", "codex", "preamble", False),
    ("opencode_openai_first", "opencode", "preamble", False),
    ("opencode_openai_toolloop", "opencode", "preamble", False),
    ("opencode_anthropic_first", "opencode", "preamble", False),
    ("opencode_openai_title", "opencode", "sidecar", True),
    ("opencode_anthropic_title", "opencode", "sidecar", True),
])
def test_body_alone_names_every_captured_client(name, vendor, evidence, sidecar):
    # Databricks forwards no headers: the body has to be enough.
    match = clients.identify(load(name)["body"])
    assert match is not None
    assert (match.vendor, match.evidence, match.sidecar) == (vendor, evidence, sidecar)


@pytest.mark.parametrize("name, vendor", [
    ("claude_code_sdk_cli_first", "claude"),
    ("codex_exec_first", "codex"),
    ("opencode_openai_first", "opencode"),
    ("opencode_anthropic_first", "opencode"),
])
def test_captured_headers_name_the_client_for_gateways_that_forward_them(name, vendor):
    fixture = load(name)
    match = clients.from_headers(fixture["headers"])
    assert match is not None and match.vendor == vendor and match.evidence == "user-agent"


def test_api_type_outranks_the_body():
    match = clients.identify({"messages": [{"role": "user", "content": "hi"}]}, api_type="cursor/v1/chat/completions")
    assert (match.vendor, match.evidence) == ("cursor", "api_type")
    assert clients.identify(load("codex_exec_first")["body"], api_type="codex/v1/responses").evidence == "api_type"


@pytest.mark.parametrize("ua, vendor, surface", [
    ("ucode/0.9.1 claude/2.1.318 (darwin 24.6.0; arm64)", "claude", "ug"),
    ("ucode/0.9.1 opencode/1.17.11 (linux; x86_64)", "opencode", "ug"),
    ("claude-cli/2.1.312 (external, claude-vscode, agent-sdk/0.2.1)", "claude", "claude-vscode"),
    ("claude-cli/2.1.312 (external, local-agent)", "cowork", "local-agent"),
])
def test_user_agents_of_launchers_and_surfaces(ua, vendor, surface):
    match = clients.from_headers({"User-Agent": ua})
    assert (match.vendor, match.surface) == (vendor, surface)


@pytest.mark.parametrize("entrypoint", ["sdk-py", "sdk-ts"])
def test_agent_sdk_applications_are_not_claude_code(entrypoint):
    # A customer's own Agent SDK app is an agent of its own, by UA and by billing line alike.
    assert clients.from_headers({"user-agent": f"claude-cli/2.1.318 (external, {entrypoint})"}) is None
    body = {"system": [{"type": "text", "text": f"x-anthropic-billing-header: cc_version=2.1.318; cc_entrypoint={entrypoint};"},
                       {"type": "text", "text": "You are a Claude agent, built on Anthropic's Claude Agent SDK."}]}
    assert clients.identify(body) is None


def test_an_application_is_not_a_coding_agent():
    body = {"model": "m", "messages": [{"role": "system", "content": "You are a support agent."},
                                       {"role": "user", "content": "refund me"}]}
    assert clients.identify(body) is None


@pytest.mark.parametrize("name, expected", [
    ("claude_code_sdk_cli_first", "metadata"),
    ("codex_exec_first", "client_metadata"),
    ("opencode_openai_first", "prompt_cache_key"),
])
def test_stated_sessions(name, expected):
    body = load(name)["body"]
    session = clients.stated_session(body, clients.identify(body))
    if expected == "metadata":
        assert session == json.loads(body["metadata"]["user_id"])["session_id"]
    elif expected == "client_metadata":
        assert session == body["client_metadata"]["session_id"] == body["prompt_cache_key"]
    else:
        assert session == body["prompt_cache_key"] and session.startswith("ses_")


def test_prompt_cache_key_is_a_session_only_for_codex_and_opencode():
    # Applications set one fixed cache key for everyone; read as a session it merges all users.
    app_body = {"model": "m", "input": [{"role": "user", "content": "hi"}], "prompt_cache_key": "shared-cache"}
    assert clients.stated_session(app_body, None) is None
    assert clients.stated_session(app_body, clients.identify(app_body)) is None


def test_opencode_title_call_opens_with_the_conversation_it_titles():
    for fmt in ("openai", "anthropic"):
        main = load(f"opencode_{fmt}_first")["body"]
        title = load(f"opencode_{fmt}_title")["body"]
        opening_main = clients.conversation_opening(main, clients.identify(main))
        opening_title = clients.conversation_opening(title, clients.identify(title))
        assert opening_main and opening_main == opening_title
