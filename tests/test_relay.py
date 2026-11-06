import json

from straiker_databricks.relay import (
    Settings,
    gemini_answer_to_chat,
    gemini_to_chat,
    parse_verdict,
    provider_body,
)


def test_policy_config_defaults_fail_closed():
    s = Settings.from_policy_config(None)
    assert (s.fail_open, s.ask_on_detect, s.name_coding_agents, s.unknown_api_type) == (False, False, False, "relay")


def test_policy_config_values_and_wrong_types():
    s = Settings.from_policy_config(json.dumps({"agent": " Support ", "fail_open": True, "ask_on_detect": "yes",
                                                "unknown_api_type": "explode"}))
    assert s.agent == "Support" and s.fail_open is True
    assert s.ask_on_detect is False          # only a JSON boolean turns it on
    assert s.unknown_api_type == "relay"     # an unknown action falls back to relaying


def test_policy_config_not_json_is_reported():
    assert Settings.from_policy_config("{oops").error


def test_responses_string_input_becomes_a_message():
    assert provider_body({"input": "hi", "prompt_cache_key": "k"}) == {"input": [{"role": "user", "content": "hi"}]}


def test_gemini_request_is_relayed_as_chat():
    body = {"systemInstruction": {"parts": [{"text": "You are helpful."}]},
            "contents": [{"role": "user", "parts": [{"text": "weather in Paris?"}]},
                         {"role": "model", "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}]},
                         {"role": "user", "parts": [{"functionResponse": {"name": "get_weather", "response": {"t": 21}}}]}]}
    msgs = gemini_to_chat(body)["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool"]
    assert msgs[2]["tool_calls"][0]["function"] == {"name": "get_weather", "arguments": '{"city": "Paris"}'}
    assert msgs[3]["content"] == '{"t": 21}'


def test_gemini_answer_is_relayed_as_a_chat_completion():
    answer = {"candidates": [{"content": {"role": "model", "parts": [{"text": "21C"}]}}]}
    assert gemini_answer_to_chat(answer, "g")["choices"][0]["message"]["content"] == "21C"


def test_detect_mode_controls_do_not_block():
    v = parse_verdict({"hookSpecificOutput": {"permissionDecision": "allow"},
                       "straiker": {"action": "detect", "controls": ["llm_evasion"], "blocked_by": []}})
    assert not v.blocked and v.controls == ("llm_evasion",)


def test_blocked_by_blocks_even_when_the_decision_says_allow():
    v = parse_verdict({"hookSpecificOutput": {"permissionDecision": "allow"},
                       "straiker": {"blocked_by": ["llm_evasion"]}})
    assert v.blocked and "llm_evasion" in v.reason
