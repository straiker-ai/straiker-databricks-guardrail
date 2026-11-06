"""Turn record_proxy.py captures into committed fixtures.

Keeps what the tests read: the body's keys and shape, the leading system text (where the client
markers live), tool names, metadata session ids, roles, the test prompt and the tool output.
Replaces every other text with a length marker, because a client's first message carries the
operator's local context (instructions files, paths). Device and installation ids are zeroed.

    python tools/make_fixtures.py captures.jsonl tests/fixtures/clients --workdir <capture dir>
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

KEEP_TEXT = re.compile(r'^"?(Run the shell command|Use the Bash tool|Generate a title for this conversation)|^alpha\s*$')
SYSTEM_KEEP = 200
LOCAL_PATH = re.compile(r"(/private)?/(tmp|var/folders|Users)/[^\s\"\\]*")
ZERO_HEX = "0" * 64
ZERO_UUID = "00000000-0000-4000-8000-000000000000"
NAMES = {0: "claude_code_sdk_cli_first", 3: "claude_code_sdk_cli_toolloop", 4: "codex_exec_first",
         5: "codex_exec_toolloop", 6: "opencode_openai_title", 7: "opencode_openai_first",
         8: "opencode_openai_toolloop", 9: "opencode_anthropic_first", 10: "opencode_anthropic_title"}


def scrub_text(text: str, workdir: str) -> str:
    text = text.replace(workdir, "/work")
    if KEEP_TEXT.search(text.strip()):
        return text
    return f"[scrubbed {len(text)} chars]"


def scrub_content(value: Any, workdir: str) -> Any:
    if isinstance(value, str):
        return scrub_text(value, workdir)
    if isinstance(value, list):
        return [scrub_content(v, workdir) for v in value]
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k in ("text", "content", "output", "arguments", "input") and isinstance(v, (str, list, dict)):
                out[k] = scrub_content(v, workdir) if not (k == "input" and isinstance(v, dict)) else {"command": "cat notes.txt"}
            elif k in ("encrypted_content", "signature"):
                out[k] = "[scrubbed]"
            else:
                out[k] = v
        return out
    return value


def system_head(value: Any) -> Any:
    if isinstance(value, str):
        return value[:SYSTEM_KEEP] + ("…[scrubbed]" if len(value) > SYSTEM_KEEP else "")
    if isinstance(value, list):
        return [dict(b, text=system_head(b.get("text", ""))) if isinstance(b, dict) else b for b in value]
    return value


def scrub_body(body: dict, workdir: str) -> dict:
    out = dict(body)
    if "system" in out:
        out["system"] = system_head(out["system"])
    if "instructions" in out:
        out["instructions"] = system_head(out["instructions"])
    for key in ("messages", "input"):
        if isinstance(out.get(key), list):
            items = []
            for m in out[key]:
                if isinstance(m, dict) and m.get("role") in ("system", "developer"):
                    items.append(dict(m, content=system_head(m.get("content"))))
                else:
                    items.append(scrub_content(m, workdir))
            out[key] = items
    if isinstance(out.get("tools"), list):
        out["tools"] = [{k: v for k, v in t.items() if k in ("name", "type", "function")} if isinstance(t, dict) else t
                        for t in out["tools"]]
        for t in out["tools"]:
            if isinstance(t.get("function"), dict):
                t["function"] = {"name": t["function"].get("name")}
    meta = out.get("metadata")
    if isinstance(meta, dict) and isinstance(meta.get("user_id"), str) and meta["user_id"].startswith("{"):
        parsed = json.loads(meta["user_id"])
        parsed["device_id"] = ZERO_HEX
        out["metadata"] = {"user_id": json.dumps(parsed, separators=(",", ":"))}
    cm = out.get("client_metadata")
    if isinstance(cm, dict):
        cm = {k: (ZERO_UUID if "installation" in k else v) for k, v in cm.items()}
        if isinstance(cm.get("x-codex-turn-metadata"), str):
            turn = json.loads(cm["x-codex-turn-metadata"])
            cm["x-codex-turn-metadata"] = json.dumps({k: (ZERO_UUID if k == "installation_id" else v)
                                                      for k, v in turn.items() if k in ("installation_id", "session_id", "turn_id")})
        out["client_metadata"] = cm
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("captures")
    ap.add_argument("out_dir")
    ap.add_argument("--workdir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, line in enumerate(open(a.captures)):
        if i not in NAMES:
            continue
        rec = json.loads(line)
        h = rec["headers"]
        fixture = {
            "captured": "2026-11-05 via tools/record_proxy.py",
            "provider_path": rec["path"],
            "headers": {k: h[k] for k in ("user-agent", "x-app", "originator", "session-id", "x-session-id",
                                           "x-claude-code-session-id") if k in h},
            "body": scrub_body(rec["body"], a.workdir),
        }
        if NAMES[i] == "claude_code_sdk_cli_toolloop":
            fixture["response_sse"] = rec["response_text"].replace(a.workdir, "/work")
        text = json.dumps(fixture, indent=1).replace(a.workdir, "/work")
        text = LOCAL_PATH.sub("/work", text)
        (out / f"{NAMES[i]}.json").write_text(text + "\n")
        print(NAMES[i])


if __name__ == "__main__":
    main()
