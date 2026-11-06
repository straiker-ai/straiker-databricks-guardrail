"""Straiker guardrail for Databricks Unity Gateway, deployed as a Databricks App.

Unity Gateway posts every governed model call, model answer, MCP tool call and MCP tool result
here (an external service policy). The app relays it to Straiker's ``POST /api/v3/detect`` with
the contract the Kong, LiteLLM and TrueFoundry integrations use, and answers ``ALLOW``, ``DENY``
or ``ASK`` in the format Databricks requires.

The app runs inside the customer's workspace. The Databricks Apps proxy authenticates every
request before it reaches this code: the Unity Gateway connection signs in as a service principal
through the workspace's own OAuth endpoint, and only that principal has CAN USE on the app. The
Straiker key comes from the app's secret resource and leaves the workspace only to Straiker.

Endpoints:
  POST /v1/databricks      the policy endpoint (also POST / and the trailing-slash form)
  GET  /healthz, /debug/loaded-config

Environment (set in app.yaml):
  STRAIKER_API_KEY                required. The Custom Gateway key, from the app's secret resource
  STRAIKER_DETECT_URL             default https://api.prod.straiker.ai/api/v3/detect
  STRAIKER_ALLOW_CUSTOM_DETECT_URL  "1" to allow a non-*.straiker.ai https host (self-hosted Straiker)
  STRAIKER_TIMEOUT_S              default 4.0, capped at 4.5: Databricks denies after 5 s, so the app
                                  answers first, with the policy's own fail mode
  STRAIKER_MAX_BODY_BYTES         default 16777216
  STRAIKER_LOG_LEVEL              default INFO
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import __version__
from .relay import (
    MODEL_CALL,
    MODEL_RESULT,
    GuardrailError,
    ReplayMemory,
    Settings,
    Verdict,
    build_headers,
    build_payload,
    databricks_reply,
    failure_reply,
    parse_event,
    parse_verdict,
    route,
    session_of,
    settings_of,
)

log = logging.getLogger("straiker_databricks")
DEFAULT_DETECT_URL = "https://api.prod.straiker.ai/api/v3/detect"
MAX_TIMEOUT_S = 4.5


def detect_url() -> str:
    url = os.environ.get("STRAIKER_DETECT_URL", DEFAULT_DETECT_URL).strip()
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise RuntimeError(f"STRAIKER_DETECT_URL must be an absolute https URL: {url!r}")
    if not parsed.hostname.endswith(".straiker.ai") and os.environ.get("STRAIKER_ALLOW_CUSTOM_DETECT_URL") != "1":
        # The relayed key is a live Straiker key; never send it to another host by accident.
        raise RuntimeError(f"STRAIKER_DETECT_URL host must be *.straiker.ai: {parsed.hostname}")
    return url


def api_key() -> str:
    key = os.environ.get("STRAIKER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("STRAIKER_API_KEY is not set: add the app's secret resource (see databricks.yml)")
    return key


TIMEOUT_S = min(float(os.environ.get("STRAIKER_TIMEOUT_S", "4.0")), MAX_TIMEOUT_S)
MAX_BODY = int(os.environ.get("STRAIKER_MAX_BODY_BYTES", str(16 * 1024 * 1024)))
REPLAYS = ReplayMemory()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=os.environ.get("STRAIKER_LOG_LEVEL", "INFO"), format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    app.state.detect_url = detect_url()
    app.state.api_key = getattr(app.state, "api_key", None) or api_key()
    app.state.client = getattr(app.state, "client", None) or httpx.AsyncClient(
        timeout=httpx.Timeout(TIMEOUT_S), limits=httpx.Limits(max_connections=200, max_keepalive_connections=50)
    )
    yield
    await app.state.client.aclose()


# No slash redirects: Databricks posts to `<API path>/` and does not follow a redirect, so a 307
# here fails closed (measured 2026-11-05: every governed call denied, none reached this server).
app = FastAPI(title="Straiker guardrail for Databricks Unity Gateway", version=__version__, lifespan=lifespan,
              redirect_slashes=False)


def _log(**fields: Any) -> None:
    fields = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields}
    log.info(json.dumps({k: v for k, v in fields.items() if v is not None}, default=str))


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {"ok": True, "version": __version__}


@app.get("/debug/loaded-config")
async def loaded_config(request: Request) -> dict[str, Any]:
    """What this deployment runs with. Never the key."""
    return {
        "version": __version__,
        "detect_url": request.app.state.detect_url,
        "timeout_s": TIMEOUT_S,
        "max_body_bytes": MAX_BODY,
        "policy_config_defaults": Settings().__dict__ | {"error": None},
    }


def _reply(body: dict[str, str]) -> JSONResponse:
    # Always 200: Databricks shows the caller our reason only on a 200 DENY.
    return JSONResponse(status_code=200, content=body)


@app.post("/")
@app.post("/v1/databricks")
@app.post("/v1/databricks/")
async def policy(request: Request) -> JSONResponse:
    started = time.monotonic()
    raw = await request.body()
    try:
        body = json.loads(raw or b"null")
    except ValueError:
        body = None
    settings = settings_of(body)
    if len(raw) > MAX_BODY:
        _log(event="error", error="payload too large", bytes=len(raw), ms=_ms(started))
        return _reply(failure_reply(settings, f"payload of {len(raw)} bytes exceeds the inspection limit"))
    try:
        event = parse_event(body)
    except GuardrailError as exc:
        _log(event="error", error=exc.message, ms=_ms(started))
        return _reply(failure_reply(settings, exc.message))

    r = route(event)
    session = session_of(event, r)
    common = dict(caller=request.headers.get("x-forwarded-email") or request.headers.get("x-forwarded-user"),
                  type=event.type, api_type=event.api_type, target=event.target, trace_id=event.trace_id,
                  agent=r.agent, agent_source=r.source, client=r.client,
                  sidecar=(r.match.sidecar or None) if r.match else None, session=session,
                  policy_config_error=event.settings.error)

    if event.api_type and event.family is None and not event.is_mcp:
        action = event.settings.unknown_api_type
        if action != "relay":
            reply = {"result": "ALLOW"} if action == "allow" else {
                "result": "DENY", "reason": f"api_type {event.api_type} is not inspected by this policy"}
            _log(event="verdict", **common, result=reply["result"], unscored="unknown api_type", ms=_ms(started))
            return _reply(reply)
    if event.type == MODEL_RESULT and event.family == "embeddings":
        # The answer is vectors: nothing to inspect. The input was inspected on model_call.
        _log(event="verdict", **common, result="ALLOW", unscored="embedding vectors", ms=_ms(started))
        return _reply({"result": "ALLOW"})

    # A resend of a conversation already blocked in this session: the platform answers a replay
    # with allow, so it is blocked here with the original reason.
    if event.type == MODEL_CALL:
        replayed = REPLAYS.blocked_reason(session, event.request)
        if replayed:
            _log(event="verdict", **common, result="DENY", replayed=True, ms=_ms(started))
            return _reply({"result": "DENY", "reason": replayed})

    payload = build_payload(event, session)
    headers = build_headers(request.app.state.api_key, event, r, session)
    try:
        resp = await request.app.state.client.post(request.app.state.detect_url, json=payload, headers=headers)
    except httpx.TimeoutException:
        _log(event="error", **common, error="straiker timeout", fail_open=event.settings.fail_open, ms=_ms(started))
        return _reply(failure_reply(event.settings, f"no answer within {TIMEOUT_S:g} s"))
    except httpx.HTTPError as exc:
        _log(event="error", **common, error=type(exc).__name__, fail_open=event.settings.fail_open, ms=_ms(started))
        return _reply(failure_reply(event.settings, f"unreachable: {type(exc).__name__}"))
    if resp.status_code != 200:
        _log(event="error", **common, straiker_status=resp.status_code, detail=resp.text[:300],
             fail_open=event.settings.fail_open, ms=_ms(started))
        return _reply(failure_reply(event.settings, f"HTTP {resp.status_code}"))
    try:
        verdict: Verdict = parse_verdict(resp.json())
    except (ValueError, AttributeError):
        _log(event="error", **common, error="straiker answer not JSON", ms=_ms(started))
        return _reply(failure_reply(event.settings, "unreadable answer"))

    # Remember content blocks only. A block that names no control is the kill switch, which the
    # platform keeps answering itself while it is engaged.
    if event.type == MODEL_CALL and verdict.blocked and session and (verdict.blocked_by or verdict.controls):
        REPLAYS.remember(session, event.request, verdict.reason or "blocked")
    reply = databricks_reply(verdict, event)
    _log(event="verdict", **common, result=reply["result"], decision=verdict.decision,
         blocked_by=list(verdict.blocked_by) or None, controls=list(verdict.controls) or None,
         turn_id=verdict.turn_id, events_scored=verdict.events_scored, ms=_ms(started))
    return _reply(reply)
