"""Recording pass-through for capturing real coding-agent traffic as test fixtures.

Point a client at http://127.0.0.1:<port>/anthropic or /openai. The request is forwarded to the
provider with the provider key from this process's environment (the client sends a dummy), the
answer is streamed back unchanged, and each exchange is appended to a JSONL file:
{client, path, headers (credentials redacted), body, response_status, response_text}.

    python tools/record_proxy.py 4610 captures.jsonl
"""
from __future__ import annotations

import json
import os
import sys

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

UPSTREAMS = {
    "anthropic": ("https://api.anthropic.com", "x-api-key", "ANTHROPIC_KEY", ""),
    "openai": ("https://api.openai.com", "authorization", "OPENAI_API_KEY", "Bearer "),
}
SECRET = {"authorization", "x-api-key", "api-key", "cookie", "proxy-authorization"}
HOP = {"host", "content-length", "connection", "accept-encoding", "transfer-encoding"}
OUT = sys.argv[2] if len(sys.argv) > 2 else "captures.jsonl"

app = FastAPI()
client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))


def _record(rec: dict) -> None:
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")


@app.api_route("/{provider}/{rest:path}", methods=["GET", "POST"])
async def relay(provider: str, rest: str, request: Request):
    base, key_header, key_env, prefix = UPSTREAMS[provider]
    raw = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP | SECRET}
    headers[key_header] = prefix + os.environ[key_env]
    try:
        body = json.loads(raw) if raw else None
    except ValueError:
        body = raw.decode("utf-8", "replace")
    rec = {
        "provider": provider, "path": "/" + rest, "method": request.method,
        "headers": {k.lower(): ("<redacted>" if k.lower() in SECRET else v) for k, v in request.headers.items()},
        "body": body,
    }
    upstream = client.build_request(request.method, f"{base}/{rest}", params=request.query_params,
                                    headers=headers, content=raw)
    resp = await client.send(upstream, stream=True)
    out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in HOP | {"content-encoding"}}
    if "text/event-stream" not in resp.headers.get("content-type", ""):
        data = await resp.aread()
        rec |= {"response_status": resp.status_code, "response_text": data.decode("utf-8", "replace")}
        _record(rec)
        return Response(data, status_code=resp.status_code, headers=out_headers)

    async def stream():
        chunks = []
        try:
            async for chunk in resp.aiter_raw():
                chunks.append(chunk)
                yield chunk
        finally:
            await resp.aclose()
            rec.update(response_status=resp.status_code,
                       response_text=b"".join(chunks).decode("utf-8", "replace"))
            _record(rec)

    return StreamingResponse(stream(), status_code=resp.status_code, headers=out_headers)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
