# Databricks Unity Gateway guardrail: engineering handoff

Prototype status, 2026-11-05: working end to end on a live Databricks workspace. Real Claude Code
and OpenCode traffic went through Unity Gateway, was checked by Straiker on input and output, and
Databricks enforced the verdicts. The final run used the Databricks App deployment below, with
nothing running outside Databricks and Straiker.

## What Databricks offers

Unity Gateway is a managed gateway. There is no plugin runtime like Kong's or LiteLLM's. A
partner guardrail runs through an **external service policy** (Beta): on each governed call,
Databricks posts the content to an HTTPS endpoint and enforces the verdict.

- Custom policy functions run inside Databricks, but Databricks transpiles their SQL to CEL, so they
  cannot make network calls. They cannot call Straiker.
- Attaching a policy is UI-only (no REST, Terraform or SDK type). Connections, model services,
  secrets and apps all have APIs.

## The contract, as measured (where it differs from the reference)

Reference: https://docs.databricks.com/aws/en/data-governance/unity-catalog/service-policies/external-policy-api

| | Reference says | Measured |
|---|---|---|
| Path | Connection host + API path | Databricks posts to `<API path>/` (trailing slash) and does **not follow redirects**: a 307 fails every call closed |
| `event.target` | Three-level UC name | The service's UC **ID** (`5a3d699d-…`). Map it, or have the operator name the app in `policy_config` |
| `config` | `policy_config`, `phases` | Also `timeout_ms: "5000"`, `on_error: "DENY"`, `dry_run: "false"` (Log mode) |
| Streamed answers | Not specified | `model_result.data` is one assembled Messages object, not chunks |
| `trace_id` | Optional | Present on every call seen |
| Identity | None | None. No user, no client headers, no User-Agent. Databricks' own SQL policies do get `event:context.actor` and request tags |
| Deadline | 5 s, fail-closed | Confirmed. Any non-200, bad JSON or unknown `result` is a DENY |
| Block shown to client | HTTP 200 + `databricks_service_policy` | Claude Code shows only "can't help with this … Message ID: databricks-guardrail-block" |

Captured envelopes: `tests/fixtures/databricks/` (scrubbed), with tests in
`tests/test_databricks_captures.py`.

## Straiker's API today

Measured against `api.prod.straiker.ai` with a Custom Gateway `sk_agt_` key:

- `POST /auth/token` with `grant_type=client_credentials` and the key as secret: **400 invalid_client**
- `POST /api/v3/detect` with a Databricks envelope: **501** ("'gateway' traffic naming no agent…")

So Databricks cannot call the v3 API directly yet. The Databricks App bridges both gaps.

## Path 1: Databricks App (built, enterprise-deployable today)

The guardrail runs as a Databricks App inside the customer's workspace, deployed with a Databricks
Asset Bundle. This fits customers that want nothing outside their Databricks.

```
client ─► Unity Gateway ─► external policy ─► Straiker App (in the workspace) ─► api.prod.straiker.ai/api/v3/detect
```

- `databricks.yml`: the app, its secret resource (the Straiker key) and `CAN USE` for one caller
  service principal. `app.yaml`: runtime. `STRAIKER_API_KEY` from the secret puts the server in
  Databricks App mode: no token endpoint, no token secret.
- Authentication: the UC connection uses OAuth M2M against the workspace's own
  `https://<workspace>/oidc/v1/token` with the caller service principal, scope `all-apis`. The
  Apps proxy authenticates every request before it reaches the app. Measured: unauthenticated and
  forged-token requests to the policy path get a 302 to sign-in; the principal's token gets through.
- Setup, all by CLI or API except the last step:
  1. `databricks secrets create-scope straiker` and `put-secret straiker gateway-key`
  2. `databricks service-principals create`, then `service-principal-secrets-proxy create`
  3. `databricks bundle deploy --var caller_service_principal=<application id>` and `bundle run straiker_guardrail`
  4. UC HTTP connection: host = app URL, API path `/v1/databricks`, client = the principal,
     token endpoint = workspace `/oidc/v1/token`, scope `all-apis`
  5. Attach the External policy to each model service (UI)
- Open items: App high availability (`compute_min_instances` is private preview), App log access
  needs OAuth U2M (`databricks apps logs` refuses a PAT), and packaging the bundle for customers.

## Path 2: native in Straiker's v3 API (the product)

Same as Portkey and TrueFoundry, whose webhook formats argus already reads. The customer then
points the connection at `api.prod.straiker.ai` and hosts nothing.

1. **argus**: a `databricks_wire.py` beside `truefoundry_wire.py` (request half = `event.data` on
   `model_call`, `event.request_data` on `model_result`; answer = `event.data`; MCP `tool_call` /
   `tool_result`), recognised in `webhook_arrival.py`, and a reply dialect in `api/v3/contracts.py`
   returning HTTP 200 `{"result": "ALLOW"|"DENY", "reason"}`. Serve the trailing-slash path.
2. **Auth service** behind `/auth/token`: a client-credentials grant whose secret is an `sk_agt_`
   key, returning a token `/api/v3/detect` accepts.
3. **Console catalog**: a Databricks gateway type, and a tile that shows the connection values.

`straiker_databricks/relay.py` is the reference for the mapping, including client recognition
from the body (Databricks forwards no headers) and session handling.

## Attribution

Coding clients follow the Custom Gateway contract: `X-S6r-Client` and `X-Claude-Code-Session-Id`,
no `X-S6r-Agent`, so they file under the tenant's coding agents (`claude`, `opencode`, `codex`).
Recognition is from the body: Claude Code's `cc_entrypoint` billing line, OpenCode's and Codex's
system prompts, `api_type` (`codex/v1/responses`, `cursor/v1/chat/completions`). Applications
are named by `policy_config.agent`. Customers asked for agents listed under the gateway; that is
`"name_coding_agents": true` here and a platform decision for Path 2.

## Caller identity

Databricks does not send the caller to an external guardrail: no user, no client headers, no
request tags. Inside the app, `X-Forwarded-Email` is the service principal Unity Gateway signs in
as. Databricks computes the caller for its own SQL policies (`event:context.actor.run_as`), so
the fix is a Databricks change: include `context.actor` and `request_tags` in the external event.
Until then: `policy_config.user` for a service with one principal, or join the app's `trace_id`
to `enduser.id` in Databricks' unified trace table for reporting.

## Test environment

The prototype was verified on a Databricks AWS workspace with a schema holding three model
services (Claude, GPT and an application), the External policy on the Claude service, the app
deployed with this bundle, and a Straiker Custom Gateway integration. Run the suite with
`pip install -r requirements.txt -r requirements-dev.txt && pytest`.
