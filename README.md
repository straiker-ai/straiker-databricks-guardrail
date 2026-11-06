# Straiker guardrail for Databricks Unity Gateway

A Databricks App that adds Straiker to Databricks Unity Gateway (AI Gateway) as an **external
service policy**. Unity Gateway sends each request to a governed model service, and each answer,
to the app. The app checks it with Straiker and returns `ALLOW` or `DENY`, which Unity Gateway
enforces.

```
client ──► Unity Gateway ──► model
               │  external service policy (input, output)
               ▼
   Straiker app (your workspace) ──► Straiker  POST /api/v3/detect
```

The app runs in your workspace. Unity Gateway signs in to it as a service principal through the
workspace's own OAuth, and that principal is the only one allowed to call it. The Straiker key is
kept in a Databricks secret.

Setup guide: https://docs.straiker.ai/defend-ai/databricks-unity-gateway-integration

## Requirements

- Unity Catalog, and the **Unity Gateway Service Policies** preview on (**Settings > Previews**)
- Workspace admin rights to create a secret scope, a service principal and an app
- [Databricks CLI](https://docs.databricks.com/aws/en/dev-tools/cli/install) with bundle support
- A Straiker **Custom Gateway** integration key (`sk_agt_…`) from the Straiker Console

## Deploy

```bash
git clone https://github.com/straiker-ai/straiker-databricks-guardrail
cd straiker-databricks-guardrail

# 1. The Straiker key, in a secret the app reads
databricks secrets create-scope straiker
databricks secrets put-secret straiker gateway-key --string-value "<sk_agt_ key>"

# 2. The service principal Unity Gateway signs in as, and its OAuth secret
databricks service-principals create --display-name straiker-unity-gateway-caller
databricks service-principal-secrets-proxy create <id from step 2>

# 3. The app, granting CAN USE to that principal only
databricks bundle deploy --var caller_service_principal=<applicationId from step 2>
databricks bundle run straiker_guardrail
```

`bundle run` prints the app URL. Then create the Unity Catalog connection Unity Gateway uses to
reach the app. It must be stored in a schema, because the policy form lists only schema-level
connections:

```bash
databricks api post /api/2.1/unity-catalog/connections --json '{
  "name": "straiker_guardrail",
  "parent": "schemas/<catalog>.<schema>",
  "connection_type": "HTTP",
  "options": {
    "host": "<app URL>",
    "port": "443",
    "base_path": "/v1/databricks",
    "client_id": "<applicationId from step 2>",
    "client_secret": "<OAuth secret from step 2>",
    "token_endpoint": "https://<workspace host>/oidc/v1/token",
    "oauth_scope": "all-apis"
  }
}'
```

Finally, attach the policy to each model service in the Databricks UI: **AI Gateway > Models >
the service > Policies > New policy > Service policy**, guardrail type **External**, both
phases, **Use existing connection**, and **Evaluate the entire conversation**. Databricks does not
offer an API for this step.

## Policy configuration

The policy's **Policy configuration** field takes a JSON object. All keys are optional.

| Key | Default | Effect |
|---|---|---|
| `agent` | none | Names the application on this service, for example `{"agent": "Support Assistant"}` |
| `user` | none | A fixed user for the service, such as a batch job's service principal |
| `fail_open` | `false` | `true` allows the call when Straiker cannot be reached; `false` denies it with a reason |
| `name_coding_agents` | `false` | `true` names coding agents for this gateway, such as `Claude Code (Databricks)` |

## Operate

- **Logs**: **Compute > Apps > straiker-guardrail > Logs**. One JSON line per call with the
  decision, agent, session and latency. Content and keys are never logged.
- **Rotate the key**: `databricks secrets put-secret straiker gateway-key …`, then
  `databricks bundle run straiker_guardrail`.
- **Availability**: Databricks denies a governed call when the app does not answer within
  5 seconds. If the app is stopped, every call on a service with the policy is denied.
- **Self-hosted Straiker**: set `STRAIKER_DETECT_URL` and `STRAIKER_ALLOW_CUSTOM_DETECT_URL: "1"`
  in `app.yaml`.

## Develop

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/pytest
```

`docs/ENG_HANDOFF.md` records how Unity Gateway behaves on a live workspace, where it differs
from the Databricks reference, and the remaining work.
