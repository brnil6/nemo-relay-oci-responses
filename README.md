# Agent observability on the OCI Responses API with NeMo Relay

A small tool-calling agent built on the [OCI Generative AI Responses API](https://docs.oracle.com/en-us/iaas/Content/generative-ai/responses-api.htm),
using the stock OpenAI SDK, with [NVIDIA NeMo Relay](https://github.com/NVIDIA/NeMo-Relay) recording and
governing every model call and tool call. No agent framework is involved. The run is exported as a
portable ATIF trajectory, with PII redaction, a blocking guardrail, and per-call cost accounting
configured outside the agent code.

Inspired by the Oracle blog post
[NVIDIA NeMo Relay now natively supports Oracle Generative AI](https://blogs.oracle.com/ai-and-datascience/nemo-relay-oracle-genai),
which shows the same pattern for the OCI Chat API through LangChain middleware. This sample takes
the other OCI inference surface, the Responses API, and wires Relay in through its plain API.

## How it fits together

The OCI Responses API uses the same wire format as the OpenAI Responses API and is served at
`/openai/v1`, so the application talks to it with the OpenAI SDK. NeMo Relay ships a built-in
[`OpenAIResponsesCodec`](https://docs.nvidia.com/nemo/relay/reference/api/rust-library-reference/nemo-relay/codec/openai_responses)
that understands that format, so Relay reads the payloads the application already sends.
Each model call is passed through `nemo_relay.llm.execute` with that codec, and each tool call
through `nemo_relay.tools.execute`. Everything else, from redaction and guardrails to cost and
export, is generic Relay configuration and would look the same for any provider.

OCI exposes three inference surfaces, and Relay has a codec for each:

| OCI surface | Wire format | Relay codec |
| --- | --- | --- |
| Native Generative AI Chat API | Oracle's own (GENERIC, COHERE, COHEREV2) | `oci_genai` |
| OpenAI-compatible Chat Completions | OpenAI | `openai_chat` |
| Responses API | OpenAI | `openai_responses` (this sample) |

## What the sample does

`oci_responses_relay.py`: a tool-calling agent on the OCI Responses API using the OpenAI SDK, with no agent framework.
Every model call goes through `nemo_relay.llm.execute` with `OpenAIResponsesCodec`; every tool call goes
through `nemo_relay.tools.execute`. Two read-only OCI tools query the live tenancy (managed model catalog,
GPU service limits per availability domain). Governance configured outside the agent:

| Check | Result |
| --- | --- |
| Guardrail (conditional execution) | A request for a "production password" was rejected before OCI was called. OCI calls made: 0. |
| Trajectory | 4 model turns, 3 tool calls, exported as a 5-step ATIF trajectory with per-step model and token usage. |
| PII redaction | `jane.doe@example.com` in the prompt is recorded as `j*******************` in every export. The original string appears nowhere. The model received it unmodified. |
| Cost | Each LLM event carries `usage.cost` priced from an inline catalog (`pricing_provider: oci`). Rates are illustrative placeholders. |
| Streaming | `oci_responses_relay_stream.py`: 86 server-sent events relayed through `llm.stream_execute`, final response decoded by the same codec, 2-step trajectory. |

Runs on 2026-09-25 against managed OCI Generative AI in `us-chicago-1`, model `openai.gpt-oss-120b`,
`nemo-relay` 0.9.2, `openai` 3.19.2. Captured output is in `results/`.

## Findings worth knowing

- **No project OCID was needed.** The OCI Generative AI API key alone authenticated against
  `/openai/v1/responses` in both `us-chicago-1` and `eu-frankfurt-1`.
- **OCI rejects its own reasoning items on replay.** Echoing `gpt-oss` reasoning output items back as
  input fails with `reasoning_text is not supported by model openai.gpt-oss-120b`. The agent loop drops
  reasoning items from history. This is OCI API behaviour, independent of Relay.
- **OCI-specific request fields pass through untouched.** `conversation`, `background`, `reasoning`, and
  hosted tool definitions (`mcp`, `file_search`, `code_interpreter`) survive codec decode/encode byte-for-byte.
- **Hosted tool *outputs* are recorded, not normalized.** `mcp_call` and `file_search_call` output items
  land in `api_specific.output_items` rather than in the normalized `tool_calls` list. Function calls are
  fully normalized. Hosted tools were not exercised live (no MCP server or vector store in this tenancy).
- **Cost lives on events, not in ATIF.** `usage.cost` is present on each LLM scope event (what OpenTelemetry
  and OpenInference export see). ATIF `final_metrics` carries token totals only.
- **Mask shape.** Other Relay examples show `j*******@example.com`; with `detector="email"`,
  `action="mask"`, `unmasked_prefix=1` the whole address after the first character is masked.

## Run it

```bash
python3 -m venv venv && source venv/bin/activate
pip install nemo-relay openai

export OCI_COMPARTMENT_ID="ocid1.compartment.oc1..."      # for the catalog tool
export OCI_REGION="us-chicago-1"
export OCI_PROFILE="CHICAGO"                               # profile in ~/.oci/config (for the CLI tools)
export OCI_GENAI_API_KEY_FILE="$HOME/.oci/genai_api_key_chicago"
# export RESPONSES_MODEL="openai.gpt-oss-120b"

python oci_responses_relay.py          # agent + guardrail + redaction + cost + ATIF
python oci_responses_relay_stream.py   # streaming variant
```

## Files

| File | Purpose |
| --- | --- |
| `oci_responses_relay.py` | The agent, its two OCI tools, the Relay boundary around the OpenAI SDK, and the governance policy. |
| `oci_responses_relay_stream.py` | Streaming call through `llm.stream_execute`. |
| `results/oci-responses-trajectory.json` | ATIF trajectory of the advisor run. |
| `results/oci-responses-events.json` | Raw Relay lifecycle events, including redacted payloads and `usage.cost`. |
| `results/oci-responses-stream-trajectory.json` | ATIF trajectory of the streaming run. |
| `results/sample-run.txt`, `results/sample-run-stream.txt` | Console transcripts. |

Availability-domain prefixes in `results/` were replaced with `EXAMPLE:`; nothing else was edited.
