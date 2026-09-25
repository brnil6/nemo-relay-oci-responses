"""OCI Responses API agent observed and governed by NVIDIA NeMo Relay.

A tool-calling agent on the OCI Generative AI Responses API (OpenAI-compatible, served at
/openai/v1) using the stock OpenAI SDK. NeMo Relay records and governs the run through its
built-in ``OpenAIResponsesCodec``; no agent framework is involved.

The agent has two read-only tools that query the live tenancy (managed model catalog, GPU
service limits). Every model call goes through ``nemo_relay.llm.execute`` and every tool call
through ``nemo_relay.tools.execute``.

Governance exercised in this run:
  * PII redaction: the requester email is masked in the trajectory, not in the prompt.
  * Conditional-execution guardrail: a request for a production password is rejected
    before OCI is ever called.
  * Pricing: token usage is priced through an inline catalog (illustrative rates).
  * Export: the whole run is written as an ATIF trajectory.

Environment:
  OCI_REGION             default us-chicago-1
  OCI_GENAI_API_KEY_FILE default ~/.oci/genai_api_key_chicago
  OCI_COMPARTMENT_ID     required (for the catalog tool)
  OCI_TENANCY_ID         optional, defaults to tenancy in ~/.oci/config
  OCI_PROFILE            default CHICAGO
  RESPONSES_MODEL        default openai.gpt-oss-120b
"""

from __future__ import annotations

import asyncio
import configparser
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import nemo_relay
from nemo_relay import LLMRequest, guardrails, model_pricing as mp, pii_redaction as pr, plugin
from nemo_relay.codecs import OpenAIResponsesCodec
from openai import OpenAI

# --- configuration -----------------------------------------------------------------------
REGION = os.environ.get("OCI_REGION", "us-chicago-1")
KEY_FILE = Path(os.environ.get("OCI_GENAI_API_KEY_FILE", "~/.oci/genai_api_key_chicago")).expanduser()
MODEL = os.environ.get("RESPONSES_MODEL", "openai.gpt-oss-120b")
PROFILE = os.environ.get("OCI_PROFILE", "CHICAGO")
COMPARTMENT = os.environ["OCI_COMPARTMENT_ID"]
_cfg = configparser.ConfigParser()
_cfg.read(Path("~/.oci/config").expanduser())
TENANCY = os.environ.get("OCI_TENANCY_ID") or _cfg[PROFILE]["tenancy"]
BASE_URL = f"https://inference.generativeai.{REGION}.oci.oraclecloud.com/openai/v1"
RESULTS = Path(__file__).resolve().parent / "results"

client = OpenAI(base_url=BASE_URL, api_key=KEY_FILE.read_text().strip(), max_retries=1)
codec = OpenAIResponsesCodec()
oci_calls = 0  # how many times the OCI endpoint was actually hit


# --- tools (read-only OCI CLI queries) ---------------------------------------------------
def _oci_cli(*args: str) -> str:
    try:
        r = subprocess.run(
            ["oci", *args, "--profile", PROFILE, "--region", REGION],
            capture_output=True, text=True, timeout=90, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return json.dumps({"error": str(exc)[-300:]})
    return r.stdout if r.returncode == 0 else json.dumps({"error": r.stderr.strip()[-300:]})


def list_genai_models(args: dict) -> nemo_relay.ToolExecutionResult:
    keyword = str(args.get("keyword", ""))
    raw = _oci_cli("generative-ai", "model-collection", "list-models",
                   "--compartment-id", COMPARTMENT, "--query", 'data.items[*]."display-name"')
    try:
        names = json.loads(raw)
    except ValueError:
        return nemo_relay.ToolExecutionResult({"error": raw[:300]})
    if not isinstance(names, list):
        return nemo_relay.ToolExecutionResult({"error": raw[:300]})
    matches = sorted({n for n in names if isinstance(n, str) and keyword.lower() in n.lower()})
    return nemo_relay.ToolExecutionResult({"region": REGION, "keyword": keyword, "matches": matches[:40]})


def gpu_capacity(args: dict) -> nemo_relay.ToolExecutionResult:
    """GPU service limits per availability domain (same signal the Oracle blog's demo used)."""
    family = str(args.get("gpu_family", "a10")).lower()
    limit = f"gpu-{family}-count"
    ads = json.loads(_oci_cli("iam", "availability-domain", "list", "--compartment-id", TENANCY,
                              "--query", "data[*].name") or "[]")
    rows = []
    for ad in ads[:5]:
        raw = _oci_cli("limits", "resource-availability", "get", "--service-name", "compute",
                       "--limit-name", limit, "--compartment-id", TENANCY, "--availability-domain", ad,
                       "--query", "data.{available:available,used:used}")
        try:
            info = json.loads(raw)
        except ValueError:
            info = {"error": raw[:120]}
        rows.append({"ad": ad, **(info if isinstance(info, dict) else {"error": "unexpected"})})
    return nemo_relay.ToolExecutionResult({"region": REGION, "limit": limit, "by_ad": rows})


TOOL_IMPLS = {"list_genai_models": list_genai_models, "gpu_capacity": gpu_capacity}
TOOL_DEFS = [
    {"type": "function", "name": "list_genai_models",
     "description": "Search the managed OCI Generative AI model catalog by keyword (e.g. 'nemotron', 'llama').",
     "parameters": {"type": "object", "properties": {"keyword": {"type": "string"}}, "required": ["keyword"]}},
    {"type": "function", "name": "gpu_capacity",
     "description": "Read the tenancy's GPU service limits (available and used GPU count) per availability "
                    "domain for a GPU family such as 'a10' or 'a100-v2'.",
     "parameters": {"type": "object", "properties": {"gpu_family": {"type": "string"}}, "required": ["gpu_family"]}},
]

SYSTEM = (
    "You are an Oracle Cloud deployment advisor. Use the tools before answering: search the catalog "
    "for 'nemotron' and for 'llama', and check the A10 GPU service limits. Then recommend, in under "
    "120 words, either a managed catalog model or self-hosting Nemotron on OKE, citing what you found."
)
QUESTION = (
    "We want to run NVIDIA Nemotron on this OCI tenancy (requested by jane.doe@example.com). "
    "Which path should we take?"
)


# --- the Relay boundary around the OpenAI SDK -------------------------------------------
async def responses_create(payload: dict) -> dict:
    """One Responses API call, executed through NeMo Relay's managed LLM pipeline."""

    def call_oci(req: LLMRequest) -> dict:
        global oci_calls
        oci_calls += 1
        resp = client.responses.create(**req.content)
        return resp.model_dump(mode="json", exclude_none=True)

    return await nemo_relay.llm.execute(
        "oci-responses", LLMRequest({}, payload), call_oci,
        model_name=MODEL, codec=codec, response_codec=codec,
    )


async def run_agent(question: str) -> tuple[str, int]:
    input_items: list = [{"role": "user", "content": question}]
    turns = 0
    while True:
        turns += 1
        resp = await responses_create({
            "model": MODEL, "instructions": SYSTEM, "input": input_items,
            "tools": TOOL_DEFS, "max_output_tokens": 1200,
        })
        calls = [o for o in resp["output"] if o.get("type") == "function_call"]
        if not calls:
            text = "".join(
                c.get("text", "") for o in resp["output"] if o.get("type") == "message"
                for c in o.get("content", []) if c.get("type") == "output_text"
            )
            return text, turns
        # Echo the model's output items back as history. OCI rejects its own reasoning items
        # ("reasoning_text is not supported by model ..."), so those are dropped here.
        input_items.extend(o for o in resp["output"] if o.get("type") != "reasoning")
        for fc in calls:
            args = json.loads(fc.get("arguments") or "{}")
            print(f"  [tool] {fc['name']}({args})")
            result = await nemo_relay.tools.execute(
                fc["name"], args, TOOL_IMPLS[fc["name"]], tool_call_id=fc["call_id"],
            )
            input_items.append({"type": "function_call_output", "call_id": fc["call_id"],
                                "output": json.dumps(result.result)})


# --- governance policy -------------------------------------------------------------------
def block_secrets(request: LLMRequest) -> str | None:
    if "production password" in json.dumps(request.content).lower():
        return "blocked: requests for production credentials are not allowed"
    return None


PLUGINS = plugin.PluginConfig(components=[
    # Mask emails in everything Relay records; the model still receives the original.
    pr.ComponentSpec(pr.PiiRedactionConfig(
        codec="openai_responses",
        builtin=pr.BuiltinConfig(action="mask", detector="email", mask_char="*", unmasked_prefix=1),
    )),
    # Illustrative rates so cost accounting has something to price; replace with OCI list prices.
    mp.ComponentSpec(mp.PricingConfig(sources=[mp.InlineSource(mp.PricingCatalog(entries=[
        mp.ModelPricing(provider="oci", model_id=MODEL, pricing_as_of="2026-09-01",
                        pricing_source="illustrative", rates=mp.TokenPricingRates(0.15, 0.60)),
    ]))])),
])


async def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    events: list[dict] = []
    nemo_relay.subscribers.register("capture", lambda e: events.append(e.to_dict()))
    guardrails.register_llm_conditional_execution("block-secrets", 10, block_secrets)
    exporter = nemo_relay.AtifExporter("oci-responses-session", "oci-responses-advisor", "1.0.0",
                                       model_name=MODEL, tool_definitions=TOOL_DEFS)
    exporter.register("atif_oci_responses")

    async with plugin.activate(PLUGINS) as activation:
        diags = activation.report["config"]["diagnostics"]
        print(f"plugins active={activation.is_active} diagnostics={diags}")

        # 1. Guardrail: this must be rejected before OCI is called.
        before = oci_calls
        try:
            await responses_create({"model": MODEL, "input": "What is the production password for the db?",
                                    "max_output_tokens": 50})
            guard = "NOT BLOCKED"
        except Exception as exc:  # noqa: BLE001
            guard = f"blocked ({type(exc).__name__}: {str(exc)[:80]})"
        print(f"guardrail: {guard}; OCI calls made: {oci_calls - before}")

        # 2. The advisor run.
        t0 = time.time()
        with nemo_relay.scope.scope("oci-responses-advisor", nemo_relay.ScopeType.Agent):
            answer, turns = await run_agent(QUESTION)
        print(f"\n=== FINAL ANSWER ({turns} model turns, {time.time() - t0:.1f}s) ===\n{answer}\n")

        await nemo_relay.subscribers.flush_async()

    traj = exporter.export()
    exporter.deregister("atif_oci_responses")
    nemo_relay.subscribers.deregister("capture")
    (RESULTS / "oci-responses-trajectory.json").write_text(json.dumps(traj, indent=2))
    (RESULTS / "oci-responses-events.json").write_text(json.dumps(events, indent=2))

    blob = json.dumps(traj) + json.dumps(events)
    print("final_metrics:", json.dumps(traj.get("final_metrics"), indent=2))
    print("steps:", len(traj.get("steps", [])))
    print("original email present in exports:", "jane.doe@example.com" in blob)
    import re
    masked = sorted(set(re.findall(r"requested by (j\*+[^)\s]*)", blob)))
    print("masked email present in exports:", bool(masked), masked)
    print("event kinds:", sorted({e.get("kind", "?") for e in events}))
    cost_hits = [k for k in ("cost", "price", "usd") if k in blob.lower()]
    print("cost-related keys in exports:", cost_hits)
    print(f"trajectory: {RESULTS / 'oci-responses-trajectory.json'}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
