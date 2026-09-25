"""Streaming OCI Responses API call observed by NeMo Relay (``llm.stream_execute``).

Companion to ``oci_responses_relay.py``. The OpenAI SDK streams server-sent events from the
OCI Responses endpoint; Relay records every chunk, and the ``response.completed`` event's
response object is handed to the Responses codec as the final, normalized result.

Environment: OCI_REGION, OCI_GENAI_API_KEY_FILE, RESPONSES_MODEL (same defaults as the main script).
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import nemo_relay
from nemo_relay import LLMAttributes, LLMRequest
from nemo_relay.codecs import OpenAIResponsesCodec
from openai import OpenAI

REGION = os.environ.get("OCI_REGION", "us-chicago-1")
KEY_FILE = Path(os.environ.get("OCI_GENAI_API_KEY_FILE", "~/.oci/genai_api_key_chicago")).expanduser()
MODEL = os.environ.get("RESPONSES_MODEL", "openai.gpt-oss-120b")
RESULTS = Path(__file__).resolve().parent / "results"

client = OpenAI(
    base_url=f"https://inference.generativeai.{REGION}.oci.oraclecloud.com/openai/v1",
    api_key=KEY_FILE.read_text().strip(), max_retries=1,
)
codec = OpenAIResponsesCodec()


async def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    events: list[dict] = []
    nemo_relay.subscribers.register("capture", lambda e: events.append(e.to_dict()))
    exporter = nemo_relay.AtifExporter("oci-responses-stream-session", "oci-responses-stream", "1.0.0",
                                       model_name=MODEL)
    exporter.register("atif_oci_responses_stream")

    payload = {"model": MODEL, "input": "Name three OCI regions, one per line.", "max_output_tokens": 200}
    chunks: list[dict] = []
    final: dict = {}

    async def producer(req: LLMRequest):
        for ev in client.responses.create(**req.content, stream=True):
            d = ev.model_dump(mode="json", exclude_none=True)
            if d.get("type") == "response.completed":
                final.update(d["response"])
            yield d

    with nemo_relay.scope.scope("oci-responses-stream-agent", nemo_relay.ScopeType.Agent):
        stream = await nemo_relay.llm.stream_execute(
            "oci-responses-stream", LLMRequest({}, payload), producer,
            chunks.append, lambda: final,
            model_name=MODEL, codec=codec, response_codec=codec,
            attributes=LLMAttributes(LLMAttributes.STREAMING),
        )
        printed = 0
        async for item in stream:
            if item.get("type") == "response.output_text.delta":
                print(item.get("delta", ""), end="", flush=True)
                printed += 1
        print()

    await nemo_relay.subscribers.flush_async()
    traj = exporter.export()
    exporter.deregister("atif_oci_responses_stream")
    nemo_relay.subscribers.deregister("capture")
    (RESULTS / "oci-responses-stream-trajectory.json").write_text(json.dumps(traj, indent=2))

    print(f"\nSSE events relayed: {len(chunks)} (text deltas: {printed})")
    print("event types:", sorted({c.get("type", "?") for c in chunks}))
    print("final_metrics:", json.dumps(traj.get("final_metrics")))
    print("steps:", len(traj.get("steps", [])))
    print(f"trajectory: {RESULTS / 'oci-responses-stream-trajectory.json'}")


if __name__ == "__main__":
    asyncio.run(main())
