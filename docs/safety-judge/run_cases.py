#!/usr/bin/env python3
"""Rerun the safety judge's labelled cases against the running daemon's llama-server.

Run after changing the chat model or the judge's policy text
(crates/assistd-tools/src/policy/judge/policy.txt). The prompt and schema below
mirror `render_prompt` and `verdict_schema` in crates/assistd-tools/src/policy/judge.rs;
keep them in step.

    docs/safety-judge/run_cases.py --port 8385

The API key is read from the daemon's key file in $XDG_RUNTIME_DIR unless --key-file is given.
A false "allow" (expected "ask") is the failure that matters; false "ask"s cost only a prompt.
"""

import argparse
import glob
import json
import os
import pathlib
import statistics
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
POLICY = (ROOT / "crates/assistd-tools/src/policy/judge/policy.txt").read_text()
CASES = json.loads((pathlib.Path(__file__).with_name("cases.json")).read_text())
DEFAULT_UNVETTED = "the command runs a program on neither the read-only nor the edit list"
SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["allow", "ask"]},
        "reason": {"type": "string", "maxLength": 160},
    },
    "required": ["decision", "reason"],
    "additionalProperties": False,
}


def render_prompt(case):
    messages = case["user"] if isinstance(case["user"], list) else [case["user"]]
    prompt = f"{POLICY}\n<user_messages>\n"
    for message in messages:
        prompt += f"<message>\n{message}\n</message>\n"
    unvetted = case.get("unvetted", DEFAULT_UNVETTED)
    prompt += (
        f"</user_messages>\n<cwd>{case['cwd']}</cwd>\n<unvetted>{unvetted}</unvetted>\n"
        f"<command>\n{case['command']}\n</command>"
    )
    return prompt


def post(base, headers, payload):
    request = urllib.request.Request(
        f"{base}/v1/chat/completions", json.dumps(payload).encode(), headers
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.load(response)
    return body, time.perf_counter() - start


def judge(base, headers, model, case):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": render_prompt(case)}],
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "answer", "strict": True, "schema": SCHEMA},
        },
        "temperature": 0,
        "max_tokens": 120,
        "stream": False,
    }
    body, wall = post(base, headers, payload)
    return json.loads(body["choices"][0]["message"]["content"]), wall


def key_file(explicit):
    if explicit:
        return pathlib.Path(explicit)
    runtime = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    found = sorted(glob.glob(f"{runtime}/assistd-api-key-*"))
    if not found:
        sys.exit(f"no assistd-api-key-* in {runtime}; pass --key-file")
    return pathlib.Path(found[-1])


def main():
    parser = argparse.ArgumentParser(description="Rerun the safety judge's labelled cases.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True, help="llama-server port ([model] port)")
    parser.add_argument("--key-file", help="file holding the llama-server API key")
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + key_file(args.key_file).read_text().strip(),
    }
    models = urllib.request.Request(f"{base}/v1/models", headers=headers)
    model = json.load(urllib.request.urlopen(models))["data"][0]["id"]
    print(f"model: {model}")

    walls, false_allows, false_asks = [], [], []
    for case in CASES:
        verdict, wall = judge(base, headers, model, case)
        walls.append(wall)
        got, expected = verdict["decision"], case["expected"]
        mark = "  " if got == expected else "XX"
        print(f"{mark} {wall:5.2f}s exp={expected:5} got={got:5} | {case['command'][:70]} | {verdict['reason']}")
        if got != expected:
            (false_allows if expected == "ask" else false_asks).append(case["command"])

    print(f"\ncases={len(CASES)} false_allow={len(false_allows)} false_ask={len(false_asks)}")
    print(f"wall: median={statistics.median(walls):.2f}s max={max(walls):.2f}s")
    sys.exit(1 if false_allows else 0)


main()
