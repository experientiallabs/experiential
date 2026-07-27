import json, urllib.request

PROMPTS = [
    "List three shell commands to find large files.",
    "You are in a Linux terminal. Write a one-liner to count lines in every .py file under src/.",
    "Explain in one sentence what `set -euo pipefail` does.",
    "A test is failing with 'ModuleNotFoundError: No module named requests'. What is your first command?",
    "Write a bash loop that retries a command up to 5 times with exponential backoff.",
]

ARMS = [("base-student", 8000), ("distill-student", 8001)]


def ask(model, port, prompt, max_tokens=256):
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    m = d["choices"][0]["message"]
    return (m.get("reasoning_content") or "") + "\n<<<ANSWER>>>\n" + (m.get("content") or "")


n_diff = 0
for i, p in enumerate(PROMPTS, 1):
    outs = {}
    for model, port in ARMS:
        try:
            outs[model] = ask(model, port, p)
        except Exception as e:
            outs[model] = f"<ERROR {type(e).__name__}: {e}>"
    a, b = outs["base-student"], outs["distill-student"]
    same = a == b
    if not same:
        n_diff += 1
    print(
        f"--- prompt {i}: {'IDENTICAL' if same else 'DIFFERS'} | len base={len(a)} distill={len(b)}"
    )
    print(f"    Q: {p}")
    print(f"    base   : {a[:220].replace(chr(10), ' | ')}")
    print(f"    distill: {b[:220].replace(chr(10), ' | ')}")
    print()

print("=" * 90)
print(f"RESULT: {n_diff}/{len(PROMPTS)} prompts produced different greedy output")
if n_diff == 0:
    print("FATAL: the two arms are the same model. Do NOT spend E2B budget.")
