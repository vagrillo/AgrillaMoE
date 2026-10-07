#!/usr/bin/env python3
"""
humaneval_run.py — HumanEval pass@1 via server OpenAI-compatibile (AgrillaMoE).

Uso: python3 humaneval_run.py <porta> <output.json> [n_problemi] [max_tokens]

Per ogni problema: invia il prompt come messaggio utente, estrae il codice dalla
risposta, lo esegue con i test canonici in subprocess con timeout, registra
pass/fail. Risultato: JSON con dettaglio per problema + pass@1 aggregato + t/s
medie dichiarate dal server (il driver le estrae dal log).
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
OUT = sys.argv[2] if len(sys.argv) > 2 else "humaneval-out.json"
NPROB = int(sys.argv[3]) if len(sys.argv) > 3 else 164
MAXTOK = int(sys.argv[4]) if len(sys.argv) > 4 else 2560
BUDGET = os.environ.get("HE_BUDGET", "1024")

API = f"http://127.0.0.1:{PORT}/v1/chat/completions"


def chat(prompt, max_tokens):
    body = json.dumps({
        "messages": [
            {"role": "system", "content": "You are an expert Python programmer."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(API, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def extract_code(text):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    code = m.group(1) if m else text
    # togli eventuali righe di testo prima della prima def/import
    lines = code.split("\n")
    for i, ln in enumerate(lines):
        if ln.startswith(("def ", "import ", "from ", "class ", "@")):
            code = "\n".join(lines[i:])
            break
    return code.rstrip()


def run_test(code, test, entry_point, timeout=12):
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "sol.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code + "\n\n" + test + f"\n\ncheck({entry_point})\n")
        try:
            r = subprocess.run([sys.executable, path], capture_output=True,
                               text=True, timeout=timeout, cwd=td)
            return r.returncode == 0, (r.stderr or "")[-400:]
        except subprocess.TimeoutExpired:
            return False, "timeout"


def main():
    from datasets import load_dataset
    try:
        ds = load_dataset("openai/openai_humaneval", split="test")
    except Exception:
        ds = load_dataset("openai_humaneval", split="test")
    problems = list(ds)[:NPROB]
    print(f"problemi: {len(problems)}, budget reasoning: {BUDGET}, max_tokens: {MAXTOK}", flush=True)

    results = []
    t0 = time.time()
    for i, p in enumerate(problems):
        prompt = (
            "Complete this Python function.\n"
            "Reply with ONLY the completed function inside a single ```python code block.\n"
            "No explanations, no tests, no example usage.\n\n"
            "```python\n" + p["prompt"] + "```"
        )
        rec = {"task_id": p["task_id"], "ok": False, "gen_tokens": 0, "err": ""}
        try:
            resp = chat(prompt, MAXTOK)
            msg = resp["choices"][0]["message"]
            text = msg.get("content") or ""
            rec["gen_tokens"] = resp["usage"]["completion_tokens"]
            rec["thinking_tokens"] = len(msg.get("reasoning_content") or "")
            # archivio completo per analisi successive: risposta e ragionamento integrali
            rec["prompt"] = prompt
            rec["content"] = text
            rec["reasoning"] = msg.get("reasoning_content") or ""
            code = extract_code(text)
            rec["ok"], rec["err"] = run_test(code, p["test"], p["entry_point"])
        except Exception as e:
            rec["err"] = str(e)[:300]
        results.append(rec)
        if (i + 1) % 10 == 0 or i == len(problems) - 1:
            npass = sum(1 for r in results if r["ok"])
            dt = time.time() - t0
            print(f"  {i+1}/{len(problems)}  pass {npass}  ({dt:.0f}s, {dt/(i+1):.1f}s/prob)", flush=True)

    npass = sum(1 for r in results if r["ok"])
    out = {
        "n": len(results),
        "passed": npass,
        "pass_at_1": round(100.0 * npass / len(results), 2),
        "total_gen_tokens": sum(r["gen_tokens"] for r in results),
        "wall_seconds": round(time.time() - t0, 1),
        "reasoning_budget": BUDGET,
        "results": results,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"PASS@1: {out['pass_at_1']}% ({npass}/{len(results)})  -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
