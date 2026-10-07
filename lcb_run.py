#!/usr/bin/env python3
"""
lcb_run.py — LiveCodeBench-v6-Plus (BenchEvolver, 91 problems) via an
OpenAI-compatible server (AgrillaMoE).

Competitive-programming format: the model writes a complete Python program
that reads stdin and writes stdout; every test case (public + private) is run
in a subprocess and compared after stripping trailing whitespace.

Usage:
  python3 lcb_run.py <port> <out.json> [--dataset URL-or-path] [--timeout 15] [--limit N]
Env: LCB_THINKING_LABEL (cosmetic label in the report), LCB_JUDGE_SELFTEST=1
     (validate the judge with the bundled reference solutions instead of the model).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

DEFAULT_DS = "https://huggingface.co/datasets/BenchEvolver/livecodebench-plus/resolve/main/{split}.jsonl"


def load_problems(ds):
    rows = []
    for split in ("medium", "hard"):
        if "://" in ds:
            url = ds.replace("{split}", split)
            with urllib.request.urlopen(url, timeout=120) as r:
                text = r.read().decode()
        else:
            with open(ds if ds.endswith(".jsonl") else os.path.join(ds, split + ".jsonl"), encoding="utf-8") as f:
                text = f.read()
        for line in text.splitlines():
            if line.strip():
                r = json.loads(line)
                r["_split"] = split
                rows.append(r)
    return rows


def chat(port, prompt, max_tokens):
    body = json.dumps({
        "messages": [
            {"role": "system", "content":
             "You are an expert competitive programmer. Solve the problem in Python 3."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def extract_code(text):
    import re
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    code = m.group(1) if m else text
    return code.rstrip() + "\n"


def norm(out):
    return "\n".join(ln.rstrip() for ln in out.replace("\r\n", "\n").split("\n")).strip()


def run_solution(code, tests, tdir, per_test_timeout):
    results = []
    for i, tc in enumerate(tests):
        path = os.path.join(tdir, f"sol{i}.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            r = subprocess.run([sys.executable, path], input=tc["input"].encode(),
                               capture_output=True, timeout=per_test_timeout, cwd=tdir)
            got = norm(r.stdout.decode(errors="replace"))
            want = norm(tc["output"])
            results.append((got == want, f"test{i}({tc.get('tier','?')}): " +
                            ("ok" if got == want else f"MISMATCH got={got[:120]!r} want={want[:120]!r}" +
                             (f" stderr={r.stderr.decode(errors='replace')[:200]!r}" if r.returncode != 0 else ""))))
        except subprocess.TimeoutExpired:
            results.append((False, f"test{i}: TIMEOUT {per_test_timeout}s"))
    return results


def judge_selftest(problems, timeout):
    """Run the bundled reference solutions (evolved problems only) to validate the judge."""
    print("== JUDGE SELFTEST (reference solutions) ==", flush=True)
    npass = 0
    for p in problems:
        if not p.get("solution_code"):
            continue
        with tempfile.TemporaryDirectory() as td:
            res = run_solution(p["solution_code"], p["test_cases"], td, timeout)
        ok = all(x[0] for x in res)
        npass += ok
        status = "OK " if ok else "FAIL"
        print(f"  [{status}] {p['_split']}/{p['problem_id'][:8]} {len(res)} tests" +
              ("" if ok else "  " + next(msg for c, msg in res if not c)[:140]), flush=True)
    tot = sum(1 for p in problems if p.get("solution_code"))
    print(f"judge selftest: {npass}/{tot} reference solutions passed", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    ap.add_argument("out")
    ap.add_argument("--dataset", default=DEFAULT_DS)
    ap.add_argument("--timeout", type=int, default=15, help="per-test timeout seconds")
    ap.add_argument("--limit", type=int, default=0, help="only first N problems (0=all)")
    ap.add_argument("--max-tokens", type=int, default=6144)
    args = ap.parse_args()

    problems = load_problems(args.dataset)
    if args.limit:
        problems = problems[:args.limit]
    print(f"problemi: {len(problems)} (medium+hard)", flush=True)

    if os.environ.get("LCB_JUDGE_SELFTEST") == "1":
        judge_selftest(problems, args.timeout)
        return

    budget = os.environ.get("LCB_THINKING_BUDGET", "4096")
    results = []
    t0 = time.time()
    for i, p in enumerate(problems):
        prompt = (
            "Solve this competitive programming problem. Read input from stdin and "
            "write the answer to stdout. Reply with ONLY a complete Python 3 program "
            "in a single ```python code block — no explanations.\n\n"
            "## Problem\n\n" + p["problem_statement"].strip()
        )
        if p.get("public_tests"):
            ex = p["public_tests"][0]
            prompt += ("\n\n## Example\n\nInput:\n```\n" + ex["input"].rstrip() +
                       "\n```\nOutput:\n```\n" + ex["output"].rstrip() + "\n```")
        rec = {"problem_id": p["problem_id"], "split": p["_split"], "title": p["title"],
               "difficulty": p["difficulty"], "ok": False, "gen_tokens": 0,
               "tests_passed": 0, "tests_total": len(p["test_cases"]), "err": ""}
        try:
            resp = chat(args.port, prompt, args.max_tokens)
            msg = resp["choices"][0]["message"]
            rec["gen_tokens"] = resp["usage"]["completion_tokens"]
            rec["reasoning_tokens"] = len(msg.get("reasoning_content") or "")
            rec["content"] = msg.get("content") or ""
            rec["reasoning"] = msg.get("reasoning_content") or ""
            code = extract_code(rec["content"])
            with tempfile.TemporaryDirectory() as td:
                res = run_solution(code, p["test_cases"], td, args.timeout)
            rec["tests_passed"] = sum(1 for c, _ in res if c)
            first_fail = next((m for c, m in res if not c), "")
            rec["ok"] = all(c for c, _ in res)
            rec["err"] = "" if rec["ok"] else first_fail[:300]
        except Exception as e:
            rec["err"] = str(e)[:300]
        results.append(rec)
        if (i + 1) % 5 == 0 or i == len(problems) - 1:
            npass = sum(1 for r in results if r["ok"])
            print(f"  {i+1}/{len(problems)}  pass {npass}  ({time.time()-t0:.0f}s)", flush=True)

    def agg(sel):
        sub = [r for r in results if sel(r)]
        if not sub:
            return None
        return {"n": len(sub), "passed": sum(r["ok"] for r in sub),
                "pass_at_1": round(100.0 * sum(r["ok"] for r in sub) / len(sub), 2)}
    out = {
        "overall": agg(lambda r: True),
        "medium": agg(lambda r: r["split"] == "medium"),
        "hard": agg(lambda r: r["split"] == "hard"),
        "thinking_budget": budget,
        "wall_seconds": round(time.time() - t0, 1),
        "results": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"OVERALL pass@1: {out['overall']['pass_at_1']}% "
          f"({out['overall']['passed']}/{out['overall']['n']})  "
          f"medium: {out['medium']}  hard: {out['hard']}  -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
