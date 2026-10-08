#!/usr/bin/env python3
"""
expbench.py — MoE-expansion quality benchmark (5 problems x 10 configs).

Stage 1 (--stage gen):   for each routing config, start AgrillaMoE with that
                         config and generate one solution per problem; save
                         reasoning + answer to runs/<cfg>/<pid>.json (resumable).
Stage 2 (--stage tests): run the reference tests on each saved solution
                         (objective pass/fail).
Stage 3 (--stage judge): blinded LLM judge — per problem, the 10 candidates are
                         labeled A-J (shuffled with a fixed seed), scored and
                         ranked by a stock-config server; final report in
                         report.json + report.md.

Usage:
  python3 expbench.py --stage gen   --model models/M.gguf --port 8098 [extra server flags after --]
  python3 expbench.py --stage tests
  python3 expbench.py --stage judge --model models/M.gguf --judge-port 8099
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "configs.json"), encoding="utf-8"))["configs"]
PROBLEMS = json.load(open(os.path.join(HERE, "problems.json"), encoding="utf-8"))["problems"]
RUNS = os.path.join(HERE, "runs")
REASONING_BUDGET = int(os.environ.get("EXPBENCH_BUDGET", "24576"))
CTX = int(os.environ.get("EXPBENCH_CTX", "32768"))
MAX_TOKENS = REASONING_BUDGET + 4096
PORT_GEN = int(os.environ.get("EXPBENCH_PORT", "8098"))
PORT_JUDGE = int(os.environ.get("EXPBENCH_JPORT", "8099"))


def http_chat(port, messages, max_tokens, temperature=0.0, timeout=3600):
    body = json.dumps({"messages": messages, "max_tokens": max_tokens,
                       "temperature": temperature}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def health(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
            return json.loads(r.read()).get("status") == "ok"
    except Exception:
        return False


def start_server(model, flags, port, log):
    cmd = ["./dist/linux/agrillamoe", "-m", model] + flags + [
        "--flash-attn", "on", "-ctk", "q8_0", "-ctv", "q8_0",
        "-c", str(CTX), "-np", "1", "-ngl", "99", "--no-browser",
        "--host", "127.0.0.1", "--port", str(port),
        "--reasoning-budget", str(REASONING_BUDGET)]
    lf = open(log, "a")
    proc = subprocess.Popen(cmd, stdout=lf, stderr=lf, cwd=HERE)
    for _ in range(400):
        time.sleep(3)
        if health(port):
            return proc
        if proc.poll() is not None:
            return None
    return None


def stop_server(proc):
    try:
        proc.terminate()
        proc.wait(timeout=20)
    except Exception:
        pass


def extract_code(text):
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if m:
        code = m.group(1)
    else:
        m2 = re.search(r"```(?:python|py)?\s*\n(.*)", text, re.S)
        code = m2.group(1) if m2 else text
    lines = code.split("\n")
    for i, ln in enumerate(lines):
        if ln.startswith(("def ", "import ", "from ", "class ", "@")):
            code = "\n".join(lines[i:])
            break
    return code.rstrip() + "\n"


def run_tests(code, problem, timeout=20):
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "sol.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code + "\n\n" + "\n".join(problem["tests"]) +
                    f"\n\ncheck_ok = True\ntry:\n    check({problem['entry']})\nexcept NameError:\n    pass\n")
        # i test sono assert diretti sulla funzione: li eseguiamo uno a uno
        passed = 0
        details = []
        for t in problem["tests"]:
            try:
                r = subprocess.run([sys.executable, "-c",
                                    f"{code}\n\n{t}", ], capture_output=True,
                                   text=True, timeout=timeout, cwd=td)
                ok = r.returncode == 0
            except subprocess.TimeoutExpired:
                ok = False
            passed += ok
            details.append((ok, t[:60]))
        return passed, len(problem["tests"]), details


def stage_gen(args):
    os.makedirs(RUNS, exist_ok=True)
    base_flags = ["--flash-attn", "on", "-ctk", "q8_0", "-ctv", "q8_0",
                  "-c", str(CTX), "-np", "1", "-ngl", "99", "--no-browser"]
    if os.environ.get("EXPBENCH_CM") == "1":
        base_flags += ["-cmoe"]  # esperti su CPU per GPU 16GB (UD-Q3_K_XL)
    if args.extra:
        base_flags += args.extra.split()
    only = args.only_cfg.split(",") if args.only_cfg else None

    for cfg in CFG:
        if only and cfg["id"] not in only:
            continue
        out_dir = os.path.join(RUNS, cfg["id"])
        todo = [p for p in PROBLEMS
                if not os.path.exists(os.path.join(out_dir, p["id"] + ".json"))]
        if not todo:
            print(f"[{cfg['id']}] già completo, skip")
            continue
        print(f"===== config {cfg['id']}: {cfg['desc']} — {len(todo)} problemi da generare =====", flush=True)
        log = os.path.join(RUNS, f"srv-{cfg['id']}.log")
        proc = start_server(args.model, cfg["flags"] + base_flags, PORT_GEN, log)
        if proc is None:
            print(f"SERVER NON PRONTO ({cfg['id']}), skip — vedi {log}")
            continue
        for p in todo:
            outp = os.path.join(out_dir, p["id"] + ".json")
            prompt = ("Solve this programming problem in Python 3.\n"
                      "Reply with ONLY a complete Python 3 program in a single ```python code block.\n"
                      "The program must define exactly the requested function; no tests, no I/O.\n\n"
                      "## Problem\n\n" + p["statement"].strip() +
                      f"\n\n## Required signature\n\n```python\n{p['function']}\n```")
            rec = {"config": cfg["id"], "desc": cfg["desc"], "problem": p["id"],
                   "reasoning": "", "content": "", "gen_tokens": 0, "err": ""}
            try:
                resp = http_chat(PORT_GEN, [
                    {"role": "system", "content": "You are an expert Python programmer."},
                    {"role": "user", "content": prompt}], MAX_TOKENS)
                msg = resp["choices"][0]["message"]
                rec["reasoning"] = msg.get("reasoning_content") or ""
                rec["content"] = msg.get("content") or ""
                rec["gen_tokens"] = resp["usage"]["completion_tokens"]
            except Exception as e:
                rec["err"] = str(e)[:300]
            with open(outp, "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=1, ensure_ascii=False)
            print(f"  [{cfg['id']}] problema {p['id']}: {rec['gen_tokens']} token"
                  + (" ERRORE: " + rec["err"][:80] if rec["err"] else ""), flush=True)
        stop_server(proc)


def stage_tests():
    out = {}
    for cfg in CFG:
        for p in PROBLEMS:
            f = os.path.join(RUNS, cfg["id"], p["id"] + ".json")
            if not os.path.exists(f):
                continue
            d = json.load(open(f, encoding="utf-8"))
            if d.get("err") or not d.get("content"):
                d["tests_passed"], d["tests_total"] = 0, len(p["tests"])
            else:
                d["tests_passed"], d["tests_total"], _ = run_tests(
                    extract_code(d["content"]), p)
            d["test_ratio"] = round(d["tests_passed"] / max(1, d["tests_total"]), 3)
            json.dump(d, open(f, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
            out.setdefault(cfg["id"], []).append((p["id"], d["test_ratio"]))
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(RUNS, "tests-summary.json"), "w"), indent=1)


def stage_judge(args):
    # server di giudizio: config stock (nessuna espansione), temperatura 0
    log = os.path.join(RUNS, "srv-judge.log")
    proc = start_server(args.model, ["--no-moe-expansion"] + [
        "--flash-attn", "on", "-ctk", "q8_0", "-ctv", "q8_0",
        "-c", "32768", "-np", "1", "-ngl", "99", "--no-browser"], PORT_JUDGE, log)
    if proc is None:
        print("JUDGE SERVER NON PRONTO")
        return
    report = {}
    rng = random.Random(42)
    letters = "ABCDEFGHIJ"
    for p in PROBLEMS:
        candidates, mapping = [], []
        cfgs = [c for c in CFG
                if os.path.exists(os.path.join(RUNS, c["id"], p["id"] + ".json"))]
        if len(cfgs) < len(letters):
            letters_use = letters[:len(cfgs)]
        else:
            letters_use = letters
        order = list(range(len(cfgs)))
        rng.shuffle(order)  # accecamento stabile per problema
        for pos, ci in zip(letters_use, order):
            cfg = cfgs[ci]
            d = json.load(open(os.path.join(RUNS, cfg["id"], p["id"] + ".json"),
                               encoding="utf-8"))
            reasoning_tail = (d.get("reasoning") or "")[-1500:]
            candidates.append({
                "label": pos, "tests": f"{d.get('tests_passed', 0)}/{d.get('tests_total', len(p['tests']))}",
                "code": d.get("content", ""), "reasoning_tail": reasoning_tail})
            mapping.append({"label": pos, "config": cfg["id"], "desc": cfg["desc"]})
        cand_text = ""
        for c in candidates:
            cand_text += (f"\n\n### Candidate {c['label']} "
                          f"(reference tests passed: {c['tests']})\n"
                          f"```python\n{c['code']}\n```\n"
                          f"Reasoning excerpt (end of thinking):\n<cite>{c['reasoning_tail']}</cite>")
        jprompt = (
            "You are a strict senior judge for Python programming solutions.\n"
            "Below is one programming problem and 10 candidate solutions produced by "
            "different inference configurations (A-J). Some candidates may be "
            "incomplete, truncated or wrong.\n\n"
            "## Problem\n\n" + p["statement"].strip() +
            "\n\n" + cand_text +
            "\n\n## Your task\n\nScore each candidate 1-10 on solution quality "
            "(approach soundness, correctness likelihood, completeness) and 1-10 on "
            "code quality. Then pick the single best candidate.\n"
            "Reply with ONLY a JSON object, no markdown:\n"
            '{"scores":[{"label":"A","quality":7,"code":8,"notes":"short"}],'
            '"best":"A","ranking":["A","B",...]}'
            "\nInclude ALL candidates in scores. ranking = full order best to worst.")
        try:
            resp = http_chat(PORT_JUDGE, [
                {"role": "system", "content": "You are a rigorous evaluation judge. Reply only with valid JSON."},
                {"role": "user", "content": jprompt}], 2048)
            raw = resp["choices"][0]["message"]["content"] or ""
            m = re.search(r"\{.*\}", raw, re.S)
            parsed = json.loads(m.group(0)) if m else {"raw": raw[:400]}
        except Exception as e:
            parsed = {"error": str(e)[:200]}
        report[p["id"]] = {"title": p["title"], "mapping": mapping, "judge": parsed}
        print(f"[judge] {p['id']}: {json.dumps(parsed)[:160]}", flush=True)
    stop_server(proc)
    json.dump(report, open(os.path.join(RUNS, "judge-report.json"), "w"),
              indent=1, ensure_ascii=False)
    print("judge-report.json scritto")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["gen", "tests", "judge"])
    ap.add_argument("--model", default=os.environ.get("EXPBENCH_MODEL", "models/Q3KXL.gguf"))
    ap.add_argument("--port", type=int, default=PORT_GEN)
    ap.add_argument("--only-cfg", default=None)
    ap.add_argument("--extra", default="", help="flag server extra per lo stage gen")
    args = ap.parse_args()
    if args.stage == "gen":
        stage_gen(args)
    elif args.stage == "tests":
        stage_tests()
    elif args.stage == "judge":
        stage_judge(args)


if __name__ == "__main__":
    main()
