#!/usr/bin/env python3
# Rerun a budget 8K delle domande (indice >= 50) il cui reasoning nel run 16K
# ha superato 8192 token: completa il segmento accoppiato 8K-vs-16K oltre le
# prime 50. Modalita': extract | run | report
import json, re, sys, time, threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BUDGET8 = 8192.0

def rtok_of(d):
    pm = d.get("model_output", {}).get("perf_metrics") or {}
    comp = pm.get("output_tokens") or 0
    blocks = (d["model_output"].get("choices") or [{}])[0].get("message", {}).get("content") or []
    if isinstance(blocks, str):
        blocks = [{"type": "text", "text": blocks}]
    rchars = sum(len(b.get("reasoning") or "") for b in blocks if isinstance(b, dict))
    tchars = sum(len(b.get("text") or "") for b in blocks if isinstance(b, dict))
    return comp * (rchars / max(1, rchars + tchars))

def extract():
    for cfg in ("noexp", "exp01"):
        sel = []
        for line in open(f"/root/gpqa16k-{cfg}/predictions/LFM25-16k-{cfg}/gpqa_diamond_default.jsonl"):
            d = json.loads(line)
            if d["index"] < 50:
                continue
            if rtok_of(d) > BUDGET8:
                m0 = d["messages"][0]
                sel.append({"index": d["index"],
                            "messages": [{"role": m0.get("role", "user"),
                                          "content": m0.get("content")}],
                            "correct_answer": (d.get("metadata") or {}).get("correct_answer")})
        with open(f"/root/cap8kset_{cfg}.jsonl", "w") as f:
            for c in sel:
                f.write(json.dumps(c) + "\n")
        print(cfg, "selezionate:", len(sel), "(indice >= 50, reasoning 16K > 8192 tok)")

def norm(s):
    s = (s or "").strip().lower()
    s = re.sub(r"\\+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .\"'`")

def options_from_prompt(p):
    ms = re.findall(r"A\)\s*([^\n]+?)\s*B\)\s*([^\n]+?)\s*C\)\s*([^\n]+?)\s*D\)\s*([^\n]+)", p)
    if not ms:
        return {}
    a, b, c, dd = ms[-1]
    return {"A": a, "B": b, "C": c, "D": dd}

def answer_letter(text):
    ms = re.findall(r"ANSWER:\s*\(?([A-D])\)?", text or "", re.I)
    return ms[-1] if ms else None

def score(rec, out_text):
    letter = answer_letter(out_text)
    if not letter:
        return 0.0, None
    chosen = norm(options_from_prompt(rec["messages"][0]["content"]).get(letter, ""))
    want = norm(rec["correct_answer"])
    ok = chosen == want or (chosen and (chosen.startswith(want) or want.startswith(chosen)))
    return (1.0 if ok else 0.0), letter

def run_cfg(cfg, port):
    recs = [json.loads(l) for l in open(f"/root/cap8kset_{cfg}.jsonl")]
    lock = threading.Lock()
    done = [0]
    out_f = open(f"/root/cap8krun_{cfg}.jsonl", "w")

    def work(rec):
        body = json.dumps({"messages": rec["messages"], "temperature": 0,
                           "max_tokens": 16384, "stream": False}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                     data=body, headers={"Content-Type": "application/json"})
        t0 = time.time()
        r = json.load(urllib.request.urlopen(req, timeout=3600))
        msg = r["choices"][0]["message"]
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
        sc, letter = score(rec, content)
        rec_out = {"index": rec["index"], "score": sc, "letter": letter,
                   "latency": time.time() - t0,
                   "out_tokens": (r.get("usage") or {}).get("completion_tokens"),
                   "answer_tail": content[-200:]}
        with lock:
            out_f.write(json.dumps(rec_out) + "\n")
            out_f.flush()
            done[0] += 1
            if done[0] % 10 == 0:
                print(f"  {cfg}: {done[0]}/{len(recs)}", flush=True)

    def safe_work(rec):
        try:
            work(rec)
        except Exception as e:
            with lock:
                done[0] += 1
                out_f.write(json.dumps({"index": rec["index"], "score": 0.0,
                                        "error": repr(e)[:160]}) + "\n")
                out_f.flush()
                print(f"  {cfg}: ERRORE su index {rec['index']}: {repr(e)[:100]}", flush=True)

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(safe_work, recs))
    out_f.close()

def report():
    for cfg in ("noexp", "exp01"):
        try:
            r8 = [json.loads(l) for l in open(f"/root/cap8krun_{cfg}.jsonl")]
        except FileNotFoundError:
            print(cfg, "run 8K non presente")
            continue
        if not r8:
            print(cfg, "run vuoto")
            continue
        r16 = {}
        rpath = f"/root/gpqa16k-{cfg}/reviews/LFM25-16k-{cfg}/gpqa_diamond_default.jsonl"
        for line in open(rpath):
            d = json.loads(line)
            sc = ((d.get("sample_score") or {}).get("score") or {}).get("value") or {}
            r16[d["index"]] = float(sc.get("accuracy", 0.0))
        done = [r for r in r8 if "error" not in r]
        a8 = sum(r["score"] for r in done) / len(done) * 100 if done else 0
        a16 = sum(r16.get(r["index"], 0) for r in done) / len(done) * 100 if done else 0
        up = sum(1 for r in done if r["score"] == 1 and r16.get(r["index"], 0) == 0)
        dn = sum(1 for r in done if r["score"] == 0 and r16.get(r["index"], 0) == 1)
        print(f"{cfg} (cap-16K, n={len(done)}): @8K {a8:.1f}% vs @16K {a16:.1f}%  (salte {up}, cadute {dn})")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "report"
    if mode == "extract":
        extract()
    elif mode == "run":
        run_cfg(sys.argv[2], int(sys.argv[3]))
    elif mode == "report":
        report()
