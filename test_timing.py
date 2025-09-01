# nl2sql_perf_lab.py
# Self-contained NL→SQL benchmark for Ollama + SQLite.
# Edit the CONFIG section below, then run:  python nl2sql_perf_lab.py

import os, re, time, json, sqlite3
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# ========= CONFIG (EDIT ME) =========
REQUIRED_TABLE_REGEX = [r"session.*run.*statistics", r"workflow.*run.*statistics"]

DB_PATH         = r".\data\db\informatica_insigts.sqlite"
QUESTION        = "top 10 workflows by max session duration (seconds) in the last 30 days"

OLLAMA_BASE     = "http://localhost:11434"
MODELS          = [
    "qwen2.5-coder:7b-instruct",   # current baseline
    "qwen2.5:3b-instruct",       # uncomment to compare faster CPU model
    # "phi3:mini",                 # another small baseline
]
KEEP_ALIVE      = "300s"            # "0" or duration like "300s","5m"
NUM_THREADS     = 6
NUM_CTX         = 1024              # smaller → faster prompt-eval
PREDICT_SWEEP   = [32, 48]      # lower = faster; must still finish the SQL
TEMP            = 0.1
TOP_P           = 0.9
STOP_STRINGS    = []
HTTP_TIMEOUT_S  = 240

# Auto-prompt sizing (no hard-coded table names)
MAX_TABLES      = 3                 # pick top-K relevant tables by keyword overlap
MAX_COLS        = 8                 # most relevant columns per table
SCHEMA_MAX_CHARS= 600               # final schema cap
PREVIEW_ROWS    = 5                 # print first N rows
WARM_UP         = True              # warm per model to avoid cold-load noise
# Extra keywords to bias selection (optional; helps relevance)
KEYWORDS_EXTRA  = ["workflow","session","duration","time","count","status","error","run"]
# ====================================

def now_ms(): return time.perf_counter() * 1000.0
def ns_to_ms(v): return (float(v)/1e6) if v else 0.0

def read_db_meta(db_path):
    con = sqlite3.connect(db_path, timeout=5)
    meta = {}
    try:
        cur = con.cursor()
        cur.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%'")
        for name, typ in cur.fetchall():
            cols, colset = [], set()
            try:
                cur.execute(f"PRAGMA table_info('{name}')")
                for _, cname, ctype, *_ in cur.fetchall():
                    cols.append((cname, ctype))
                    colset.add(cname.lower())
            except Exception:
                pass
            fks = []
            try:
                cur.execute(f"PRAGMA foreign_key_list('{name}')")
                for (_, _, ref_table, from_col, to_col, *_rest) in cur.fetchall():
                    fks.append((from_col, ref_table, to_col))
            except Exception:
                pass
            meta[name] = {"type": typ, "cols": cols, "colset": colset, "fks": fks}
    finally:
        con.close()
    return meta

def tok(s): return re.findall(r"[a-zA-Z0-9_]+", (s or "").lower())

def select_tables(question, meta, k=MAX_TABLES):
    q_tokens = set(tok(question) + [w.lower() for w in KEYWORDS_EXTRA])

    # score all tables
    def score_table(tname):
        info = meta[tname]
        name_overlap = len(q_tokens & set(tok(tname)))
        col_tokens = set()
        for cname, _ in info["cols"]:
            col_tokens.update(tok(cname))
        col_overlap = len(q_tokens & col_tokens)
        join_bonus = sum(1 for c in col_tokens if c.endswith("id") or c in {
            "workflow_id","session_id","run_id","workflow_run_id","session_run_id"
        })
        # extra boost if table looks like session/workflow stats
        bonus = 3 if re.search(r"(session|workflow).*(stat|run)", tname, re.I) else 0
        return 3*name_overlap + 2*col_overlap + join_bonus + bonus

    scored = sorted(((score_table(t), t) for t in meta.keys()), reverse=True)

    # force-include best matches for required regexes
    required = []
    for pat in REQUIRED_TABLE_REGEX:
        cand = [t for _, t in scored if re.search(pat, t, re.I)]
        if cand:
            best = cand[0]
            if best not in required:
                required.append(best)

    # fill the rest by score
    picked = required[:]
    for _, t in scored:
        if t not in picked:
            picked.append(t)
        if len(picked) >= k:
            break

    return picked[:k]

def pick_columns(info, q_tokens, max_cols=MAX_COLS):
    cols = info["cols"]
    def weight(cn, ct):
        name = cn.lower(); w = 0
        if name in q_tokens: w += 3
        if name.endswith("_id") or name in {"id","workflow_id","session_id","run_id","workflow_run_id"}: w += 3
        if any(k in name for k in ["time","date","dur","sec","min","count","total","status","error","success","fail"]): w += 2
        if any(k in (ct or "").lower() for k in ["int","real","num","date","time"]): w += 1
        return w
    ranked = sorted(cols, key=lambda c: weight(c[0], c[1]), reverse=True)
    return ranked[:max_cols]

def build_schema_snippet(meta, tables, question):
    q_tokens = set(tok(question) + [w.lower() for w in KEYWORDS_EXTRA])
    parts = []
    for t in tables:
        sel = pick_columns(meta[t], q_tokens, MAX_COLS)
        parts.append(f"TABLE {t}(" + ", ".join(f"{c} {tp}" for c,tp in sel) + ")")
    s = "\n".join(parts)
    return s[:SCHEMA_MAX_CHARS]

def join_hints(meta, tables):
    if len(tables) < 2: return ""
    keyish = lambda n: n.endswith("_id") or n in {"id","workflow_id","session_id","run_id","workflow_run_id"}
    key_cols = {t: {c for c,_ in meta[t]["cols"] if keyish(c.lower())} for t in tables}
    hints = []
    for i,t1 in enumerate(tables):
        for t2 in tables[i+1:]:
            for c in sorted(key_cols[t1] & key_cols[t2]):
                hints.append(f"{t1}.{c} = {t2}.{c}")
    hints = sorted(set(hints))[:6]
    return "\n".join(f"- {h}" for h in hints)

def make_prompt(schema_snip, question, join_hint_text):
    rules = (
        "You generate SQLite SQL only.\n"
        "- Return ONE query that returns rows (SELECT or WITH allowed).\n"
        "- No PRAGMA/DDL/DML. No comments. No explanations. No code fences.\n"
        "- Start with SELECT or WITH.\n"
    )
    jh = (f"Joins to consider:\n{join_hint_text}\n" if join_hint_text else "")
    return f"{rules}\n{jh}\nSCHEMA:\n{schema_snip}\n\nQUESTION:\n{question}\n\nSQL:"

def post_json(url, payload, timeout_s):
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))

def call_ollama(model, prompt, num_predict):
    url = OLLAMA_BASE.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "num_ctx": NUM_CTX,
            "num_thread": NUM_THREADS,
            "num_predict": int(num_predict),
            "temperature": TEMP,
            "top_p": TOP_P,
            "stop": STOP_STRINGS,
        },
    }
    t0 = now_ms()
    try:
        d = post_json(url, payload, timeout_s=HTTP_TIMEOUT_S)
    except HTTPError as e:
        body = e.read().decode("utf-8","ignore")
        raise RuntimeError(f"Ollama HTTP {e.code}: {body}") from None
    except URLError as e:
        raise RuntimeError(f"Ollama connection error: {e}") from None
    wall = now_ms() - t0
    return {
        "load_ms": ns_to_ms(d.get("load_duration")),
        "prompt_ms": ns_to_ms(d.get("prompt_eval_duration")),
        "prompt_tok": int(d.get("prompt_eval_count") or 0),
        "gen_ms": ns_to_ms(d.get("eval_duration")),
        "gen_tok": int(d.get("eval_count") or 0),
        "wall_ms": wall,
        "text": (d.get("response") or "").strip(),
    }

def strip_sql(text):
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.strip("`"); s = re.sub(r"^sql\s+", "", s, flags=re.I).strip()
    parts = [p.strip() for p in re.split(r";\s*", s) if p.strip()]
    for stmt in reversed(parts):
        if re.match(r"^\s*(select|with)\b", stmt, re.I):
            return stmt
    m = re.search(r"(?is)\b(select|with)\b.*", s)
    return m.group(0).strip() if m else s

def ensure_limit(sql, max_rows=500):
    s = sql.strip().rstrip(";")
    if re.match(r"^\s*select\b", s, re.I) and not re.search(r"\blimit\s+\d+\b", s, re.I):
        s += f" LIMIT {max_rows}"
    return s + ";"

def run_sqlite(db_path, sql):
    con = sqlite3.connect(db_path, timeout=5, isolation_level=None)
    try:
        cur = con.cursor()
        cur.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall() if cols else []
    finally:
        con.close()
    return cols, rows

def main():
    print("==== NL→SQL Perf Lab ====")
    print(f"DB: {DB_PATH}")
    meta = read_db_meta(DB_PATH)
    picked = select_tables(QUESTION, meta, MAX_TABLES)
    schema_snip = build_schema_snippet(meta, picked, QUESTION)
    joins = join_hints(meta, picked)
    prompt = make_prompt(schema_snip, QUESTION, joins)
    print("Picked tables:", ", ".join(picked))
    print(f"Schema chars: {len(schema_snip)}")
    if joins: print("Join hints:\n" + joins)
    print("----")

    best = None
    for model in MODELS:
        print(f"[MODEL] {model}")
        if WARM_UP:
            try: _ = call_ollama(model, "OK", 8)
            except Exception: pass

        for np in PREDICT_SWEEP:
            r = call_ollama(model, prompt, np)
            sql = ensure_limit(strip_sql(r["text"]))
            try:
                cols, rows = run_sqlite(DB_PATH, sql)
                ok = len(rows) > 0
            except Exception as e:
                cols, rows, ok = [], [], False
                sql += f"  -- ERROR: {e}"
            snip = sql.replace("\n"," ")[:160] + ("…" if len(sql)>160 else "")
            print(f"  num_predict={np:>2} | load={r['load_ms']:.0f}ms  prompt={r['prompt_tok']}tok/{r['prompt_ms']:.0f}ms  "
                  f"gen={r['gen_tok']}tok/{r['gen_ms']:.0f}ms  wall={r['wall_ms']:.0f}ms  rows={len(rows)}")
            print(f"    SQL: {snip}")
            if ok and (best is None or r["wall_ms"] < best["wall_ms"]):
                best = {"model": model, "np": np, **r, "rows": len(rows)}
        print("----")

    if best:
        print(f"BEST: {best['model']} with num_predict={best['np']}  "
              f"wall={best['wall_ms']:.0f}ms  (gen={best['gen_ms']:.0f}ms, prompt={best['prompt_ms']:.0f}ms, rows={best['rows']})")
    else:
        print("No configuration returned rows. Increase num_predict or adjust STOP_STRINGS.")
    print("========================")

if __name__ == "__main__":
    main()
