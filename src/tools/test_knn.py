import os, json, array
import httpx, redis

q = os.environ.get("KNN_QUERY", "top slow workflows")
ollama = os.environ.get("OLLAMA_HOST", "http://ollama:11434").rstrip("/")
embed_model = os.environ.get("EMBED_MODEL", "nomic-embed-text")
redis_url = os.environ.get("REDIS_URL", "redis://:devpass@ds_redis:6379/0")

# 1) embed the query
with httpx.Client(timeout=30.0) as cx:
    r = cx.post(f"{ollama}/api/embeddings", json={"model": embed_model, "prompt": q})
    r.raise_for_status()
    emb = r.json()["embedding"]
vec = array.array("f", emb).tobytes()

# 2) KNN search (KNN clause as ONE string; no __score usage)
r = redis.from_url(redis_url)
query = "*=>[KNN 3 @nl_desc_vector $vec]"
res = r.execute_command(
    "FT.SEARCH", "idx:queries", query,
    "PARAMS", "2", "vec", vec,
    "DIALECT", "2",
    "RETURN", "2", "nl_desc", "sql_template",
    "LIMIT", "0", "3"
)

out = {"count": res[0] if res else 0, "q": q, "items": []}
i = 1
while i < len(res):
    key = res[i].decode() if isinstance(res[i], (bytes, bytearray)) else str(res[i])
    fields = res[i+1]
    obj = {"key": key}
    for j in range(0, len(fields), 2):
        k = fields[j].decode() if isinstance(fields[j], (bytes, bytearray)) else str(fields[j])
        v = fields[j+1]
        if isinstance(v, (bytes, bytearray)):
            try: v = v.decode()
            except: v = f"<{len(v)} bytes>"
        obj[k] = v
    out["items"].append(obj)
    i += 2

print(json.dumps(out, indent=2))