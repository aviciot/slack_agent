import requests, csv, os, time

API=os.getenv("API_URL","http://localhost:8090")
QUESTIONS=[
  "statements count by merchant for 2025-07 by status",
  "workflows using table MS_PDF_SUMMARY as source",
]

def run(q):
  r = requests.post(f"{API}/debug/agent/run", json={"text": q})
  r.raise_for_status()
  return r.json()

rows=[]
for q in QUESTIONS:
  a1 = run(q); time.sleep(0.5)
  a2 = run(q)
  rows.append([q, a1.get("summary",{}).get("route"), a1.get("summary",{}).get("tables"),
               a1.get("status"), "first"])
  rows.append([q, a2.get("summary",{}).get("route"), a2.get("summary",{}).get("tables"),
               a2.get("status"), "second"])

with open("qcache_smoke.csv","w",newline="",encoding="utf-8") as f:
  w=csv.writer(f); w.writerow(["question","route","tables","status","run"])
  w.writerows(rows)

print("Wrote qcache_smoke.csv")
