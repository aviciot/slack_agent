# DS_agent

ngrok config add-authtoken YOUR_TOKEN
to run the service
ngrok http 8090



# 1) Copy .env.example → .env and fill Slack tokens
# 2) Start ollama sidecar
docker compose up -d ollama
# 3) Pull a model (inside ollama container)
docker exec -it ollama bash -lc "ollama pull mistral"
# 4) Build & run API
docker compose up -d --build
# 5) Tail logs
docker compose logs -f api


# for dummy db 
/sql total sales in the last 7 days

/sql total sales in the last month

/sql top 3 products by revenue this month

/sql daily sales totals for the past 14 days

/sql average order value last week

If your /sql expects more explicit phrasing, use:

/sql compute the sum of sales.amount for the last 7 days

/sql list top 3 products by sum(sales.amount) in the current month



docker compose run --rm kb-seed python /app/tools/seed_queries.py


NL2SQL checks (to test the LLM path)
/sql list the first 5 rows from v_wf_hierarchy_explorer


/sql top 10 workflows by number of sessions

/sql show 3 columns (subject_area, workflow_name, task_type_name) from session_run_statistics limit 5


/sql top 10 workflows by total sessions in the last 30 days

/sql failure rate (failed/total) by workflow in the last 30 days, top 10 by total volume

/sql average session duration (seconds) by workflow in the last 7 days, top 10

/sql daily total sessions for the last 14 days (date, sessions)

/sql top 10 workflows by sessions in the last 7 days where total sessions ≥ 100

/sql success rate by workflow in the last 7 days (success/total), show top 10 by volume

/sql top 10 workflows by max session duration (seconds) in the last 7 days

/sql workflows with error rate > 5% in the last 7 days, order by total sessions desc

/sql top 10 workflows by growth in sessions vs previous 7 days

/sql hourly sessions today (00:00–23:59)

 in Slack:

Quick smoke tests
/ask What columns are in session_run_statistics?

/ask Show 5 sample rows from workflow_run_statistics.

Simple analytics
/ask Which 10 workflows have the most sessions? Return a table and a short takeaway.

/ask What’s the busiest subject_area? Give counts by subject_area (top 5).

/ask For v_wf_hierarchy_explorer, list 5 example rows and explain what each column represents.

“Figure it out” / agent-y prompts
/ask Find any workflows with unusually high run counts compared to the median. Explain briefly.

/ask Are there any workflows that look like outliers by sessions per day? Show the table you used.

Debug / schema-aware
/ask Do we have a timestamp column to filter recent runs? If yes, show 5 most recent session_run_statistics rows.

/ask Build the SQL to list workflow_name and total sessions, sorted desc. Then run it.

And a follow-up in the same thread (to test conversational memory):

/ask Now do the same, but only for the top 3 subject areas.

### docker ollama 


'{"model":"anindya/prem1b-sql-ollama-fp116","prompt":"warm","stream":false,"options":{"num_predict":1},"keep_alive":"10m"}' |
  Set-Content -Encoding ASCII warm.json
curl.exe -s -X POST http://localhost:11434/api/generate -H "Content-Type: application/json" --data-binary "@warm.json"

# See live usage
docker stats ollama

# What models are currently LOADED in RAM (and by whom)
docker exec -it ollama ollama ps

# What models are INSTALLED on disk
docker exec -it ollama ollama list

# Inspect container limits (helps spot no memory cap / swap thrash)
docker inspect ollama --format='Memory={{.HostConfig.Memory}} Swap={{.HostConfig.MemorySwap}}'


docker exec -it ollama ollama stop mistral:latest
docker exec -it ollama ollama stop llama3.1:8b-instruct-q4_K_M
docker exec -it ollama ollama ps   # should now be empty


# Qwen coder
python sanity_ollama.py -m qwen2.5-coder:7b-instruct --threads 6 --show_ps

# Mistral (pick the tag you have)
python sanity_ollama.py -m mistral:latest --threads 6 --show_ps


#Redis
docker exec ds_redis redis-cli -a devpass FLUSHALL

docker compose exec redis redis-cli -a devpass FLUSHALL
docker compose exec redis redis-cli -a devpass FT.DROPINDEX idx:qcache DD

docker compose exec api sh -lc "env | grep -E 'CATALOG|TABLE|JSON|DB_PATH'"


docker compose run --rm --build kb-seed

docker compose exec redis redis-cli -a devpass KEYS "*"

If you also want to pull fresh base images (in case Python or deps changed):

docker compose build --no-cache api


And then restart:

docker compose up -d api


⚡ Pro tip: If you only changed requirements.txt, you can trigger just the dependency layer rebuild:

docker compose up -d --build api


tree

tree /f /a | findstr /v "\\Test" | findstr /v "\\V1" > tree_output.txt



get env variable from the dockers
docker exec -it ds_agent_api env | Select-String PROVIDER