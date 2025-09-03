from src.services.task_bank import TaskBank

if __name__ == "__main__":
    tb = TaskBank("/app/data/task_bank.json")
    queries = [
        "show me statements count by status",
        "compare yesterday csv with today csv",
        "email me the report"
    ]
    for q in queries:
        result = tb.search(q)
        print(f"\nQuery: {q}")
        for r in result:
            print(f"  -> {r.id} (score={r.score:.2f}, reasons={r.reasons})")
