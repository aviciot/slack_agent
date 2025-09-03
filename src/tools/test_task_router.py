from src.services.task_router import TaskRouter

if __name__ == "__main__":
    tr = TaskRouter("/app/data/task_bank.json")
    tests = [
        "show me statements count by status",
        "compare yesterday csv with today csv",
        "email me the report"
    ]
    for t in tests:
        d = tr.route(t)
        print(f"\nText: {t}")
        print(f" -> task_type={d.task_type}, conf={d.confidence:.2f}, source={d.source}")
        for c in d.candidates:
            print(f"    cand: {c.id} score={c.score:.2f} reasons={c.reasons}")
