import json
import statistics

def analyze(path, label):
    snapshots = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["type"] == "snapshot":
                snapshots.append(rec)

    utils = [s["utilization_pct"] for s in snapshots]
    print(f"\n=== {label} ===")
    print(f"mean utilization: {statistics.mean(utils):.1f}%")
    print(f"median utilization: {statistics.median(utils):.1f}%")
    print(f"snapshots at 0% util (idle GPU-ticks): {sum(1 for u in utils if u == 0)} / {len(utils)}")

analyze("logs/fifo_baseline.jsonl", "FIFO baseline")
analyze("logs/best_fit.jsonl", "Best-fit")
