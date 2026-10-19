"""Print the markdown result tables used in REPORT.md from results/*.json."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {"1_bhavya": "Park funding", "2_trucks": "Truck backhaul", "3_dfc_rakes": "DFC rakes",
         "4_timetable": "Timetabling", "5_multimodal": "Multimodal routing"}


def fmt(r, m):
    v = r.get(m)
    return "infeasible" if not r["feasible"] else f"{v:g}" if isinstance(v, (int, float)) else str(v)


def table(size):
    path = os.path.join(HERE, "results", f"suite_{size}_p1.json")
    if not os.path.exists(path):
        return f"(no results for {size})\n"
    d = json.load(open(path))
    m0 = "metric"
    rows = ["| Use case | Qubits | 2-qubit gates | Progress, perfect QC | Progress, IBM | Valid answers, random | Valid answers, IBM | Best IBM answer | Best possible |",
            "|---|---|---|---|---|---|---|---|---|"]
    for k, r in d.items():
        m = r["meta"][m0]
        q, g = r["qpu"], r["random_guess"]
        perfect = r.get("perfect_quantum_computer")
        best = r["classical_exact"]
        tag = "" if best.get("proven_optimal") else " (not proven)"
        rows.append("| {} | {} | {} | {} | {}% | {}% | {}% | {} | {}{} |".format(
            NAMES[k], r["qubits"], r["circuit"]["two_qubit_gates"],
            f"{perfect['progress_vs_random_pct']}%" if perfect else "n/a",
            q["progress_vs_random_pct"], g["feasible_pct"], q["feasible_pct"],
            fmt(q["best_raw"], m), fmt(best, m), tag))
    return "\n".join(rows) + "\n"


def jobs():
    out = ["| Run | Use case | IBM job id |", "|---|---|---|"]
    for size in ("small", "medium", "large"):
        path = os.path.join(HERE, "results", f"suite_{size}_p1.json")
        if os.path.exists(path):
            for k, r in json.load(open(path)).items():
                out.append(f"| {size} | {NAMES[k]} | {r['qpu']['job_id']} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    for s in ("small", "medium", "large"):
        print(f"### {s}\n\n{table(s)}")
    print(jobs())
