"""Classical vs quantum annealing on the two PoC problems.

  python run.py --mode sim              # free: classical exact + simulated annealing,
                                        #       and checks the problem fits a D-Wave chip
  python run.py --mode qpu              # real D-Wave QPU (needs Leap token configured)
  python run.py --mode qpu --hybrid     # also one Leap hybrid run on BHAVYA (~5 s of trial time)

Results go to results/<mode>.json and results/<mode>.md
"""
import argparse, json, os, time
from collections import defaultdict

import dimod
from dimod import cqm_to_bqm
from ortools.sat.python import cp_model

import problems as P

RESULTS = os.path.join(os.path.dirname(__file__), "results")


# ------------------------------------------------------------ classical exact
def exact_bhavya(data, time_limit=60):
    m = cp_model.CpModel()
    x = {p: m.NewBoolVar(p) for p in data["parks"]}
    y = {}
    for (a, b) in data["competition"]:
        y[(a, b)] = m.NewBoolVar(f"{a}_{b}")
        m.Add(y[(a, b)] >= x[a] + x[b] - 1)
    m.Add(sum(data["cost"][p] * x[p] for p in x) <= data["budget"])
    m.Maximize(sum(data["benefit"][p] * x[p] for p in x)
               - sum(c * y[k] for k, c in data["competition"].items()))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_workers = 4
    t0 = time.perf_counter()
    st = s.Solve(m)
    dt = time.perf_counter() - t0
    chosen = [p for p in x if s.Value(x[p])]
    return dict(chosen=chosen, seconds=dt, optimal=(st == cp_model.OPTIMAL))


def exact_trucks(data, time_limit=60):
    m = cp_model.CpModel()
    x = {p: m.NewBoolVar(f"{p[0]}-{p[1]}") for p in data["pairs"]}
    for key in (0, 1):
        groups = defaultdict(list)
        for p in x:
            groups[p[key]].append(x[p])
        for g in groups.values():
            m.Add(sum(g) <= 1)
    R = data["reward"]
    m.Maximize(sum(int(round((R - km) * 10)) * x[p] for p, km in data["pairs"].items()))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_workers = 4
    t0 = time.perf_counter()
    st = s.Solve(m)
    dt = time.perf_counter() - t0
    chosen = [p for p in x if s.Value(x[p])]
    return dict(chosen=chosen, seconds=dt, optimal=(st == cp_model.OPTIMAL))


# ------------------------------------------------------------ helpers
def bhavya_repair(data, chosen):
    """Classical clean-up of a quantum sample: drop worst value-per-cost parks
    until within budget, then greedily add any park that still fits and helps."""
    s = set(chosen)
    ratio = lambda p: data["benefit"][p] / data["cost"][p]
    spend = lambda: sum(data["cost"][p] for p in s)
    while spend() > data["budget"]:
        s.remove(min(s, key=ratio))
    for p in sorted(set(data["parks"]) - s, key=ratio, reverse=True):
        if spend() + data["cost"][p] <= data["budget"]:
            before, _ = P.bhavya_value(data, s)
            s.add(p)
            after, _ = P.bhavya_value(data, s)
            if after <= before:
                s.remove(p)
    return sorted(s)


def bhavya_bqm(cqm, data):
    # Penalty weight for breaking the budget. Too small: answers go over budget.
    # Too big: the QPU's limited precision can no longer "see" the benefits.
    lag = max(data["benefit"].values()) / max(data["cost"].values()) * 2
    bqm, invert = cqm_to_bqm(cqm, lagrange_multiplier=lag)
    return bqm, invert, lag


def summarise_bhavya(data, sampleset, invert):
    best_raw, best_rep, feas = None, None, 0
    for rec in sampleset.data(["sample"]):
        s = invert(rec.sample)
        chosen = [p for p in data["parks"] if s.get(p)]
        val, ok = P.bhavya_value(data, chosen)
        feas += ok
        if ok and (best_raw is None or val > best_raw):
            best_raw = val
        rv, _ = P.bhavya_value(data, bhavya_repair(data, chosen))
        best_rep = rv if best_rep is None else max(best_rep, rv)
    return dict(best_raw=best_raw, best_repaired=best_rep,
                feasible_pct=round(100 * feas / len(sampleset), 1))


def summarise_trucks(data, sampleset):
    best, feas = None, 0
    names = {f"{p[0]}-{p[1]}": p for p in data["pairs"]}
    for rec in sampleset.data(["sample"]):
        chosen = [names[v] for v, on in rec.sample.items() if on]
        n, km, ok = P.trucks_value(data, chosen)
        feas += ok
        score = n * data["reward"] - km
        if ok and (best is None or score > best[0]):
            best = (score, n, km)
    return dict(best_loads=best and best[1], best_empty_km=best and best[2],
                feasible_pct=round(100 * feas / len(sampleset), 1))


def fits_on_chip(bqm):
    """Offline check that the problem can be embedded on a D-Wave-like chip."""
    import dwave_networkx as dnx, minorminer
    out = {}
    for name, G in [("Advantage (Pegasus P16)", dnx.pegasus_graph(16)),
                    ("Advantage2 (Zephyr Z12)", dnx.zephyr_graph(12, 4))]:
        t0 = time.perf_counter()
        emb = minorminer.find_embedding(list(bqm.quadratic), G.edges, random_seed=1, timeout=60)
        if emb:
            lens = [len(c) for c in emb.values()]
            out[name] = dict(fits=True, physical_qubits=sum(lens), max_chain=max(lens),
                             embed_seconds=round(time.perf_counter() - t0, 1))
        else:
            out[name] = dict(fits=False)
    return out


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sim", "qpu"], default="sim")
    ap.add_argument("--problem", choices=["all", "tiny", "bhavya", "trucks"], default="all")
    ap.add_argument("--reads", type=int, default=1000)
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--qpu-budget-s", type=float, default=20.0,
                    help="stop before using more than this many seconds of QPU access time")
    a = ap.parse_args()

    if a.mode == "qpu":
        from dwave.system import DWaveSampler, EmbeddingComposite
        qpu = DWaveSampler()
        print(f"QPU: {qpu.solver.name}  ({len(qpu.nodelist)} working qubits)")
        sampler = EmbeddingComposite(qpu)
    else:
        from dwave.samplers import SimulatedAnnealingSampler
        sampler = SimulatedAnnealingSampler()
    qpu_used_us = 0.0

    def anneal(bqm, label):
        nonlocal qpu_used_us
        if a.mode == "qpu" and qpu_used_us / 1e6 > a.qpu_budget_s:
            raise SystemExit(f"QPU budget {a.qpu_budget_s}s reached; stopping before {label}")
        t0 = time.perf_counter()
        kw = dict(num_reads=a.reads)
        if a.mode == "qpu":
            kw.update(label=f"sap-poc-{label}", return_embedding=True)
        ss = sampler.sample(bqm, **kw)
        ss.resolve()
        wall = time.perf_counter() - t0
        info = dict(wall_seconds=round(wall, 3), reads=a.reads)
        if a.mode == "qpu":
            t = ss.info["timing"]
            qpu_used_us += t["qpu_access_time"]
            info.update(qpu_access_ms=round(t["qpu_access_time"] / 1000, 1),
                        anneal_us_per_read=t.get("qpu_anneal_time_per_sample"),
                        chain_break_pct=round(100 * float(ss.record.chain_break_fraction.mean()), 2)
                        if "chain_break_fraction" in ss.record.dtype.names else None,
                        solver=ss.info.get("problem_label") or label)
            emb = ss.info.get("embedding_context", {}).get("embedding")
            if emb:
                info["physical_qubits"] = sum(len(c) for c in emb.values())
                info["max_chain"] = max(len(c) for c in emb.values())
        return ss, info

    results = {}
    todo = ["tiny", "bhavya", "trucks"] if a.problem == "all" else [a.problem]

    for name in todo:
        print(f"\n=== {name} ===")
        if name in ("tiny", "bhavya"):
            cqm, data = P.bhavya_tiny() if name == "tiny" else P.bhavya()
            ex = exact_bhavya(data)
            ex_val, _ = P.bhavya_value(data, ex["chosen"])
            bqm, invert, lag = bhavya_bqm(cqm, data)
            r = dict(size=dict(parks=len(data["parks"]), qubo_switches=bqm.num_variables,
                               qubo_pairs=bqm.num_interactions, lagrange=round(lag, 3)),
                     classical_exact=dict(value=ex_val, seconds=round(ex["seconds"], 4),
                                          proven_optimal=ex["optimal"], funded=len(ex["chosen"])))
            ss, info = anneal(bqm, name)
            r["annealer"] = {**info, **summarise_bhavya(data, ss, invert)}
            if name == "tiny":
                r["annealer"]["best_sample_parks"] = [p for p in data["parks"]
                                                       if invert(ss.first.sample).get(p)]
            if a.mode == "sim" and name == "bhavya":
                r["embedding_check"] = fits_on_chip(bqm)
            if a.hybrid and a.mode == "qpu" and name == "bhavya":
                from dwave.system import LeapHybridCQMSampler
                h = LeapHybridCQMSampler()
                t0 = time.perf_counter()
                hs = h.sample_cqm(cqm, time_limit=h.min_time_limit(cqm), label="sap-poc-bhavya-hybrid")
                hs.resolve()
                feas = hs.filter(lambda d: d.is_feasible)
                best = None
                if len(feas):
                    chosen = [p for p in data["parks"] if feas.first.sample.get(p)]
                    best, _ = P.bhavya_value(data, chosen)
                r["leap_hybrid"] = dict(value=best, wall_seconds=round(time.perf_counter() - t0, 2),
                                        qpu_access_ms=round(hs.info.get("qpu_access_time", 0) / 1000, 1),
                                        run_time_s=round(hs.info.get("run_time", 0) / 1e6, 2))
        else:
            cqm, data = P.trucks()
            ex = exact_trucks(data)
            n, km, _ = P.trucks_value(data, ex["chosen"])
            bqm = P.trucks_bqm(data)
            r = dict(size=dict(trucks=len(data["trucks"]), loads=len(data["loads"]),
                               qubo_switches=bqm.num_variables, qubo_pairs=bqm.num_interactions),
                     classical_exact=dict(loads=n, empty_km=km, seconds=round(ex["seconds"], 4),
                                          proven_optimal=ex["optimal"]))
            ss, info = anneal(bqm, name)
            r["annealer"] = {**info, **summarise_trucks(data, ss)}
            if a.mode == "sim":
                r["embedding_check"] = fits_on_chip(bqm)
        results[name] = r
        print(json.dumps(r, indent=2))

    if a.mode == "qpu":
        results["total_qpu_access_s"] = round(qpu_used_us / 1e6, 3)
        print(f"\nTotal QPU access time used: {results['total_qpu_access_s']} s")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"{a.mode}.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results/{a.mode}.json")


if __name__ == "__main__":
    main()
