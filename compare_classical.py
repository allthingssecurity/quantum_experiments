"""Quantum vs classical on the same QUBOs.

For every use case and size that has a real IBM result, run a classical sampler
(simulated annealing, dwave-samplers) with the same number of samples (4096),
and put it next to the exact solver (OR-Tools CP-SAT) and the IBM run.

  python compare_classical.py          # writes results/classical_compare.json
"""
import json, os, time

import numpy as np
from dwave.samplers import SimulatedAnnealingSampler
import dimod

import usecases as U
import qaoa_suite as S

HERE = os.path.dirname(os.path.abspath(__file__))


def to_bqm(p):
    return dimod.BinaryQuadraticModel(
        {i: float(c) for i, c in enumerate(p["lin"])},
        {k: float(v) for k, v in p["quad"].items()}, p["const"], "BINARY")


def main():
    out = {}
    sa = SimulatedAnnealingSampler()
    for size in ("small", "medium", "large"):
        path = os.path.join(HERE, "results", f"suite_{size}_p1.json")
        if not os.path.exists(path):
            continue
        done = json.load(open(path))
        probs = {p["name"]: p for p in U.build_all(size)}
        for name, r in done.items():
            p = probs[name]
            Q = S.qmat(p)
            bqm = to_bqm(p)
            t0 = time.perf_counter()
            ss = sa.sample(bqm, num_reads=r["qpu"]["samples"], seed=1)
            secs = time.perf_counter() - t0
            X = np.array([[row[i] for i in range(p["n"])] for row in ss.record.sample])
            E = S.energies(p, X, Q)
            sol, _, _ = S.exact(p, time_limit=60)
            E_opt = float(S.energies(p, sol[None], Q)[0])
            feas = [p["decode"](x)["feasible"] for x in X]
            best = X[int(np.argmin(E))]
            out[f"{size}/{name}"] = dict(
                samples=len(X), seconds=round(secs, 2),
                feasible_pct=round(100 * float(np.mean(feas)), 2),
                hit_best_pct=round(100 * float(np.mean(np.isclose(E, E_opt, atol=1e-6))), 2),
                best=p["decode"](best))
            print(size, name, out[f"{size}/{name}"], flush=True)
    json.dump(out, open(os.path.join(HERE, "results", "classical_compare.json"), "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
