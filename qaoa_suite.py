"""All 5 use cases on a real IBM quantum computer (QAOA, 1 layer).

  python qaoa_suite.py --size medium --dry-run   # build + classical + circuit size, no QPU
  python qaoa_suite.py --size medium             # real QPU
  python qaoa_suite.py --size large --only trucks,timetable

Per use case:
  classical exact (OR-Tools CP-SAT on the same QUBO)
  random guessing (same number of samples)            <- the bar quantum must beat
  real QPU: angles tuned classically (noiseless sim, or transferred from the
            small instance when too big to simulate), then 4096 shots
  'polished' = simple classical bit-flip clean-up on the top samples (shown separately)
"""
import argparse, json, os, time

import numpy as np
from ortools.sat.python import cp_model
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from scipy.optimize import minimize
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

import usecases as U

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------ QUBO helpers
def qmat(p):
    Q = np.zeros((p["n"], p["n"]))
    for (i, j), c in p["quad"].items():
        Q[i, j] = c
    return Q


def energies(p, X, Q):
    X = X.astype(float)
    return p["const"] + X @ p["lin"] + np.einsum("ij,jk,ik->i", X, Q, X)


def polish(p, x, Q):
    """Greedy one-bit-flip descent: classical clean-up of a sample."""
    x = x.copy().astype(float)
    S = Q + Q.T
    while True:
        delta = (1 - 2 * x) * (p["lin"] + S @ x)
        i = int(np.argmin(delta))
        if delta[i] >= -1e-9:
            return x.astype(int)
        x[i] = 1 - x[i]


def exact(p, time_limit=60, scale=100):
    m = cp_model.CpModel()
    x = [m.NewBoolVar(f"x{i}") for i in range(p["n"])]
    obj = [int(round(scale * c)) * x[i] for i, c in enumerate(p["lin"]) if abs(c) > 1e-12]
    for (i, j), c in p["quad"].items():
        y = m.NewBoolVar(f"y{i}_{j}")
        if c > 0:
            m.Add(y >= x[i] + x[j] - 1)
        else:
            m.Add(y <= x[i]); m.Add(y <= x[j])
        obj.append(int(round(scale * c)) * y)
    m.Minimize(sum(obj))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_workers = 4
    t0 = time.perf_counter()
    st = s.Solve(m)
    sol = np.array([s.Value(v) for v in x])
    return sol, time.perf_counter() - t0, st == cp_model.OPTIMAL


# ------------------------------------------------------------ circuit
def ising(p):
    """x=(1-z)/2  ->  h_i z_i + J_ij z_i z_j, normalised so the largest coefficient is 1."""
    h = -p["lin"] / 2
    J = {}
    for (i, j), c in p["quad"].items():
        h[i] -= c / 4
        h[j] -= c / 4
        J[(i, j)] = c / 4
    scale = max(np.abs(h).max(), max((abs(v) for v in J.values()), default=0))
    return h / scale, {k: v / scale for k, v in J.items()}, scale


def qaoa_circuit(p, reps=1):
    n = p["n"]
    h, J, _ = ising(p)
    betas, gammas = ParameterVector("b", reps), ParameterVector("g", reps)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for layer in range(reps):
        for (i, j), c in J.items():
            qc.rzz(2 * gammas[layer] * c, i, j)
        for i in range(n):
            if abs(h[i]) > 1e-12:
                qc.rz(2 * gammas[layer] * h[i], i)
        qc.rx(2 * betas[layer], range(n))
    qc.measure_all()
    return qc, list(betas) + list(gammas)


SIM_MAX_QUBITS = 20


def tune(p, Q, reps=1, starts=8):
    """Classical step (free): find QAOA angles on a noiseless simulator.
    Returns angles [b_1..b_p, g_1..g_p] and the noiseless 'progress vs random'."""
    from run_ibm import fast_qaoa_state
    n = p["n"]
    X = (np.arange(2 ** n)[:, None] >> np.arange(n)) & 1
    E = energies(p, X, Q)
    _, _, scale = ising(p)
    d = E / scale                       # same operator as the circuit, up to a constant
    Er, Eo = E.mean(), E.min()

    def mean_E(th):
        pr = np.abs(fast_qaoa_state(d, n, np.asarray(th), reps)) ** 2
        return float((pr * E).sum())
    rng = np.random.default_rng(0)
    best = None
    for _ in range(starts):
        x0 = np.concatenate([rng.uniform(0, 0.8, reps), rng.uniform(0, 1.0, reps)])
        r = minimize(mean_E, x0, method="COBYLA", options=dict(maxiter=300))
        if best is None or r.fun < best.fun:
            best = r
    pr = np.abs(fast_qaoa_state(d, n, best.x, reps)) ** 2
    return best.x, dict(progress_vs_random_pct=round(100 * (Er - best.fun) / (Er - Eo), 1),
                        hit_best_pct=round(100 * float(pr[np.isclose(E, Eo)].sum()), 3))


def to_X(counts, n):
    rows, w = [], []
    for bits, c in counts.items():
        rows.append([int(ch) for ch in bits[::-1][:n]])
        w.append(c)
    return np.array(rows), np.array(w)


def summarise(p, X, w, Q, E_opt, E_rand_mean):
    E = energies(p, X, Q)
    mean = float((E * w).sum() / w.sum())
    order = np.argsort(E)
    best = X[order[0]]
    feas = np.array([p["decode"](x)["feasible"] for x in X])
    top = X[order[:30]]
    pol = min((polish(p, x, Q) for x in top), key=lambda x: energies(p, x[None], Q)[0])
    span = E_rand_mean - E_opt
    return dict(
        samples=int(w.sum()),
        feasible_pct=round(100 * float((feas * w).sum() / w.sum()), 2),
        hit_best_pct=round(100 * float(w[np.isclose(E, E_opt, atol=1e-6)].sum() / w.sum()), 3),
        progress_vs_random_pct=round(100 * (E_rand_mean - mean) / span, 1) if span > 0 else None,
        best_raw=p["decode"](best),
        best_raw_gap_pct=round(100 * (E[order[0]] - E_opt) / abs(span), 2) if span else None,
        best_polished=p["decode"](pol),
    )


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=list(U.SIZES), default="medium")
    ap.add_argument("--only", default=None)
    ap.add_argument("--backend", default="ibm_marrakesh")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--reps", type=int, default=1, help="QAOA layers")
    a = ap.parse_args()

    probs = U.build_all(a.size)
    if a.only:
        keep = a.only.split(",")
        probs = [p for p in probs if any(k in p["name"] for k in keep)]

    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    svc = QiskitRuntimeService(instance="open-instance")
    backend = svc.backend(a.backend)
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=1)
    print(f"Backend: {backend.name} ({backend.num_qubits} qubits)  size={a.size}")

    out_path = os.path.join(HERE, "results", f"suite_{a.size}_p{a.reps}{'_dry' if a.dry_run else ''}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}
    rng = np.random.default_rng(0)
    angles_path = os.path.join(HERE, "results", f"angles_p{a.reps}.json")
    angles = json.load(open(angles_path)) if os.path.exists(angles_path) else {}

    for p in probs:
        print(f"\n=== {p['name']}  {p['meta']}  qubits={p['n']}  pairs={len(p['quad'])} ===", flush=True)
        if p["n"] > backend.num_qubits:
            print("  too many qubits, skipping"); continue
        Q = qmat(p)
        sol, ex_s, proven = exact(p)
        E_opt = float(energies(p, sol[None], Q)[0])
        R = rng.integers(0, 2, size=(a.shots, p["n"]))
        E_rand_mean = float(energies(p, R, Q).mean())
        r = dict(meta=p["meta"], backend=backend.name, qubits=p["n"], qubo_pairs=len(p["quad"]),
                 classical_exact=dict(**p["decode"](sol), seconds=round(ex_s, 3), proven_optimal=proven))
        r["random_guess"] = summarise(p, R, np.ones(len(R)), Q, E_opt, E_rand_mean)

        qc, params = qaoa_circuit(p, a.reps)
        t0 = time.perf_counter()
        isa = pm.run(qc)
        ops = isa.count_ops()
        r["circuit"] = dict(depth=isa.depth(), two_qubit_gates=int(sum(v for k, v in ops.items() if k in ("cz", "ecr", "cx"))),
                            transpile_s=round(time.perf_counter() - t0, 1))
        print("  classical:", r["classical_exact"])
        print("  circuit:", r["circuit"], flush=True)

        key = p["name"]
        if p["n"] <= SIM_MAX_QUBITS:
            theta, ideal = tune(p, Q, a.reps)
            r["perfect_quantum_computer"] = ideal
            angles[key] = list(map(float, theta))
            json.dump(angles, open(angles_path, "w"), indent=1)
            r["angles_source"] = "tuned on noiseless simulator (this instance)"
        elif key in angles:
            theta = np.array(angles[key])
            r["angles_source"] = "transferred from the small instance of the same use case"
        else:
            print("  no angles available; run --size small first"); continue
        r["angles"] = [round(float(t), 4) for t in theta]
        print("  angles:", r["angles"], r.get("perfect_quantum_computer"), flush=True)
        if a.dry_run:
            results[p["name"]] = r
            continue
        sampler = SamplerV2(mode=backend)
        t0 = time.perf_counter()
        order = {par.name: k for k, par in enumerate(params)}
        vals = np.array([[theta[order[par.name]] for par in isa.parameters]])
        job2 = sampler.run([(isa, vals)], shots=a.shots)
        print("  job", job2.job_id(), flush=True)
        X, w = to_X(job2.result()[0].data.meas.get_counts(loc=0), p["n"])
        r["qpu"] = summarise(p, X, w, Q, E_opt, E_rand_mean)
        r["qpu"].update(job_id=job2.job_id(), qpu_seconds=job2.usage(), wall_s=round(time.perf_counter() - t0, 1))
        print(json.dumps({k2: r[k2] for k2 in ("random_guess", "qpu")}, indent=1, default=str), flush=True)
        results[p["name"]] = r
        json.dump(results, open(out_path, "w"), indent=2, default=str)

    json.dump(results, open(out_path, "w"), indent=2, default=str)
    print("\nSaved", out_path)


if __name__ == "__main__":
    main()
