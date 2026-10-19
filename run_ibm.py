"""QAOA on a real IBM quantum computer for the two small PoC examples.

  python run_ibm.py --save-key   # once: asks for your IBM API key + instance, stores it locally
  python run_ibm.py --mode sim   # free: noiseless simulator only
  python run_ibm.py --mode qpu   # real IBM hardware (uses a few seconds of the 10 free minutes)

Steps per problem:
  1. Write the problem as a QUBO (yes/no switches + score).
  2. Tune the QAOA angles on a classical simulator (free, saves QPU time).
  3. Run the tuned circuit on the real QPU, 4096 shots.
  4. Compare: how often the QPU returns the best answer vs random guessing.
"""
import argparse, itertools, json, os, time

import numpy as np
from scipy.optimize import minimize
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------ the two problems
def parks4():
    """4 parks, budget Rs 100 Cr. Best: fund B, C, D (benefit 18).
    The budget rule uses 'unbalanced penalisation': a penalty on the leftover
    budget h = 100 - spend, so no extra slack qubits are needed (4 qubits total)."""
    names = ["A", "B", "C", "D"]
    cost = np.array([4, 3, 5, 2])           # in Rs 10 Cr
    benefit = np.array([8, 5, 9, 4])
    B, l1, l2 = 10, 2.0, 2.0      # penalty weights checked by brute force: B+C+D is the unique best

    def score(x):
        h = B - cost @ x
        return -benefit @ x - l1 * h + l2 * h * h

    def real(x):
        return (int(benefit @ x), bool(cost @ x <= B))
    return dict(name="parks4", labels=names, n=4, score=score, real=real,
                describe=lambda x: "fund " + "+".join(n for n, b in zip(names, x) if b))


def trucks3():
    """3 empty trucks, 3 return loads, empty km table from the slides.
    Best: T1->A, T2->B, T3->C = 30 empty km. 9 qubits."""
    km = np.array([[10, 40, 30], [35, 5, 50], [25, 45, 15]])
    R, P = 60, 150                           # reward per load carried, clash penalty
    labels = [f"T{t+1}{'ABC'[l]}" for t in range(3) for l in range(3)]

    def score(x):
        X = np.array(x).reshape(3, 3)
        s = ((km - R) * X).sum()
        for t in range(3):
            s += P * (X[t].sum() * (X[t].sum() - 1)) / 2        # truck takes 2+ loads
        for l in range(3):
            s += P * (X[:, l].sum() * (X[:, l].sum() - 1)) / 2  # load gets 2+ trucks
        return s

    def real(x):
        X = np.array(x).reshape(3, 3)
        ok = X.sum(1).max() <= 1 and X.sum(0).max() <= 1
        return (int(X.sum()), int((km * X).sum()), bool(ok))
    return dict(name="trucks3", labels=labels, n=9, score=score, real=real,
                describe=lambda x: ", ".join(l[:2] + "->" + l[2] for l, b in zip(labels, x) if b) or "none")


# ------------------------------------------------------------ QUBO -> quantum
def to_qubo(prob):
    """Read the QUBO coefficients off the score function (exact for quadratics)."""
    n, f = prob["n"], prob["score"]
    e = np.eye(n, dtype=int)
    c0 = f(np.zeros(n, dtype=int))
    lin = np.array([f(e[i]) - c0 for i in range(n)])
    Q = {}
    for i, j in itertools.combinations(range(n), 2):
        q = f(e[i] + e[j]) - c0 - lin[i] - lin[j]
        if abs(q) > 1e-9:
            Q[(i, j)] = q
    return c0, lin, Q


def hamiltonian(n, c0, lin, Q):
    """Switch x_i in {0,1} becomes (1 - Z_i)/2 on qubit i."""
    terms, const = [], c0
    for i, h in enumerate(lin):
        const += h / 2
        terms.append(("Z", [i], -h / 2))
    for (i, j), q in Q.items():
        const += q / 4
        terms += [("Z", [i], -q / 4), ("Z", [j], -q / 4), ("ZZ", [i, j], q / 4)]
    op = SparsePauliOp.from_sparse_list(terms, num_qubits=n).simplify()
    scale = np.abs(op.coeffs).max()          # normalise so angles live in a sane range
    return op / scale, const, scale


def bits_to_x(bitstr, n):
    return np.array([int(b) for b in bitstr[::-1][:n]])   # qiskit is little-endian


def brute_force(prob):
    xs = [np.array(b) for b in itertools.product([0, 1], repeat=prob["n"])]
    best = min(xs, key=prob["score"])
    return best, [x for x in xs if prob["score"](x) <= prob["score"](best) + 1e-9]


# ------------------------------------------------------------ run
def fast_qaoa_state(diag, n, theta, reps):
    """Noiseless QAOA state in plain numpy. The cost layer is diagonal, and the
    mixer is an RX(2*beta) on every qubit. Matches QAOAAnsatz parameter order
    [beta_0..beta_{p-1}, gamma_0..gamma_{p-1}]."""
    betas, gammas = theta[:reps], theta[reps:]
    psi = np.full(2 ** n, 2 ** (-n / 2), dtype=complex)
    for b, g in zip(betas, gammas):
        psi = psi * np.exp(-1j * g * diag)
        c, s = np.cos(b), -1j * np.sin(b)
        psi = psi.reshape([2] * n)
        for q in range(n):
            ax = n - 1 - q                       # qubit q is bit q (little-endian)
            a0, a1 = np.take(psi, 0, axis=ax), np.take(psi, 1, axis=ax)
            psi = np.stack([c * a0 + s * a1, s * a0 + c * a1], axis=ax)
        psi = psi.reshape(-1)
    return psi


def tune_angles(ansatz, H, reps, seed=0):
    """Classical step: find QAOA angles on a noiseless simulator."""
    n = H.num_qubits
    diag = np.real(H.to_matrix(sparse=True).diagonal())

    def energy(theta):
        psi = fast_qaoa_state(diag, n, theta, reps)
        return float(np.real(np.vdot(psi, diag * psi)))

    # sanity check once: the fast simulator agrees with Qiskit's own
    t = np.random.default_rng(1).uniform(0, np.pi, 2 * reps)
    ref = float(np.real(Statevector(ansatz.assign_parameters(t)).expectation_value(H)))
    assert abs(ref - energy(t)) < 1e-6, (ref, energy(t))
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(6):
        x0 = rng.uniform(0, np.pi, 2 * reps)
        r = minimize(energy, x0, method="COBYLA", options=dict(maxiter=250))
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def analyse(prob, counts, opt_set):
    shots = sum(counts.values())
    opt_keys = {tuple(x) for x in opt_set}
    p_opt = sum(c for b, c in counts.items() if tuple(bits_to_x(b, prob["n"])) in opt_keys) / shots
    top = max(counts, key=counts.get)
    best_seen = min(counts, key=lambda b: prob["score"](bits_to_x(b, prob["n"])))
    return dict(shots=shots,
                p_best_answer=round(p_opt, 4),
                random_guess_p=round(len(opt_set) / 2 ** prob["n"], 4),
                most_frequent=prob["describe"](bits_to_x(top, prob["n"])),
                most_frequent_p=round(counts[top] / shots, 4),
                best_answer_found=prob["describe"](bits_to_x(best_seen, prob["n"])),
                best_answer_real=prob["real"](bits_to_x(best_seen, prob["n"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sim", "qpu"], default="sim")
    ap.add_argument("--save-key", action="store_true")
    ap.add_argument("--reps", type=int, default=2, help="QAOA layers")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--only", default=None, help="parks4 or trucks3")
    ap.add_argument("--backend", default=None, help="IBM machine name (default: least busy)")
    a = ap.parse_args()

    if a.save_key:
        from getpass import getpass
        from qiskit_ibm_runtime import QiskitRuntimeService
        QiskitRuntimeService.save_account(channel="ibm_quantum_platform",
                                          token=getpass("IBM API key (hidden): ").strip(),
                                          instance=input("Instance CRN or name: ").strip(),
                                          overwrite=True, set_as_default=True)
        print("Saved to ~/.qiskit/qiskit-ibm.json")
        return

    if a.mode == "qpu":
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        svc = QiskitRuntimeService()
        backend = svc.backend(a.backend) if a.backend else svc.least_busy(operational=True, simulator=False)
        print(f"Real QPU: {backend.name} ({backend.num_qubits} qubits)")
    else:
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import SamplerV2
        backend = AerSimulator()

    out = dict(mode=a.mode, backend=backend.name, reps=a.reps, problems={})
    for prob in [p for p in (parks4(), trucks3()) if a.only in (None, p["name"])]:
        print(f"\n=== {prob['name']} ({prob['n']} qubits) ===")
        best, opt_set = brute_force(prob)
        print("Known best answer:", prob["describe"](best), prob["real"](best))

        c0, lin, Q = to_qubo(prob)
        H, const, scale = hamiltonian(prob["n"], c0, lin, Q)
        ansatz = QAOAAnsatz(H, reps=a.reps)
        t0 = time.perf_counter()
        theta = tune_angles(ansatz, H, a.reps)
        tune_s = time.perf_counter() - t0

        # Noiseless expectation: what a perfect quantum computer would give
        psi = fast_qaoa_state(np.real(H.to_matrix(sparse=True).diagonal()), prob["n"], theta, a.reps)
        ideal = {format(i, f"0{prob['n']}b"): p for i, p in enumerate(np.abs(psi) ** 2) if p > 1e-12}
        ideal_p = sum(p for b, p in ideal.items()
                      if tuple(bits_to_x(b, prob["n"])) in {tuple(x) for x in opt_set})

        circ = ansatz.assign_parameters(theta)
        circ.measure_all()
        isa = generate_preset_pass_manager(optimization_level=3, backend=backend).run(circ)
        sampler = SamplerV2(mode=backend) if a.mode == "qpu" else SamplerV2()
        t0 = time.perf_counter()
        job = sampler.run([isa], shots=a.shots)
        print("Job submitted:", job.job_id() if hasattr(job, "job_id") else "local")
        res = job.result()
        wall = time.perf_counter() - t0
        counts = res[0].data.meas.get_counts()

        r = dict(qubits=prob["n"], qubo_pairs=len(Q),
                 known_best=prob["describe"](best),
                 circuit_depth_on_chip=isa.depth(),
                 two_qubit_gates_on_chip=sum(v for k, v in isa.count_ops().items() if k in ("cz", "ecr", "cx")),
                 angle_tuning_s=round(tune_s, 1),
                 perfect_qc_p_best=round(ideal_p, 4),
                 wall_seconds=round(wall, 1),
                 **analyse(prob, counts, opt_set))
        if a.mode == "qpu":
            r["job_id"] = job.job_id()
            try:
                r["qpu_seconds"] = job.usage()
            except Exception:
                pass
        out["problems"][prob["name"]] = r
        print(json.dumps(r, indent=2))

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", f"ibm_{a.mode}.json")
    json.dump(out, open(path, "w"), indent=2, default=str)
    print("\nSaved", path)


if __name__ == "__main__":
    main()
