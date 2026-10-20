"""Figures for REPORT.md (saved to figures/).

  python make_figures.py

Needs IBM access only for the hardware histogram (fetches job dbb144hgv2cc73e7jld0).
"""
import itertools, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_ibm as RI
import qaoa_suite as S
import usecases as U

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
PARKS4_JOB = "dbb144hgv2cc73e7jld0"


def as_problem(prob):
    """Turn a run_ibm problem (score function) into the dict form qaoa_suite uses."""
    c0, lin, Q = RI.to_qubo(prob)
    return dict(name=prob["name"], n=prob["n"], labels=prob["labels"], lin=np.array(lin, float),
                quad=dict(Q), const=float(c0), decode=None, meta={})


def qubo_table(prob):
    c0, lin, Q = RI.to_qubo(prob)
    print(f"QUBO for {prob['name']}: constant {c0:g}")
    for i, l in enumerate(prob["labels"]):
        print(f"  switch {l}: {lin[i]:g}")
    for (i, j), q in Q.items():
        print(f"  pair {prob['labels'][i]}-{prob['labels'][j]}: {q:g}")


def draw_logical(prob, fname, title):
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    p = as_problem(prob)
    h, J, _ = S.ising(p)
    reps = 2                                   # the warm-up hardware runs used 2 layers
    b, g = ParameterVector("β", reps), ParameterVector("γ", reps)
    from qiskit import QuantumRegister
    qc = QuantumCircuit(*[QuantumRegister(1, l) for l in prob["labels"]])   # one named wire per switch
    qc.h(range(p["n"]))
    for k in range(reps):
        qc.barrier(label=f"cost {k+1}")
        for (i, j), c in J.items():
            qc.rzz(round(2 * c, 2) * g[k], i, j)
        for i in range(p["n"]):
            if abs(h[i]) > 1e-12:
                qc.rz(round(2 * h[i], 2) * g[k], i)
        qc.barrier(label=f"mix {k+1}")
        qc.rx(2 * b[k], range(p["n"]))
    qc.measure_all()
    fig = qc.draw("mpl", fold=-1, idle_wires=False, style="bw")
    fig.suptitle(title, fontsize=12)
    fig.savefig(os.path.join(FIG, fname), dpi=130, bbox_inches="tight")
    plt.close(fig)


def parks4_histogram():
    prob = RI.parks4()
    p = as_problem(prob)
    Q = S.qmat(p)
    # what a perfect quantum computer would return with the tuned angles
    c0, lin, Qd = RI.to_qubo(prob)
    H, const, scale = RI.hamiltonian(4, c0, lin, Qd)
    from qiskit.circuit.library import QAOAAnsatz
    theta = RI.tune_angles(QAOAAnsatz(H, reps=2), H, 2)
    diag = np.real(H.to_matrix(sparse=True).diagonal())
    ideal = np.abs(RI.fast_qaoa_state(diag, 4, theta, 2)) ** 2

    from qiskit_ibm_runtime import QiskitRuntimeService
    counts = QiskitRuntimeService(instance="open-instance").job(PARKS4_JOB).result()[0].data.meas.get_counts()
    shots = sum(counts.values())

    names = "ABCD"
    rows = []
    for k in range(16):
        x = np.array([(k >> i) & 1 for i in range(4)])
        bits = "".join(str(b) for b in x[::-1])          # qiskit key
        label = "+".join(n for n, b in zip(names, x) if b) or "none"
        benefit, ok = prob["real"](x)
        rows.append((label, counts.get(bits, 0) / shots, ideal[k], ok, benefit))
    rows.sort(key=lambda r: -r[1])

    fig, ax = plt.subplots(figsize=(10, 4.2))
    xs = np.arange(len(rows))
    ax.bar(xs - 0.2, [r[1] * 100 for r in rows], 0.4, label="real IBM hardware (ibm_marrakesh)", color="#4C72B0")
    ax.bar(xs + 0.2, [r[2] * 100 for r in rows], 0.4, label="perfect quantum computer (simulated)", color="#BBBBBB")
    ax.axhline(100 / 16, ls="--", color="#C44E52", lw=1, label="random guess (6.25%)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r[0]}\n{'ok' if r[3] else 'over'}" for r in rows], fontsize=7)
    best = [i for i, r in enumerate(rows) if r[0] == "B+C+D"][0]
    ax.get_xticklabels()[best].set_color("#2a7a2a")
    ax.get_xticklabels()[best].set_fontweight("bold")
    ax.set_ylabel("% of 4,096 shots")
    ax.set_title("Four parks, budget 100 crore: which answers came back (best = B+C+D)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "parks4_hardware_histogram.png"), dpi=130)
    plt.close(fig)
    return rows


def gates_vs_quality():
    names = {"1_bhavya": "Park funding", "2_trucks": "Trucks", "3_dfc_rakes": "Rakes",
             "4_timetable": "Timetabling", "5_multimodal": "Multimodal"}
    marker = {"small": "o", "medium": "s", "large": "^"}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for size in ("small", "medium", "large"):
        d = json.load(open(os.path.join(HERE, "results", f"suite_{size}_p1.json")))
        for k, r in d.items():
            g = r["circuit"]["two_qubit_gates"]
            v = r["qpu"]["feasible_pct"]
            ax.scatter(g, max(v, 0.01), marker=marker[size], s=50, color="#4C72B0")
            dy = -9 if k == "3_dfc_rakes" and size == "small" else 3
            ax.annotate(f"{names[k]} ({r['qubits']}q)", (g, max(v, 0.01)), fontsize=7,
                        xytext=(4, dy), textcoords="offset points")
    for size, m in marker.items():
        ax.scatter([], [], marker=m, color="#4C72B0", label=f"{size} size")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("two-qubit gates after compiling for the chip (log scale)")
    ax.set_ylabel("% of quantum answers that break no rule (log scale)")
    ax.axhline(0.01, color="#999999", lw=0.6, ls=":")
    ax.text(105, 0.0125, "points on this line: no valid answer in 4,096 shots", fontsize=7, color="#666666")
    ax.set_title("Longer circuits, fewer valid answers")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "gates_vs_valid_answers.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    qubo_table(RI.parks4())
    draw_logical(RI.parks4(), "circuit_parks4.png", "Park funding, 4 parks: every pair of qubits is linked by the budget rule")
    draw_logical(RI.trucks3(), "circuit_trucks3.png", "Trucks, 3 x 3: only clashing pairs are linked")
    rows = parks4_histogram()
    print("hardware vs perfect, top 5:", [(r[0], round(100 * r[1], 1), round(100 * r[2], 1)) for r in rows[:5]])
    gates_vs_quality()
    print("figures written to", FIG)
