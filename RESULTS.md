# Classical vs annealing: first numbers (19 Oct 2026)

All data synthetic. Same problem given to every solver.
Fixstars Amplify AE = GPU annealer (quantum-inspired, **not** quantum hardware).
Real D-Wave QPU: not yet run (Leap trial gives no API token).

## BHAVYA: fund parks under a budget (100 parks, 110 switches, ~6,000 pairs)

Higher benefit is better. 1,012 is proven best (classical solver, 300 s).

| Solver | Time | Benefit | % of best |
|---|---|---|---|
| Simple greedy rule | instant | 813 | 80% |
| Simulated annealing (laptop) | ~4 s | ~900 | 89% |
| **Classical exact (OR-Tools CP-SAT)** | 1 s | 988 | 98% |
| **Classical exact (OR-Tools CP-SAT)** | 10 s | **1,012** | 100% (proof of optimality takes 300 s) |
| **Amplify AE, same QUBO as D-Wave** | 1 s | 934 | 92% |
| **Amplify AE, same QUBO as D-Wave** | 10 s | 989 | 98% |
| **Amplify AE, native budget rule** | 1 s | **1,007** | 99.5% |
| **Amplify AE, native budget rule** | 10 s | 1,010 | 99.8% |

## Trucks: empty trucks to return loads (20 trucks, 20 loads, 86 switches)

| Solver | Time | Loads carried | Empty km |
|---|---|---|---|
| **Classical exact (CP-SAT)** | 0.003 s | 18 | **403.3** (proven best) |
| Simulated annealing (laptop) | 0.9 s | 18 | 411.5 |
| **Amplify AE** | 1 s | 18 | **403.3** (best) |

## Does it fit a real D-Wave chip? (offline check)

| Problem | Physical qubits | Longest chain |
|---|---|---|
| BHAVYA | ~1,400 | 17-21 |
| Trucks | ~160-190 | 3-4 |

## Takeaways

1. Annealing-style solvers find best or near-best answers (99.5%+ for BHAVYA, exact for trucks).
2. A good classical solver is still as good or better, and faster on the small truck problem.
3. How the rule is written matters: letting Amplify handle the budget natively beat our hand-built QUBO (1,007 vs 934 at 1 s).
4. Next: real QPU numbers (D-Wave LaunchPad, or IBM QAOA on the 4-park example).
