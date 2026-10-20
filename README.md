# quantum_experiments

Five freight and industrial-corridor planning problems, run on a real IBM quantum computer and compared with ordinary classical methods.

## Executive summary

We wanted to know whether today's quantum hardware can help with the kind of planning problems that come up in Indian freight and industrial-corridor work. We picked five:

- deciding which industrial parks to fund under a fixed budget
- matching empty trucks at a hub to return loads
- assigning empty rakes on a dedicated freight corridor to loading requests
- giving freight trains conflict-free departure slots on one block section
- choosing road, rail or rail plus road for a set of shipments

Each problem was written as a list of yes/no decisions with a score (a QUBO), turned into a QAOA circuit, and run on IBM's 156-qubit Heron processors (`ibm_marrakesh` and `ibm_kingston`) on the free plan. We ran three sizes, from 17 to 120 qubits. Every quantum run was compared with an exact classical solver (Google OR-Tools) and with a classical sampler (simulated annealing on a laptop) that was given the same 4,096 samples as the quantum computer.

What we found:

1. All five problems ran end to end on real quantum hardware. Each one was written in a form a quantum computer accepts, compiled for the chip and executed in about 3 seconds of quantum time.
2. At 17 to 18 qubits the results are strong. The quantum computer found the exact best answer for two of the five problems (rake assignment and timetabling) and came within 1% to 7% on the other three. It reached 73% to 88% of what a perfect, noise-free quantum computer would achieve with the same circuit, so at this size the hardware is already close to what a noise-free machine would deliver.
3. At 25 to 45 qubits the quantum computer still moved its answers well toward the best solution, but valid answers became rarer and the best answers were typically 8% to 29% from optimal.
4. At 90 to 120 qubits the problems ran on the hardware, but noise kept answers from fully satisfying all the rules. This marks where today's processors reach their current limit for these problems.
5. Classical solvers were the reference point throughout and, at these sizes, remain faster and more accurate. That is expected for current hardware and problem sizes, and it gives a clear baseline to measure future quantum hardware against.

What this means: the five corridor problems are now quantum-ready, with formulations, circuits and a tested pipeline in place. We have measured where today's hardware performs well (around 20 qubits) and what limits it beyond that (circuit depth and noise). Both of these improve with each hardware generation, so the same code can be rerun as processors get larger and cleaner. The natural next steps are to run the same problems on a quantum annealer (D-Wave), which handles thousands of decisions at once, and to try deeper QAOA circuits with error mitigation on IBM hardware.

All data is synthetic, generated from fixed seeds. The whole experiment used 153 seconds of IBM quantum time.

## More detail

The full write-up is in [REPORT.md](REPORT.md). It covers each use case, how a problem becomes a circuit, how the runs were done, all result tables, a mistake we made along the way and how we caught it, and the IBM job ids for every run.

## Running it yourself

```
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_ibm.py --save-key          # your own IBM Quantum API key
.venv/bin/python qaoa_suite.py --size small     # all five problems, 17 to 18 qubits
.venv/bin/python compare_classical.py           # classical sampler on the same problems
.venv/bin/python make_tables.py                 # result tables
```
