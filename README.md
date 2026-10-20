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

1. Classical methods won at every size. They found the best answer, or came within a few percent of it, in well under a second to a few seconds, and nearly all of their answers were valid.
2. At 17 to 18 qubits the quantum computer did real work. It found the best answer for two of the five problems and came within 1% to 7% on the other three. It reached 73% to 88% of what a perfect, noise-free quantum computer would get with the same circuit.
3. At 25 to 45 qubits its best answers were 8% to 29% worse than classical (six times worse on timetabling), and under 1% of its answers were valid for most problems.
4. At 90 to 120 qubits it did not return a single valid answer in 4,096 tries on any problem.
5. The main limit is noise. On this chip, problems where many decisions affect each other need long circuits, and each extra two-qubit gate adds error.

What this means: for problems of this size there is no case for using quantum hardware today. What we have now is the five problems written in a form quantum machines accept, a measured picture of where current hardware stops, and code that can be rerun as hardware improves. The next sensible step is to run the same problems on a quantum annealer (D-Wave), which is built for this kind of problem and handles far more decisions at once.

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
