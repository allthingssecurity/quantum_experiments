# quantum_experiments

Five freight and industrial-corridor planning problems (park funding, truck backhaul, freight-corridor rakes, timetabling, multimodal routing), written as QUBOs and run with QAOA on IBM's 156-qubit Heron processors. Every hardware run is compared with an exact classical solver and with random guessing.

The full write-up, including how each problem becomes a circuit, all results, and the mistakes made along the way, is in [REPORT.md](REPORT.md).

Quick start:

```
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_ibm.py --save-key          # your own IBM Quantum API key
.venv/bin/python qaoa_suite.py --size small     # all five problems, 17 to 18 qubits
```

All problem data is synthetic and generated from fixed seeds.
