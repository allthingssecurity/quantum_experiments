"""Same PoC problems on Fixstars Amplify AE (GPU annealer, quantum-inspired).

  python run_amplify.py            # time limits 1 s and 10 s
Token is read from .amplify_tokens (never committed).
"""
import datetime, json, os, time

from amplify import (VariableGenerator, Model, solve, AmplifyAEClient,
                     less_equal, Poly)

import problems as P
import run as R

HERE = os.path.dirname(os.path.abspath(__file__))


def client(ms):
    tok = dict(l.strip().split("=", 1) for l in open(os.path.join(HERE, ".amplify_tokens")) if "=" in l)
    c = AmplifyAEClient()
    c.token = tok["AMPLIFY_AE_TOKEN"]
    c.parameters.time_limit_ms = datetime.timedelta(milliseconds=ms)
    return c


def bqm_to_amplify(bqm):
    """The exact same QUBO the D-Wave path uses, re-expressed for Amplify."""
    names = list(bqm.variables)
    q = VariableGenerator().array("Binary", len(names))
    idx = {v: i for i, v in enumerate(names)}
    f = Poly(bqm.offset)
    for v, h in bqm.linear.items():
        f += h * q[idx[v]]
    for (u, v), j in bqm.quadratic.items():
        f += j * q[idx[u]] * q[idx[v]]
    return Model(f), q, names


def run_qubo(bqm, ms):
    model, q, names = bqm_to_amplify(bqm)
    t0 = time.perf_counter()
    res = solve(model, client(ms))
    wall = time.perf_counter() - t0
    vals = q.evaluate(res.best.values)
    sample = {names[i]: int(round(vals[i])) for i in range(len(names))}
    return sample, dict(wall_seconds=round(wall, 2),
                        solver_seconds=round(res.execution_time.total_seconds(), 3))


def bhavya_native(data, ms):
    """Amplify's own constraint handling (it tunes the penalty itself)."""
    parks = data["parks"]
    x = VariableGenerator().array("Binary", len(parks))
    i = {p: k for k, p in enumerate(parks)}
    obj = -sum(data["benefit"][p] * x[i[p]] for p in parks)
    obj += sum(c * x[i[a]] * x[i[b]] for (a, b), c in data["competition"].items())
    con = less_equal(sum(data["cost"][p] * x[i[p]] for p in parks), data["budget"])
    t0 = time.perf_counter()
    res = solve(Model(obj, con), client(ms))
    wall = time.perf_counter() - t0
    vals = x.evaluate(res.best.values)
    chosen = [p for p in parks if round(vals[i[p]])]
    val, ok = P.bhavya_value(data, chosen)
    return dict(value=val, feasible=ok, wall_seconds=round(wall, 2),
                solver_seconds=round(res.execution_time.total_seconds(), 3))


def main():
    out = {}
    for ms in (1000, 10000):
        tag = f"{ms // 1000}s"
        # BHAVYA, same QUBO as D-Wave path
        cqm, data = P.bhavya()
        bqm, invert, _ = R.bhavya_bqm(cqm, data)
        s, info = run_qubo(bqm, ms)
        chosen = [p for p in data["parks"] if invert(s).get(p)]
        val, ok = P.bhavya_value(data, chosen)
        rep, _ = P.bhavya_value(data, R.bhavya_repair(data, chosen))
        out[f"bhavya_qubo_{tag}"] = dict(value=val, feasible=ok, repaired=rep, **info)
        out[f"bhavya_native_{tag}"] = bhavya_native(data, ms)
        # Trucks, same QUBO as D-Wave path
        _, td = P.trucks()
        s, info = run_qubo(P.trucks_bqm(td), ms)
        names = {f"{p[0]}-{p[1]}": p for p in td["pairs"]}
        n, km, ok = P.trucks_value(td, [names[v] for v, on in s.items() if on])
        out[f"trucks_{tag}"] = dict(loads=n, empty_km=km, feasible=ok, **info)
        print(json.dumps({k: v for k, v in out.items() if k.endswith(tag)}, indent=2))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "amplify_ae.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
