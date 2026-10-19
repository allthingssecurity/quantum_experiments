"""The 5 NICDC use cases as QUBOs, sized by a 'scale' knob.

Each builder returns a dict:
  name, n (qubits), labels, lin (n,), quad {(i,j): q}, const,
  decode(x) -> dict of real-world numbers incl. 'feasible'
All data synthetic but shaped like the real problem.
"""
import itertools, random
import numpy as np


class QB:
    """Tiny QUBO builder: add linear/quadratic terms and squared penalties."""
    def __init__(self, labels):
        self.labels = labels
        self.idx = {l: i for i, l in enumerate(labels)}
        self.n = len(labels)
        self.lin = np.zeros(self.n)
        self.quad = {}
        self.const = 0.0

    def L(self, a, c):
        self.lin[self.idx[a] if not isinstance(a, int) else a] += c

    def Q(self, a, b, c):
        i = self.idx[a] if not isinstance(a, int) else a
        j = self.idx[b] if not isinstance(b, int) else b
        if i == j:
            self.lin[i] += c
            return
        k = (min(i, j), max(i, j))
        self.quad[k] = self.quad.get(k, 0.0) + c

    def square(self, terms, target, weight):
        """weight * (sum c_k x_k - target)^2  (x^2 = x for binaries)"""
        self.const += weight * target ** 2
        for a, c in terms:
            self.L(a, weight * (c * c - 2 * target * c))
        for (a, ca), (b, cb) in itertools.combinations(terms, 2):
            self.Q(a, b, weight * 2 * ca * cb)

    def at_most_one(self, vs, weight):
        for a, b in itertools.combinations(vs, 2):
            self.Q(a, b, weight)

    def out(self, name, decode, **meta):
        return dict(name=name, n=self.n, labels=self.labels, lin=self.lin,
                    quad={k: v for k, v in self.quad.items() if abs(v) > 1e-12},
                    const=self.const, decode=decode, meta=meta)


# 1 ------------------------------------------------------------ BHAVYA
def bhavya(n_parks=20, seed=3):
    """Fund parks under a budget. Dense: the budget links every pair of parks.
    Budget rule via binary slack switches: (spend + slack - budget)^2."""
    rng = random.Random(seed)
    parks = [f"P{i:02d}" for i in range(n_parks)]
    cost = {p: rng.randint(1, 6) for p in parks}
    ben = {p: cost[p] + rng.randint(-1, 3) for p in parks}
    corridor = {p: rng.randrange(4) for p in parks}
    comp = {(a, b): rng.randint(1, 2) for a, b in itertools.combinations(parks, 2)
            if corridor[a] == corridor[b] and rng.random() < 0.4}
    budget = int(0.4 * sum(cost.values()))
    nslack = int(np.ceil(np.log2(budget + 1)))
    slack = [f"s{k}" for k in range(nslack)]
    q = QB(parks + slack)
    for p in parks:
        q.L(p, -ben[p])
    for (a, b), c in comp.items():
        q.Q(a, b, c)
    lam = max(ben.values()) + 1
    q.square([(p, cost[p]) for p in parks] + [(s, 2 ** k) for k, s in enumerate(slack)], budget, lam)

    def decode(x):
        chosen = [p for p in parks if x[q.idx[p]]]
        spend = sum(cost[p] for p in chosen)
        val = sum(ben[p] for p in chosen) - sum(c for (a, b), c in comp.items() if a in chosen and b in chosen)
        return dict(feasible=spend <= budget, benefit=val, spend=spend, budget=budget, parks_funded=len(chosen))
    return q.out("1_bhavya", decode, parks=n_parks, slack_qubits=nslack, metric="benefit", better="higher")


# 2 ------------------------------------------------------------ trucks
def trucks(n=15, options=3, seed=5):
    """Empty trucks -> return loads at one hub. Sparse: only nearby, same-type pairs."""
    rng = random.Random(seed)
    T = [(rng.uniform(0, 80), rng.uniform(0, 80)) for _ in range(n)]
    Lp = [(rng.uniform(0, 80), rng.uniform(0, 80)) for _ in range(n)]
    d = lambda a, b: round(np.hypot(a[0] - b[0], a[1] - b[1]), 1)
    pairs = {}
    for t in range(n):
        for l in sorted(range(n), key=lambda l: d(T[t], Lp[l]))[:options]:
            pairs[(t, l)] = d(T[t], Lp[l])
    R = 100
    labels = [f"T{t}L{l}" for t, l in pairs]
    q = QB(labels)
    keys = list(pairs)
    for i, p in enumerate(keys):
        q.L(i, pairs[p] - R)
    for t in range(n):
        q.at_most_one([i for i, p in enumerate(keys) if p[0] == t], 2 * R)
    for l in range(n):
        q.at_most_one([i for i, p in enumerate(keys) if p[1] == l], 2 * R)

    def decode(x):
        on = [keys[i] for i in range(len(keys)) if x[i]]
        ok = len({p[0] for p in on}) == len(on) and len({p[1] for p in on}) == len(on)
        return dict(feasible=ok, loads_carried=len(on), empty_km=round(sum(pairs[p] for p in on), 1),
                    score=round(len(on) * R - sum(pairs[p] for p in on), 1))
    return q.out("2_trucks", decode, trucks=n, loads=n, metric="score", better="higher")


# 3 ------------------------------------------------------------ DFC rakes
def rakes(n_rakes=10, n_requests=10, options=3, seed=7):
    """Empty rakes on a DFC section -> loading requests at terminals.
    Extra rule vs trucks: a terminal can load only one rake per time slot."""
    rng = random.Random(seed)
    terminals = 4
    req = [dict(term=rng.randrange(terminals), slot=rng.randrange(3), tons=rng.choice([2500, 3000, 3600]))
           for _ in range(n_requests)]
    pos = [rng.uniform(0, 300) for _ in range(n_rakes)]           # km along the corridor
    tpos = [rng.uniform(0, 300) for _ in range(terminals)]
    pairs = {}
    for r in range(n_rakes):
        near = sorted(range(n_requests), key=lambda j: abs(pos[r] - tpos[req[j]["term"]]))[:options]
        for j in near:
            pairs[(r, j)] = round(abs(pos[r] - tpos[req[j]["term"]]), 1)
    keys = list(pairs)
    q = QB([f"R{r}Q{j}" for r, j in keys])
    rev = lambda j: req[j]["tons"] / 10                           # revenue proxy
    for i, (r, j) in enumerate(keys):
        q.L(i, pairs[(r, j)] - rev(j))
    P = 2 * max(rev(j) for j in range(n_requests))
    for r in range(n_rakes):
        q.at_most_one([i for i, k in enumerate(keys) if k[0] == r], P)
    for j in range(n_requests):
        q.at_most_one([i for i, k in enumerate(keys) if k[1] == j], P)
    for a, b in itertools.combinations(range(len(keys)), 2):     # terminal-slot clash
        ja, jb = keys[a][1], keys[b][1]
        if ja != jb and req[ja]["term"] == req[jb]["term"] and req[ja]["slot"] == req[jb]["slot"]:
            q.Q(a, b, P)

    def decode(x):
        on = [keys[i] for i in range(len(keys)) if x[i]]
        ok = len({k[0] for k in on}) == len(on) and len({k[1] for k in on}) == len(on)
        ts = [(req[j]["term"], req[j]["slot"]) for _, j in on]
        ok = ok and len(ts) == len(set(ts))
        return dict(feasible=ok, requests_served=len(on), empty_km=round(sum(pairs[k] for k in on), 1),
                    tons=sum(req[j]["tons"] for _, j in on),
                    score=round(sum(rev(j) - pairs[(r, j)] for r, j in on), 1))
    return q.out("3_dfc_rakes", decode, rakes=n_rakes, requests=n_requests, metric="score", better="higher")


# 4 ------------------------------------------------------------ timetabling
def timetable(n_trains=12, n_slots=3, headway=1, seed=9):
    """Freight trains on one DFC block section. Each train picks one departure
    slot (0, +10, +20 min). Two trains too close together = conflict.
    Goal: no conflicts, least total delay (priority trains weigh more)."""
    rng = random.Random(seed)
    want = sorted(rng.randrange(0, int(1.5 * n_trains)) for _ in range(n_trains))  # wished time (10-min units)
    prio = [rng.choice([1, 1, 2, 3]) for _ in range(n_trains)]
    labels = [f"t{t}s{s}" for t in range(n_trains) for s in range(n_slots)]
    q = QB(labels)
    v = lambda t, s: t * n_slots + s
    P = 3 * max(prio) * n_slots + 5
    for t in range(n_trains):
        for s in range(n_slots):
            q.L(v(t, s), prio[t] * s)                               # delay cost
        q.square([(v(t, s), 1) for s in range(n_slots)], 1, 2 * P) # exactly one slot
    for t1, t2 in itertools.combinations(range(n_trains), 2):
        for s1 in range(n_slots):
            for s2 in range(n_slots):
                if abs((want[t1] + s1) - (want[t2] + s2)) < headway:
                    q.Q(v(t1, s1), v(t2, s2), P)                    # headway conflict

    def decode(x):
        X = np.array(x).reshape(n_trains, n_slots)
        one = (X.sum(1) == 1).all()
        times = [want[t] + int(np.argmax(X[t])) for t in range(n_trains)]
        conflicts = sum(1 for a, b in itertools.combinations(range(n_trains), 2) if abs(times[a] - times[b]) < headway)
        delay = sum(10 * prio[t] * int(np.argmax(X[t])) for t in range(n_trains))
        return dict(feasible=bool(one and conflicts == 0), conflicts=int(conflicts) if one else None,
                    weighted_delay_min=delay)
    return q.out("4_timetable", decode, trains=n_trains, slots=n_slots, metric="weighted_delay_min", better="lower")


# 5 ------------------------------------------------------------ multimodal
def multimodal(n_ship=10, seed=13):
    """Each shipment picks a mode: road, rail, or rail+last-mile road.
    Cost = money + time. Shared rail path gets congested: every pair of
    shipments both on rail adds a congestion cost (quadratic)."""
    rng = random.Random(seed)
    modes = ["road", "rail", "rail+road"]
    labels = [f"S{i}{m}" for i in range(n_ship) for m in ("R", "L", "M")]
    q = QB(labels)
    v = lambda i, m: 3 * i + m
    costs = []
    for i in range(n_ship):
        km = rng.uniform(300, 1200)
        c = [round(km * 3.5 / 100, 1),                 # road: dearest per ton-km, fastest
             round(km * 1.0 / 100 + rng.uniform(2, 6), 1),   # rail: cheap, terminal handling
             round(km * 1.3 / 100 + rng.uniform(1, 4), 1)]   # rail + last-mile road
        costs.append(c)
        for m in range(3):
            q.L(v(i, m), c[m])
    cong = 1.5
    for i, j in itertools.combinations(range(n_ship), 2):
        for mi in (1, 2):
            for mj in (1, 2):
                q.Q(v(i, mi), v(j, mj), cong)
    P = max(max(c) for c in costs) + cong * n_ship
    for i in range(n_ship):
        q.square([(v(i, m), 1) for m in range(3)], 1, P)

    def decode(x):
        X = np.array(x).reshape(n_ship, 3)
        one = (X.sum(1) == 1).all()
        pick = X.argmax(1)
        on_rail = int((pick > 0).sum())
        total = sum(costs[i][pick[i]] for i in range(n_ship)) + cong * on_rail * (on_rail - 1) / 2
        return dict(feasible=bool(one), total_cost=round(float(total), 1),
                    road=int((pick == 0).sum()), rail=int((pick == 1).sum()), rail_road=int((pick == 2).sum()))
    return q.out("5_multimodal", decode, shipments=n_ship, metric="total_cost", better="lower")


SIZES = {
    "small":  dict(bhavya=dict(n_parks=12), trucks=dict(n=6), rakes=dict(n_rakes=6, n_requests=6),
                   timetable=dict(n_trains=6), multimodal=dict(n_ship=6)),
    "medium": dict(bhavya=dict(n_parks=20), trucks=dict(n=15), rakes=dict(n_rakes=12, n_requests=12),
                   timetable=dict(n_trains=12), multimodal=dict(n_ship=12)),
    "large":  dict(bhavya=dict(n_parks=30), trucks=dict(n=40), rakes=dict(n_rakes=30, n_requests=30),
                   timetable=dict(n_trains=35), multimodal=dict(n_ship=30)),
}
BUILDERS = dict(bhavya=bhavya, trucks=trucks, rakes=rakes, timetable=timetable, multimodal=multimodal)


def build_all(size):
    return [BUILDERS[k](**kw) for k, kw in SIZES[size].items()]
