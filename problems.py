"""The two PoC problems, built once and handed to every solver.

All data here is synthetic (no government data leaves the machine).
Each builder returns a ConstrainedQuadraticModel (switches + goal + rules)
plus the raw data the classical exact solver needs.
"""
import random
from dimod import ConstrainedQuadraticModel, BinaryQuadraticModel, Binary, quicksum


# ---------------------------------------------------------------- BHAVYA
def bhavya(n_parks=100, n_corridors=11, seed=7):
    """Pick which industrial parks to fund under one budget.

    Goal: maximise total benefit.
    Rules: total cost <= budget.
    Twist (what makes it quadratic): two funded parks on the same corridor
    compete for the same port/rail capacity, so their combined benefit is
    reduced by a 'competition' amount.
    Costs are in units of Rs 5 Cr to keep numbers small.
    """
    rng = random.Random(seed)
    parks = [f"P{i:03d}" for i in range(n_parks)]
    corridor = {p: rng.randrange(n_corridors) for p in parks}
    cost = {p: rng.randint(4, 40) for p in parks}                       # 20..200 Cr
    benefit = {p: round(cost[p] * rng.uniform(0.6, 1.4)) + rng.randint(0, 5) for p in parks}
    budget = int(0.35 * sum(cost.values()))

    competition = {}
    for i, a in enumerate(parks):
        for b in parks[i + 1:]:
            if corridor[a] == corridor[b] and rng.random() < 0.5:
                competition[(a, b)] = rng.randint(2, 10)

    x = {p: Binary(p) for p in parks}
    cqm = ConstrainedQuadraticModel()
    cqm.set_objective(
        -quicksum(benefit[p] * x[p] for p in parks)
        + quicksum(c * x[a] * x[b] for (a, b), c in competition.items())
    )
    cqm.add_constraint(quicksum(cost[p] * x[p] for p in parks) <= budget, label="budget")

    data = dict(parks=parks, cost=cost, benefit=benefit, budget=budget,
                competition=competition, corridor=corridor)
    return cqm, data


def bhavya_tiny():
    """The 4-park slide example: budget 100 Cr, answer B+C+D (benefit 18)."""
    cost = {"A": 40, "B": 30, "C": 50, "D": 20}
    benefit = {"A": 8, "B": 5, "C": 9, "D": 4}
    x = {p: Binary(p) for p in cost}
    cqm = ConstrainedQuadraticModel()
    cqm.set_objective(-quicksum(benefit[p] * x[p] for p in cost))
    cqm.add_constraint(quicksum(cost[p] * x[p] for p in cost) <= 100, label="budget")
    data = dict(parks=list(cost), cost=cost, benefit=benefit, budget=100, competition={}, corridor={})
    return cqm, data


def bhavya_value(data, chosen):
    """Net benefit of a set of funded parks, and whether it is within budget."""
    s = set(chosen)
    val = sum(data["benefit"][p] for p in s)
    val -= sum(c for (a, b), c in data["competition"].items() if a in s and b in s)
    spend = sum(data["cost"][p] for p in s)
    return val, spend <= data["budget"]


# ---------------------------------------------------------------- Trucks
TRUCK_TYPES = ["open", "container", "reefer"]


def trucks(n_trucks=20, n_loads=20, radius_km=60, max_options=6, seed=11):
    """Match empty trucks at a hub to waiting return loads.

    Switch x[t,l] = 1 means truck t drives empty to load l and carries it.
    Goal: carry as many loads as possible, with as few empty km as possible
          (each matched load is worth REWARD, each empty km costs 1).
    Rules: a truck takes at most one load; a load gets at most one truck.
    Only sensible pairs are switches: same truck type, within radius_km,
    and each truck keeps its max_options nearest loads.
    """
    REWARD = 100
    rng = random.Random(seed)
    pt = lambda: (rng.uniform(0, 100), rng.uniform(0, 100))             # 100x100 km region
    T = {f"T{i:02d}": dict(pos=pt(), kind=rng.choice(TRUCK_TYPES)) for i in range(n_trucks)}
    L = {f"L{j:02d}": dict(pos=pt(), kind=rng.choice(TRUCK_TYPES)) for j in range(n_loads)}
    dist = lambda a, b: round(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5, 1)

    pairs = {}
    for t, tv in T.items():
        opts = [(dist(tv["pos"], lv["pos"]), l) for l, lv in L.items()
                if lv["kind"] == tv["kind"] and dist(tv["pos"], lv["pos"]) <= radius_km]
        for km, l in sorted(opts)[:max_options]:
            pairs[(t, l)] = km

    x = {p: Binary(f"{p[0]}-{p[1]}") for p in pairs}
    cqm = ConstrainedQuadraticModel()
    cqm.set_objective(quicksum((km - REWARD) * x[p] for p, km in pairs.items()))
    for t in T:
        mine = [x[p] for p in pairs if p[0] == t]
        if len(mine) > 1:
            cqm.add_constraint(quicksum(mine) <= 1, label=f"truck_{t}")
    for l in L:
        mine = [x[p] for p in pairs if p[1] == l]
        if len(mine) > 1:
            cqm.add_constraint(quicksum(mine) <= 1, label=f"load_{l}")

    data = dict(trucks=T, loads=L, pairs=pairs, reward=REWARD)
    return cqm, data


def trucks_bqm(data, penalty=None):
    """Direct QUBO for trucks, written by hand: no slack switches needed,
    because 'at most one' is just a pair penalty between clashing switches.
    This keeps the QPU problem small and sparse."""
    pairs, R = data["pairs"], data["reward"]
    penalty = penalty or 2 * R
    bqm = BinaryQuadraticModel("BINARY")
    names = {p: f"{p[0]}-{p[1]}" for p in pairs}
    for p, km in pairs.items():
        bqm.add_variable(names[p], km - R)
    keys = list(pairs)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if a[0] == b[0] or a[1] == b[1]:
                bqm.add_interaction(names[a], names[b], penalty)
    return bqm


def trucks_value(data, chosen_pairs):
    """(loads matched, empty km, feasible?) for a set of (truck, load) pairs."""
    ts = [p[0] for p in chosen_pairs]
    ls = [p[1] for p in chosen_pairs]
    feasible = len(ts) == len(set(ts)) and len(ls) == len(set(ls))
    km = sum(data["pairs"][p] for p in chosen_pairs)
    return len(chosen_pairs), round(km, 1), feasible
