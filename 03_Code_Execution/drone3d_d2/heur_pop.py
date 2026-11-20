"""Population baselines on the same cost model and solution encoding as heur.anneal (benchmark for the scale bridge).
Artificial bee colony (discrete variant: neighbour source = heur._neighbor) and a genetic algorithm (request-based crossover + the same mutation).
Objective E + PEN*max(0, makespan - T); trips infeasible in capacity/battery are discarded. Returns the best FEASIBLE (makespan <= T) solution seen, or None.
Budget = number of cost evaluations (calls of _cost), equal across methods. Written for this study; standard textbook algorithms, not the VB library of the authors."""
from __future__ import annotations
import random
from .analysis import _req_events
from .heur import _Eval, _cost, _neighbor, initial, _remove, _insert_events


class _Run:
    def __init__(self, ev_, T, budget):
        self.ev, self.T, self.budget, self.used = ev_, T, budget, 0
        self.best = None   # (E, mk, sol)

    def f(self, sol):
        """Evaluate; returns objective or None (infeasible trip). Tracks the best time-feasible solution."""
        self.used += 1
        c = _cost(self.ev, sol, self.T)
        if c is None:
            return None
        if c[2] <= self.T + 1e-9 and (self.best is None or c[1] < self.best[0]):
            self.best = (c[1], c[2], sol)
        return c[0]

    @property
    def done(self):
        return self.used >= self.budget


def _diversified_start(run, n, k, reqs, rng, moves):
    base = initial(run.ev, n, k)
    if base is None:
        return None
    sol = base
    for _ in range(moves):
        s2 = _neighbor(sol, rng, reqs, n, k)
        if s2 is not None:
            sol = s2
    return sol


def _population(run, n, k, reqs, rng, size):
    pop = []
    tries = 0
    while len(pop) < size and tries < 20 * size and not run.done:
        tries += 1
        s = _diversified_start(run, n, k, reqs, rng, 0 if not pop else rng.randint(1, 30))
        if s is None:
            return None
        o = run.f(s)
        if o is not None:
            pop.append([s, o])
    return pop


def abc(leg, reqs, dt, tm, k, T, budget=80000, seed=1001, SN=20, limit=50, evaluator=None):
    rng = random.Random(seed)
    ev_ = evaluator or _Eval(leg, reqs, dt, tm)
    n = len(reqs)
    run = _Run(ev_, T, budget)
    src = _population(run, n, k, reqs, rng, SN)
    if not src:
        return None
    trials = [0] * len(src)

    def move(i):
        s = _neighbor(src[i][0], rng, reqs, n, k)
        if s is not None and rng.random() < 0.3:
            s2 = _neighbor(s, rng, reqs, n, k)
            s = s2 if s2 is not None else s
        if s is None:
            return
        o = run.f(s)
        if o is not None and o < src[i][1]:
            src[i] = [s, o]; trials[i] = 0
        else:
            trials[i] += 1

    while not run.done:
        for i in range(len(src)):          # employed bees
            if run.done:
                break
            move(i)
        fmin = min(x[1] for x in src)
        w = [1.0 / (1.0 + (x[1] - fmin) / (0.05 * max(fmin, 1.0))) for x in src]
        for _ in range(len(src)):          # onlooker bees
            if run.done:
                break
            move(rng.choices(range(len(src)), weights=w)[0])
        for i in range(len(src)):          # scout bees
            if trials[i] > limit and not run.done:
                s = _diversified_start(run, n, k, reqs, rng, rng.randint(1, 30))
                if s is not None:
                    o = run.f(s)
                    if o is not None:
                        src[i] = [s, o]
                trials[i] = 0
    return run.best


def _cross(a, b, rng, reqs, n, k):
    ch = [[list(t) for t in p] for p in a]
    for r in range(n):
        if rng.random() >= 0.5:
            continue
        loc = None
        for d, p in enumerate(b):
            for i, t in enumerate(p):
                if any(e[0] == r for e in t):
                    loc = (d, i)
        evs = _req_events(reqs, r)
        ch = _remove(ch, r)
        d, i = loc
        if i < len(ch[d]):
            ch[d][i] = _insert_events(ch[d][i], evs, rng)
        else:
            ch[d].append(list(evs))
    return ch


def ga(leg, reqs, dt, tm, k, T, budget=80000, seed=1001, pop_size=40, pc=0.8, pm=0.4, elite=2, evaluator=None):
    rng = random.Random(seed)
    ev_ = evaluator or _Eval(leg, reqs, dt, tm)
    n = len(reqs)
    run = _Run(ev_, T, budget)
    pop = _population(run, n, k, reqs, rng, pop_size)
    if not pop:
        return None

    def tour():
        c = rng.sample(range(len(pop)), min(3, len(pop)))
        return pop[min(c, key=lambda i: pop[i][1])][0]

    while not run.done:
        pop.sort(key=lambda x: x[1])
        nxt = [list(x) for x in pop[:elite]]
        while len(nxt) < pop_size and not run.done:
            child = _cross(tour(), tour(), rng, reqs, n, k) if rng.random() < pc else [[list(t) for t in p] for p in tour()]
            if rng.random() < pm:
                m = _neighbor(child, rng, reqs, n, k)
                child = m if m is not None else child
            o = run.f(child)
            if o is not None:
                nxt.append([child, o])
        if len(nxt) < 2:
            break
        pop = nxt
    return run.best
