"""Summaries of phaseA_rows_<tag>.csv and phaseA_rules_<tag>.csv (SYNTHETIC instances). Usage: python analyze_phaseA.py <tag>"""
import sys, os, csv, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
tag = sys.argv[1]
rng = np.random.default_rng(20261113)


def load(fn):
    with open(os.path.join(HERE, fn), encoding="utf-8") as f:
        return list(csv.DictReader(f))


rows = load(f"phaseA_rows_{tag}.csv")
rules = load(f"phaseA_rules_{tag}.csv")
fl = lambda x: None if x == "" else float(x)
D = collections.defaultdict(dict)       # (cfg, seed, k, T) -> (E_min, E_at_T, Tmin)
seeds = sorted({int(r["seed"]) for r in rows})
for r in rows:
    if r["T"] == "":
        continue
    D[(r["cfg"], int(r["seed"]), int(r["k"]), int(r["T"]))] = (fl(r["E_min"]), fl(r["E_at_T"]), fl(r["Tmin"]))
KS = (1, 2, 3, 4)
TS = (900, 1200, 1800, 2700, 3600)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def boot_med(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    b = [np.median(rng.choice(x, len(x))) for _ in range(10000)]
    return (np.median(x), np.percentile(b, 2.5), np.percentile(b, 97.5))


def prem(cfg, seed, k, T):
    v = D.get((cfg, seed, k, T))
    if not v or v[1] is None:
        return None
    return v[1] / v[0] - 1.0


def kstar(cfg, seed, T):
    for k in KS:
        if prem(cfg, seed, k, T) is not None:
            return k
    return 5   # > KMAX


out = []
w = out.append
w(f"# PHASE A (SENTETİK, {tag}): özet\nSentetik örnekler (gerçek şehir verisi DEĞİL); {len(seeds)} bağımsız çekim; tek tip M600 Pro, s/τ taban 60/180 s aksi belirtilmedikçe.\n")

# 1. base premium by (T,k)
w("## 1. Taban: enerji primi π(T;k) = E*/E_min − 1 (medyan [%95 bootstrap GA], pratik eşik %1 üstü oran)\n")
w("| T (s) | k | uygun oran [Wilson] | uygun örneklerde medyan π % [GA] | π>0,5% / >1% / >2% (uygun içinde) | tüm çekimlerde π>1% oranı [Wilson] |\n|---|---|---|---|---|---|")
for T in TS:
    for k in KS:
        vals = [prem("s60_tau180", s, k, T) for s in seeds]
        feas = [v for v in vals if v is not None]
        lo_f, hi_f = wilson(len(feas), len(seeds))
        if not feas:
            w(f"| {T} | {k} | 0,00 [{lo_f:.2f}, {hi_f:.2f}] | — | — | — |"); continue
        m, lo, hi = boot_med(feas)
        n1 = sum(v > 0.01 for v in feas); l1, h1 = wilson(n1, len(seeds))
        w(f"| {T} | {k} | {len(feas)/len(seeds):.2f} [{lo_f:.2f}, {hi_f:.2f}] | {100*m:.2f} [{100*lo:.2f}, {100*hi:.2f}] | {np.mean([v > 0.005 for v in feas]):.2f} / {np.mean([v > 0.01 for v in feas]):.2f} / {np.mean([v > 0.02 for v in feas]):.2f} | {n1/len(seeds):.2f} [{l1:.2f}, {h1:.2f}] |")

w("\n## 2. k*(T): en küçük yeterli filo (tüm çekimler; 5 = k>4)\n")
w("| T (s) | k*=1 | 2 | 3 | 4 | >4 |\n|---|---|---|---|---|---|")
for T in TS:
    c = collections.Counter(kstar("s60_tau180", s, T) for s in seeds)
    w(f"| {T} | " + " | ".join(f"{c.get(i,0)/len(seeds):.2f}" for i in (1, 2, 3, 4, 5)) + " |")

# 3. s x tau heatmap for T=1800, k=2 premium median and k*
w("\n## 3. s × τ ısı haritası (T=1800 s): k* medyanı / π(k=2) medyanı %\n")
S = (15, 30, 60, 120, 240); TAU = (0, 60, 180, 360, 600)
w("| s \\ τ | " + " | ".join(str(t) for t in TAU) + " |\n|---|" + "---|" * len(TAU))
for s in S:
    cells = []
    for tau in TAU:
        cfg = f"s{s}_tau{tau}"
        ks = [kstar(cfg, sd, 1800) for sd in seeds]
        pr = [prem(cfg, sd, 2, 1800) for sd in seeds]; pr = [p for p in pr if p is not None]
        cells.append(f"{np.median(ks):.0f} / " + (f"{100*np.median(pr):.1f}" if pr else "—"))
    w(f"| {s} | " + " | ".join(cells) + " |")

# 4. robustness: other configs vs base
w("\n## 4. Enerji modeli ve hız dayanıklılığı (T=1800 s; taban ile aynı k* ve π işareti kalan çekim oranı)\n")
w("| yapılandırma | k* medyan | π(k=2,T=1800) medyan % | k* tabanla aynı oranı | E_min medyan Wh | π(k=3,T=900) uygun oran / medyan % |\n|---|---|---|---|---|---|")
base_k = {sd: kstar("s60_tau180", sd, 1800) for sd in seeds}
for cfg in ("s60_tau180", "kv_v8", "kv_v10", "kv_v15", "kappa0.8", "kappa0.9", "kappa1.1", "kappa1.2", "alpha1.0", "alpha1.2", "alpha1.6", "v8", "v15", "land_nohover", "s0"):
    ks = [kstar(cfg, sd, 1800) for sd in seeds]
    pr = [prem(cfg, sd, 2, 1800) for sd in seeds]; pr = [p for p in pr if p is not None]
    em = [D[(cfg, sd, 1, 1800)][0] for sd in seeds if (cfg, sd, 1, 1800) in D]
    if not em:
        continue
    same = np.mean([a == base_k[sd] for a, sd in zip(ks, seeds)])
    p39 = [prem(cfg, sd, 3, 900) for sd in seeds]; p39f = [x for x in p39 if x is not None]
    w(f"| {cfg} | {np.median(ks):.0f} | {100*np.median(pr) if pr else float('nan'):.1f} | {same:.2f} | {np.median(em):.0f} | {len(p39f)/len(seeds):.2f} / {100*np.median(p39f) if p39f else float('nan'):.2f} |")

# 5. rules and decomposition and LB
R = collections.defaultdict(list)
for r in rules:
    R[(r["kind"], int(r["k"]))].append(r)
w("\n## 5. Mekanizma ayrıştırması (taban; frontier uçları)\n")
w("| k | uç | medyan E Wh | hizmet-hover payı % | sefer sayısı medyan | kullanılan drone medyan |\n|---|---|---|---|---|---|")
for k in KS:
    for tag_, nm in (("decomp_Emin_end", "min-E"), ("decomp_Tmin_end", "min-T")):
        rs = R.get((tag_, k), [])
        if rs:
            E = np.array([float(r["E"]) for r in rs]); sv = np.array([float(r["service"]) for r in rs])
            w(f"| {k} | {nm} | {np.median(E):.0f} | {100*np.median(sv/E):.1f} | {np.median([int(r['trips']) for r in rs]):.0f} | {np.median([int(r['drones']) for r in rs]):.0f} |")
w("\n## 6. Basit kurallara karşı boşluk: E_kural / E*(T_kural; k) − 1 (medyan %, [GA])\n")
w("| kural | k | medyan boşluk % | GA | boşluk>1% oranı |\n|---|---|---|---|---|")
for rule in ("rule_one_trip_per_request", "rule_nn_bagging"):
    for k in KS:
        rs = [float(r["extra"]) for r in R.get((rule, k), []) if r["extra"] != ""]
        if rs:
            m, lo, hi = boot_med(rs)
            w(f"| {rule[5:]} | {k} | {100*m:.2f} | [{100*lo:.2f}, {100*hi:.2f}] | {np.mean([x > 0.01 for x in rs]):.2f} |")
w("\n## 7. k için analitik alt sınır ⌈W/T⌉ ile kesin k* (T=1800 ve 900 s, taban)\n")
w("| T | k* = alt sınır oranı | kesin k* alt sınırdan büyük oranı | medyan (k*−alt sınır) |\n|---|---|---|---|")
for T in (900, 1800):
    lb = {}
    for r in R.get(("klb", 1), []):
        if int(float(r["E"])) == T:
            lb[int(r["seed"])] = int(r["extra"]) if r["extra"] not in ("", "None") else None
    diffs = [(kstar("s60_tau180", sd, T) - lb[sd]) for sd in seeds if lb.get(sd)]
    if diffs:
        w(f"| {T} | {np.mean([d == 0 for d in diffs]):.2f} | {np.mean([d > 0 for d in diffs]):.2f} | {np.median(diffs):.1f} |")
th = collections.defaultdict(list)
for r in rules:
    if r["kind"] == "theta":
        th[(int(r["k"]), float(r["extra"]))].append((int(r["seed"]), float(r["E"]), float(r["T"])))
if th:
    w("\n## 8. Normalize T düzeyi θ (T = T_min(k) + θ (T_Emin − T_min(k))): enerji primi ve sekant eğimi\n")
    emin = {sd: e for sd, e, _ in th.get((1, 1.0), [])}
    w("| k | θ | medyan π % [GA] | π>1% oranı | medyan T (s) |\n|---|---|---|---|---|")
    for k in KS:
        for t_ in (0.0, 0.25, 0.5, 0.75, 1.0):
            L = th.get((k, t_), [])
            pr_ = [e / emin[sd] - 1 for sd, e, _ in L if sd in emin]
            if pr_:
                m, lo, hi = boot_med(pr_)
                w(f"| {k} | {t_} | {100*m:.2f} [{100*lo:.2f}, {100*hi:.2f}] | {np.mean([x > 0.01 for x in pr_]):.2f} | {np.median([t for _, _, t in L]):.0f} |")
    L0 = {sd: (e, t) for sd, e, t in th.get((4, 0.0), [])}; L1 = {sd: (e, t) for sd, e, t in th.get((4, 1.0), [])}
    sl = [((L0[sd][0] - L1[sd][0]) / max((L1[sd][1] - L0[sd][1]) / 60.0, 1e-9)) for sd in L0 if sd in L1 and L1[sd][1] > L0[sd][1]]
    if sl:
        m, lo, hi = boot_med(sl)
        w(f"\nSekant eğimi (k = 4, θ=0 → θ=1): medyan {m:.2f} Wh/dk [{lo:.2f}, {hi:.2f}] (n = {len(sl)}).")
open(os.path.join(HERE, f"PHASEA_SUMMARY_{tag}.md"), "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
