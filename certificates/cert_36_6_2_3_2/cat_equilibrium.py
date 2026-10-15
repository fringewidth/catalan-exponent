"""
cat_equilibrium.py -- dual (equilibrium) form of the real-place bound of Prop. 6.1, for the paper's
design (48, 7, 4, 4, 2) and the design (96, 15, 6, 8, 5).

DUALITY (derived from eqs. (81)-(82) of the preprint at r = 1; proof of the min-max step to be written):
for fixed kappa the infimum over trial sequences p, v (and lambda >= 0 when kappa = 1) of the
right side of (78) equals the maximum over probability measures mu on [-1, 1], nu on [0, 1] of
  Phi(mu, nu) = (alpha + gg + kappa/2) log 2 + int W_kappa dmu + int W_s dnu
                + (kappa/2) II log|x - x'| dmu dmu + (1/2) II log(1 - x x') dmu dmu
                + II log|s - s'| dnu dnu - II log(1 - 2 x s + x^2) dmu dnu,
with the constraint int D dmu >= 0 when kappa = 1. At the optimum p_k = int T_k dmu and
v_k = int x^k dmu - 2 int T_k dnu (Chebyshev and power moments).
alpha = (b+q)/n, beta = b/n, gq = q/n, gg = g/n, eta = h/n;
W_kappa = (alpha + 2gg) log|x| + 2 eta log(1-x) - (kappa/2 + alpha + eta + gg) log(1+x^2),
W_s = beta log s + gq log(1-s), D = 2gg - 2x^2/(1+x^2).

METHOD: mu, nu with piecewise-constant densities on Chebyshev-spaced cells (x cells with an edge at 0);
exact cell integrals of log|x-y|, log|x|, log(1-x), log s, log(1-s); Gauss-Legendre (12 points per
cell) for log(1+x^2), D, log(1-xx'), log(1-2xs+x^2). Phi is a concave quadratic in the cell weights;
it is maximised by FISTA with projection onto the two simplices. kappa = 1: min over lambda >= 0 of
max Phi + lambda int D dmu (golden section on [0, 8], warm-started).
Every value of Phi over this restricted class of measures is <= the best bound U*_kappa the method
can give (weak duality); every primal value (any admissible sequences) is >= U*_kappa.

GATES (pre-registered)
  E1 grid convergence: kappa = 2 dual value, fine grid (500, 330) minus coarse grid (300, 200), lies
     in (-5e-6, 3e-5) for both designs (each value is a lower bound of U*, so refinement should
     not lower it beyond quadrature error)                                              [0 fails]
  E2 weak duality against known primal values (bug check): dual <= primal, for
     paper: kappa=2 printed -2.290939875, our 300-coefficient -2.29192; kappa=1 printed -2.296789875;
     new:   kappa=2 certified -2.27779256 and 300-coefficient -2.2779664; kappa=1 certified
            -2.41413176 (cat_cert7_new.py)                                                [0 fails]
  E3 paper design, kappa = 2: the moments p_k, v_k of the optimal measures agree with the paper's
     printed Section-7 sequences to within 5e-3 for k = 1..10                           [0 fails]
  E4 constraint (67) is slack at the kappa = 2 optimum: int D dmu < 0 (both designs)     [0 fails]
REPORT: dual values, the bracket for the best provable score of the method,
  f_low - max(primal) <= score* <= f_low - max(dual), supports of mu and nu, the moment table,
  lambda for kappa = 1. Densities are written to the JSON output for later analysis.

RUN: python3 cat_equilibrium.py   (no arguments; needs numpy; BLAS threads limited to 2)
TIME: about 2 min.
OUTPUT: screen + cat_equilibrium_log_<YYYYMMDD_HHMMSS>.txt + cat_equilibrium_out.json, next to the script.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"
import json, time, math
import numpy as np

GRID_COARSE = (300, 200)
GRID_FINE = (500, 330)
GL = 12
ITERS = 15000
ITERS_LAMBDA = 4000
LAM_RANGE = (0.0, 8.0)
LAM_STEPS = 22
E1_TOL = 3e-5
E1_LOW = 5e-6
E3_TOL = 5e-3
DESIGNS = {
 "paper": {"des": (48, 7, 4, 4, 2), "f_low": -8609/4608 - (2809/4608)*math.log(2),
           "primal": {2: [("printed (Sec. 7)", -2.290939875), ("300-coefficient", -2.29192)],
                      1: [("printed (Sec. 7)", -2.296789875)]}},
 "new":   {"des": (36, 6, 2, 3, 2), "f_low": -4825/2592 - (0.5 + 7/72 + 49/10368)*math.log(2),
           "primal": {2: [("certified (cat_cert7_new.py, edges)", -2.2812647361289353)],
                      1: [("certified (cat_cert7_new.py, edges)", -2.4192376465502576)]}}}
# paper's printed kappa = 2 sequences (Section 7 tables; integers x 1e-8; pairs (a, b): a Re z^k + b Im z^k)
FINp = [45559127, -50750856, -6578767, 13970217, 4786184, -4292433, -576704, 1311615, 671564, -346453]
FINv = [-23910158, 21152432, -2885110, -11558199, 6485289, 1456821, -1912176, -2263524, 2742210, -1162454]
EXPp = [('.85', '0', -9338452, None), ('.94', '0', -2141509, None), ('0', '.7', -66277922, -31907569),
        ('0', '.85', -1231651, 6002645), ('.092', '.92', -3225918, 8928234), ('-.092', '.92', -2105536, -9091287)]
EXPv = [('-.8', '0', 15199211, None), ('-.96', '0', 4451662, None), ('.88', '0', 2545398, None),
        ('.95', '0', -4932634, None), ('.984', '0', 11618157, None), ('0', '.78', 27238714, -38447936),
        ('0', '.9', -31341084, -30188786), ('0', '.955', -6693542, 11912254), ('0', '.984', 2055213, -21715849)]

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
STAMP = time.strftime('%Y%m%d_%H%M%S')
LOGPATH = os.path.join(HERE, 'cat_equilibrium_log_' + STAMP + '.txt')
LOGF = open(LOGPATH, 'w')
def log(s):
    line = f"[{time.time()-T0:6.1f}s] {s}"
    print(line, flush=True); LOGF.write(line + '\n'); LOGF.flush()
FAILS = {}
def gate(name, ok, info=""):
    FAILS[name] = FAILS.get(name, 0) + (0 if ok else 1)
    if not ok: log(f"FAIL {name} {info}")

def F2(u):
    """second antiderivative of log|u| vanishing with its derivative's u-part at 0: u^2 log|u|/2 - 3u^2/4."""
    u = np.asarray(u, float); out = np.zeros_like(u); nz = u != 0
    out[nz] = u[nz]**2/2*np.log(np.abs(u[nz])) - 3*u[nz]**2/4
    return out
def cell_logabs(e1, e2):
    a, b = e1[:-1, None], e1[1:, None]; c, d = e2[None, :-1], e2[None, 1:]
    return F2(b - c) - F2(a - c) - F2(b - d) + F2(a - d)
def gauss(edges, G):
    t, w = np.polynomial.legendre.leggauss(G)
    a, b = edges[:-1], edges[1:]
    return (a[:, None] + b[:, None])/2 + (b - a)[:, None]/2*t[None, :], (b - a)[:, None]/2*w[None, :]
def int_logabs(a, b):
    f = lambda x: np.where(x == 0, 0.0, x*np.log(np.abs(np.where(x == 0, 1, x))) - x)
    return f(b) - f(a)
def int_log1m(a, b):
    f = lambda x: np.where(x == 1, -1.0, -(1 - x)*np.log(np.where(x == 1, 1, 1 - x)) - x)
    return f(b) - f(a)
def proj_simplex(v):
    u = np.sort(v)[::-1]; css = np.cumsum(u) - 1; k = np.arange(1, len(v) + 1)
    r = k[u - css/k > 0][-1]
    return np.maximum(v - css[r-1]/r, 0)

class Problem:
    def __init__(self, des, kap, Nx, Ns):
        n, b, q, g, h = des
        self.al, self.be, self.gq, self.gg, self.et = (b+q)/n, b/n, q/n, g/n, h/n
        self.kap, self.Nx, self.Ns = kap, Nx, Ns
        ex = -np.cos(np.pi*np.arange(Nx + 1)/Nx); ex[Nx//2] = 0.0; ex[0], ex[-1] = -1.0, 1.0
        es = (1 - np.cos(np.pi*np.arange(Ns + 1)/Ns))/2; es[0], es[-1] = 0.0, 1.0
        self.ex, self.es = ex, es
        hx, hs = np.diff(ex), np.diff(es); self.hx, self.hs = hx, hs
        Xg, Xw = gauss(ex, GL); Sg, Sw = gauss(es, GL)
        self.Xg, self.Xw, self.Sg, self.Sw = Xg, Xw, Sg, Sw
        al, be, gq, gg, et = self.al, self.be, self.gq, self.gg, self.et
        self.lx0 = ((al + 2*gg)*int_logabs(ex[:-1], ex[1:]) + 2*et*int_log1m(ex[:-1], ex[1:])
                    - (kap/2 + al + et + gg)*np.sum(Xw*np.log(1 + Xg**2), 1))/hx
        self.dx = np.sum(Xw*(2*gg - 2*Xg**2/(1 + Xg**2)), 1)/hx           # cell average of D
        self.ls = (be*int_logabs(es[:-1], es[1:]) + gq*int_log1m(es[:-1], es[1:]))/hs
        Kxx = cell_logabs(ex, ex)/np.outer(hx, hx)
        Kss = cell_logabs(es, es)/np.outer(hs, hs)
        xf, xw = Xg.ravel(), Xw.ravel(); sf, sw = Sg.ravel(), Sw.ravel()
        Kp = np.zeros((Nx, Nx)); Kxs = np.zeros((Nx, Ns)); BL = 25          # blocks of BL x-cells (memory)
        for i0 in range(0, Nx, BL):
            i1 = min(Nx, i0 + BL); xb, wb = Xg[i0:i1].ravel(), Xw[i0:i1].ravel(); m = i1 - i0
            Kp[i0:i1] = (np.log(np.maximum(1 - np.outer(xb, xf), 1e-300))*np.outer(wb, xw)).reshape(m, GL, Nx, GL).sum((1, 3))
            Kxs[i0:i1] = (np.log((1 - xb[:, None])**2 + 2*xb[:, None]*(1 - sf[None, :]))*np.outer(wb, sw)).reshape(m, GL, Ns, GL).sum((1, 3))
        Kp /= np.outer(hx, hx); Kxs /= np.outer(hx, hs)
        M = np.zeros((Nx + Ns, Nx + Ns))
        M[:Nx, :Nx] = kap*Kxx + Kp; M[Nx:, Nx:] = 2*Kss
        M[:Nx, Nx:] = -Kxs; M[Nx:, :Nx] = -Kxs.T
        self.M = (M + M.T)/2
        ev = np.linalg.eigvalsh(self.M); self.eig = (ev.min(), ev.max())
        self.Lc = abs(ev.min())
        self.const = (al + gg + kap/2)*math.log(2)
        self.w = np.concatenate([proj_simplex(hx/2), proj_simplex(hs)])
    def P(self, w): return np.concatenate([proj_simplex(w[:self.Nx]), proj_simplex(w[self.Nx:])])
    def phi(self, w, lam=0.0):
        l = np.concatenate([self.lx0 + lam*self.dx, self.ls])
        return self.const + l @ w + 0.5*w @ self.M @ w
    def maximise(self, lam=0.0, iters=ITERS, w0=None):
        l = np.concatenate([self.lx0 + lam*self.dx, self.ls])
        w = self.w if w0 is None else w0; y = w.copy(); tk = 1.0
        for _ in range(iters):
            wn = self.P(y + (l + self.M @ y)/self.Lc)
            tn = (1 + math.sqrt(1 + 4*tk*tk))/2
            y = wn + (tk - 1)/tn*(wn - w); w, tk = wn, tn
        return self.phi(w, lam), w
    def moments(self, w, K):
        Nx = self.Nx
        mx = (self.Xw*(w[:Nx]/self.hx)[:, None]).ravel(); xs = self.Xg.ravel()
        ms = (self.Sw*(w[Nx:]/self.hs)[:, None]).ravel(); ss = self.Sg.ravel()
        A, B, C = np.zeros(K), np.zeros(K), np.zeros(K)
        t0, t1, xp, u0, u1 = np.ones_like(xs), xs.copy(), xs.copy(), np.ones_like(ss), ss.copy()
        for k in range(K):
            A[k], B[k], C[k] = t1 @ mx, xp @ mx, u1 @ ms
            t0, t1 = t1, 2*xs*t1 - t0; xp = xp*xs; u0, u1 = u1, 2*ss*u1 - u0
        return A, B - 2*C, float(mx @ (2*self.gg - 2*xs**2/(1 + xs**2)))
    def segments(self, w, which, rel=1e-4):
        if which == 'x': dens, e = w[:self.Nx]/self.hx, self.ex
        else: dens, e = w[self.Nx:]/self.hs, self.es
        on = dens > rel*dens.max(); segs = []; i = 0
        while i < len(dens):
            if on[i]:
                j = i
                while j + 1 < len(dens) and on[j+1]: j += 1
                segs.append((round(float(e[i]), 4), round(float(e[j+1]), 4))); i = j + 1
            else: i += 1
        return segs

def kappa1(prob):
    """min over lambda in LAM_RANGE of max_w Phi + lambda int D dmu (golden section, warm start)."""
    gr = (math.sqrt(5) - 1)/2; a, b = LAM_RANGE; w = None; cache = {}
    def f(lam):
        nonlocal w
        v, w = prob.maximise(lam, ITERS_LAMBDA, w); cache[lam] = (v, w.copy()); return v
    c, d = b - gr*(b - a), a + gr*(b - a); fc, fd = f(c), f(d)
    for _ in range(LAM_STEPS):
        if fc < fd: b, d, fd = d, c, fc; c = b - gr*(b - a); fc = f(c)
        else: a, c, fc = c, d, fd; d = a + gr*(b - a); fd = f(d)
    lam = (a + b)/2
    v, w = prob.maximise(lam, ITERS, cache[min(cache, key=lambda t: abs(t - lam))][1])
    return v, w, lam

def paper_seq(FIN, EXP, K):
    u = np.zeros(K)
    for k in range(1, K + 1):
        s = FIN[k-1]*1e-8 if k <= len(FIN) else 0.0
        for re, im, a, b in EXP:
            z = complex(float(re), float(im))
            s += a*1e-8*z.real**k if b is None else (complex(a, -b)*1e-8*z**k).real
        u[k-1] = s
    return u

OUT = {"script": "cat_equilibrium.py", "grids": [GRID_COARSE, GRID_FINE]}
for name, info in DESIGNS.items():
    des = info["des"]; res = {}
    log(f"== {name} design {des}: f_low = {info['f_low']:.9f}")
    # kappa = 2
    vals = {}
    for grid in (GRID_COARSE, GRID_FINE):
        pr = Problem(des, 2, *grid); v, w = pr.maximise()
        vals[grid] = v
        log(f"  kappa=2 grid {grid}: dual max {v:.9f} (quadratic form eigenvalues in [{pr.eig[0]:.3g}, {pr.eig[1]:.3g}])")
    dg = vals[GRID_FINE] - vals[GRID_COARSE]
    gate("E1", -E1_LOW < dg < E1_TOL, f"{name}: fine - coarse = {dg:.2e}")
    d2 = vals[GRID_FINE]
    p_mom, v_mom, intD = pr.moments(w, 24)
    gate("E4", intD < 0, f"{name}: int D dmu = {intD}")
    segx, segs = pr.segments(w, 'x'), pr.segments(w, 's')
    log(f"  kappa=2: int D dmu = {intD:.4f}; supp mu ~ {segx}; supp nu ~ {segs}")
    res["kappa2"] = {"dual": d2, "dual_coarse": vals[GRID_COARSE], "int_D": intD, "supp_mu": segx, "supp_nu": segs,
                     "p_moments": p_mom.tolist(), "v_moments": v_mom.tolist(),
                     "x_edges": pr.ex.tolist(), "mu_density": (w[:pr.Nx]/pr.hx).tolist(),
                     "s_edges": pr.es.tolist(), "nu_density": (w[pr.Nx:]/pr.hs).tolist()}
    if name == "paper":
        pp, vp = paper_seq(FINp, EXPp, 24), paper_seq(FINv, EXPv, 24)
        log("   k   p (paper)  p (moment) |  v (paper)  v (moment)")
        for k in range(24): log(f"  {k+1:2d}  {pp[k]:+.5f}   {p_mom[k]:+.5f}  |  {vp[k]:+.5f}   {v_mom[k]:+.5f}")
        dmax = max(np.max(np.abs(pp[:10] - p_mom[:10])), np.max(np.abs(vp[:10] - v_mom[:10])))
        gate("E3", dmax < E3_TOL, f"max |paper - moment| for k <= 10: {dmax:.2e}")
        log(f"  E3: max |paper sequence - equilibrium moment| over k <= 10: {dmax:.2e}")
        res["paper_sequences"] = {"p": pp.tolist(), "v": vp.tolist()}
    # kappa = 1
    pr1 = Problem(des, 1, *GRID_FINE); d1, w1, lam1 = kappa1(pr1)
    _, _, intD1 = pr1.moments(w1, 2)
    log(f"  kappa=1: dual (min over lambda) {d1:.9f}, lambda {lam1:.4f}, int D dmu = {intD1:.2e}")
    res["kappa1"] = {"dual": d1, "lambda": lam1, "int_D": intD1}
    # weak duality and bracket
    for kap, dv in ((2, d2), (1, d1)):
        for lab, pv in info["primal"][kap]:
            gate("E2", dv <= pv + 1e-12, f"{name} kappa={kap}: dual {dv:.9f} > primal {lab} {pv}")
    Ulo = max(d2, d1); Uhi = max(min(pv for _, pv in info["primal"][2]), min(pv for _, pv in info["primal"][1]))
    lo, hi = info["f_low"] - Uhi, info["f_low"] - Ulo
    log(f"  best provable score of the method: {lo:.5f} <= score* <= {hi:.5f}   (U* in [{Ulo:.6f}, {Uhi:.6f}])")
    res["bracket_score"] = [lo, hi]; res["bracket_U"] = [Ulo, Uhi]
    OUT[name] = res

total = sum(FAILS.values())
OUT.update({"gates": FAILS, "GATE_FAILS": total, "seconds": round(time.time() - T0, 1)})
with open(os.path.join(HERE, "cat_equilibrium_out.json"), "w") as fh: json.dump(OUT, fh)
log(f"gates: {FAILS}")
print(f"GATE_FAILS = {total}")
LOGF.write(f"GATE_FAILS = {total}\n"); LOGF.close()
print(f"log saved to {LOGPATH}")
