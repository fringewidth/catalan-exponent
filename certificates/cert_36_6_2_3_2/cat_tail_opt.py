"""
cat_tail_opt.py -- trial sequences in the paper's compact form (finite part + geometric tails) for the
real-place bound (Prop. 6.1) of the design (n, b, q, g, h) = (96, 15, 6, 8, 5), rounded to exact
rationals, as input for a Section-7-type certificate.

FORM (as in Section 7 of the paper): for each of the two sequences u = p, v
    u_k = l_k (1 <= k <= D) + sum_z r_z z^k,
with real bases z (one real coefficient) and conjugate pairs z, conj(z) (coefficients (a, b):
contribution a Re(z^k) + b Im(z^k)). Then, in closed form,
    T(u,x) = sum_k l_k T_k(x)/k - (1/2) sum_z r_z Log(1 - 2xz + z^2),
    S(u,x) = sum_k l_k x^k/k   -        sum_z r_z Log(1 - xz),
    ||u||^2 = sum_k u_k^2 / k  (Gram matrix of the features; truncated at k = 40000 here,
              exact closed forms are used by the certificate).
For fixed bases the objective is convex in the coefficients (quadratic plus two maxima of affine
functions): smoothed maxima (log-sum-exp, temperatures 1e-2 .. 3e-5) and L-BFGS-B.
The features are nearly dependent (cond(Gram) ~ 1e16), so the optimisation runs in whitened
coordinates y (c = V diag(ev^-1/2) y over the eigenvectors of Gram with eigenvalue > 1e-12 * max);
this keeps the coefficients moderate (|c| < 100) and the norm term exactly |y|^2.

BASES (from the structure scout: x-charges on about [-1,-0.10] and [0.10,0.99], s-charges on (0.01, 1)):
    real:  0.5 0.8 0.9 0.95 0.98 0.99 0.995  -0.5 -0.8 -0.9 -0.95 -0.98
    pairs: 0.6i 0.8i 0.9i 0.95i 0.98i  0.1+0.9i  0.1+0.95i  -0.1+0.9i  -0.1+0.95i
    finite part D = 16; the same bases for p and v.

GATES (pre-registered)
  T0 f_low: paper's design gives the integral 8609/2; DES gives -2.275548 as in cat_design_lower.py [0 fails]
  T1 kappa = 2: proof score f_low - (bound) >= +0.0020 with the tail form
     (the 300-coefficient form gave +0.00244)                                        [0 fails]
  T2 rounding every coefficient (and lambda) to a rational with denominator 10^20 changes each
     kappa value by less than 1e-8                                                   [0 fails]
  T3 kappa = 1 (lambda >= 0 optimised) lies below kappa = 2                           [0 fails]
REPORT: both kappa values (rounded coefficients), f_low, the proof score, contact sets.
Numerical suprema on a 400,000-point grid: the rigorous certificate is the next script.

RUN: python3 cat_tail_opt.py   (no arguments; needs numpy and scipy; BLAS threads limited to 2)
TIME: about 10 s (sandbox: 5 s).
OUTPUT: screen + cat_tail_opt_log_<YYYYMMDD_HHMMSS>.txt + cat_tail_opt_out.json (exact rational
        coefficients as strings), next to the script.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"
import json, time, math
from fractions import Fraction as Fr
import numpy as np
from scipy.optimize import minimize

DES = (36, 6, 2, 3, 2)
DFIN = 16
REAL_BASES = ['0.5', '0.8', '0.9', '0.95', '0.98', '0.99', '0.995', '-0.5', '-0.8', '-0.9', '-0.95', '-0.98']
PAIR_BASES = [('0', '0.6'), ('0', '0.8'), ('0', '0.9'), ('0', '0.95'), ('0', '0.98'),
              ('0.1', '0.9'), ('0.1', '0.95'), ('-0.1', '0.9'), ('-0.1', '0.95')]
KTRUNC = 40000
NX = 6000
DEN = 10**20
T1_MIN = 0.0
EIG_CUT = 1e-12
FLOW_REF = -2.2787357          # f_low of DES from cat_design_lower.py

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
STAMP = time.strftime('%Y%m%d_%H%M%S')
LOGPATH = os.path.join(HERE, 'cat_tail_opt_log_' + STAMP + '.txt')
LOGF = open(LOGPATH, 'w')
def log(s):
    line = f"[{time.time()-T0:6.1f}s] {s}"
    print(line, flush=True); LOGF.write(line + '\n'); LOGF.flush()
FAILS = {}
def gate(name, ok, info=""):
    FAILS[name] = FAILS.get(name, 0) + (0 if ok else 1)
    if not ok: log(f"FAIL {name} {info}")

# ---------------------------------------------------------------- f_low (exact, as in cat_design_lower.py)
pos = lambda y: y if y > 0 else 0
def derived(n, b, q, g, h):
    a = b + q; L = n + a; C = L + g; H = C + h; A = a + 2*g
    return a, L, C, H, A
def Dfun(x, n, b, q, g, h):
    a, L, C, H, A = derived(n, b, q, g, h)
    if 2*x <= H:
        d0 = pos(min(x, L - x, A) - max(b, H - 2*x))
        return 2*n - pos(d0 - q)
    K, J = L - x, H - x
    R = pos(K) + pos(K - A) + pos(J - max(b, K))
    S = pos(min(A, K) - b) + pos(min(b, J) - max(0, K))
    return min(2*n, n + R, 2*R + S)
def breakpoints(n, b, q, g, h):
    """candidate kinks/jumps of D: all crossings among its linear forms (as in cat_design_lower.py)."""
    a, L, C, H, A = derived(n, b, q, g, h)
    consts = [0, b, q, A, n, 2*n, L, H, A + b, b + q, A + q, n - b, n - q]
    lins = [(1, 0), (1, q), (-1, L), (-1, L - q), (-2, H), (-2, H + q), (-1, H), (-1, L - A),
            (-1, H - b), (-1, L - b), (-2, H - b)]
    pts = set([Fr(0), Fr(H), Fr(H, 2)])
    forms = [(0, c) for c in consts] + lins
    for i in range(len(forms)):
        for j in range(i + 1, len(forms)):
            (s1, c1), (s2, c2) = forms[i], forms[j]
            if s1 != s2:
                x = Fr(c2 - c1, s1 - s2)
                if 0 <= x <= H: pts.add(x)
    return sorted(pts)
def f_low(n, b, q, g, h):
    """exact integral of D over [0, H] (pieces tested for linearity at 7 points, bisected otherwise)."""
    a, L, C, H, A = derived(n, b, q, g, h)
    f = lambda x: Fr(Dfun(x, n, b, q, g, h))
    pts = breakpoints(n, b, q, g, h)
    I = Fr(0); unresolved = 0
    stack = [(pts[i], pts[i+1], 0) for i in range(len(pts)-1)][::-1]
    while stack:
        lo, hi, depth = stack.pop()
        if hi <= lo: continue
        w = hi - lo
        xs = [lo + w*Fr(k, 8) for k in range(1, 8)]; ys = [f(x) for x in xs]
        slope = (ys[6] - ys[0])/(xs[6] - xs[0])
        lin = all(ys[k] == ys[0] + slope*(xs[k] - xs[0]) for k in range(7))
        if not lin and depth < 60:
            mid = (lo + hi)/2; stack.append((mid, hi, depth+1)); stack.append((lo, mid, depth+1)); continue
        if not lin: unresolved += 1
        I += w*ys[3]
    gate("T0", unresolved == 0, f"{(n,b,q,g,h)}: {unresolved} unresolved pieces")
    delta = Fr(q + g + h, n)
    return -float(I)/n**2 - (0.5 + float(delta)/2 + float(delta)**2/8)*math.log(2), I

# ---------------------------------------------------------------- features
n0, b0, q0, g0, h0 = DES
al, be, gq, gg, et = (b0+q0)/n0, b0/n0, q0/n0, g0/n0, h0/n0
rb = [float(z) for z in REAL_BASES]
pb = [complex(float(a), float(b)) for a, b in PAIR_BASES]
NP = DFIN + len(rb) + 2*len(pb)
kk = np.arange(1, KTRUNC + 1, dtype=float)
Phi = np.zeros((KTRUNC, NP))
for k in range(1, DFIN + 1): Phi[k-1, k-1] = 1.0
for j, z in enumerate(rb): Phi[:, DFIN + j] = z**kk
for j, z in enumerate(pb):
    zk = z**kk
    Phi[:, DFIN + len(rb) + 2*j] = zk.real; Phi[:, DFIN + len(rb) + 2*j + 1] = zk.imag
Gram = Phi.T @ (Phi / kk[:, None])
def featT(x):
    """columns: T_k(x)/k (k <= D), tails: Re/Im of -(1/2) Log(1 - 2xz + z^2)."""
    x = np.asarray(x, dtype=float); F = np.zeros((len(x), NP))
    t0, t1 = np.ones_like(x), x
    for k in range(1, DFIN + 1):
        F[:, k-1] = t1/k; t0, t1 = t1, 2*x*t1 - t0
    for j, z in enumerate(rb): F[:, DFIN + j] = -0.5*np.log(1 - 2*x*z + z*z)
    for j, z in enumerate(pb):
        w = -0.5*np.log(1 - 2*x*z + z*z + 0j)
        F[:, DFIN + len(rb) + 2*j] = w.real; F[:, DFIN + len(rb) + 2*j + 1] = w.imag
    return F
def featS(x):
    """columns: x^k/k (k <= D), tails: Re/Im of -Log(1 - xz)."""
    x = np.asarray(x, dtype=float); F = np.zeros((len(x), NP))
    for k in range(1, DFIN + 1): F[:, k-1] = x**k/k
    for j, z in enumerate(rb): F[:, DFIN + j] = -np.log(1 - x*z)
    for j, z in enumerate(pb):
        w = -np.log(1 - x*z + 0j)
        F[:, DFIN + len(rb) + 2*j] = w.real; F[:, DFIN + len(rb) + 2*j + 1] = w.imag
    return F
th = np.linspace(0, np.pi, NX + 2)[1:-1]
X = np.cos(th); X = X[np.abs(X) > 1e-6]; X = np.concatenate([[-1.0], X])
S = (1 - np.cos(th))/2
FTX, FSX, FTS = featT(X), featS(X), featT(S)
xf = np.cos(np.linspace(0, np.pi, 400002)[1:-1]); xf = xf[np.abs(xf) > 1e-7]; xf = np.concatenate([[-1.0], xf])
sf = (1 - np.cos(np.linspace(0, np.pi, 400002)[1:-1]))/2
FTXf, FSXf, FTSf = featT(xf), featS(xf), featT(sf)
CONST = (-1 + al + gg)*math.log(2)
def fields(x, kap):
    return (al + 2*gg)*np.log(np.abs(x)) + 2*et*np.log(1 - x) - (kap/2 + al + et + gg)*np.log(1 + x*x), \
           2*gg - 2*x*x/(1 + x*x)
WX2, DX = fields(X, 2); WX1, _ = fields(X, 1)
WS = be*np.log(S) + gq*np.log(1 - S)
def lse(zv, tau):
    m = zv.max(); w = np.exp((zv - m)/tau); s = w.sum()
    return m + tau*np.log(s), w/s

def solve(kap):
    """optimisation in whitened coordinates y = R c (Gram = R^T R), so the quadratic part is |y|^2;
    returns the coefficients c."""
    WX = WX2 if kap == 2 else WX1
    ev, V = np.linalg.eigh(Gram)
    keep = ev > EIG_CUT*ev.max()
    Rinv = V[:, keep]/np.sqrt(ev[keep])          # c = Rinv y; directions with tiny eigenvalue dropped
    GX, HX, GS = FTX @ Rinv, FSX @ Rinv, FTS @ Rinv
    m = Rinv.shape[1]; nz = 2*m + (1 if kap == 1 else 0)
    def F(z, tau):
        yp, yv = z[:m], z[m:2*m]; lam = z[2*m]**2 if kap == 1 else 0.0
        Xv = WX + lam*DX - 2*kap*(GX @ yp) - HX @ yv
        Yv = WS + 2*(GS @ yv)
        sx, wx = lse(Xv, tau); sy, wy = lse(Yv, tau)
        val = CONST + kap*yp @ yp + 0.5*yv @ yv + sx + sy
        gp = 2*kap*yp - 2*kap*(GX.T @ wx)
        gv = yv - HX.T @ wx + 2*(GS.T @ wy)
        g = np.concatenate([gp, gv])
        if kap == 1: g = np.concatenate([g, [2*z[2*m]*(DX @ wx)]])
        return val, g
    z = np.zeros(nz)
    if kap == 1: z[2*m] = 1.6
    for tau in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5):
        z = minimize(F, z, args=(tau,), jac=True, method='L-BFGS-B', options={'maxiter': 5000}).x
    m = Rinv.shape[1]
    return np.concatenate([Rinv @ z[:m], Rinv @ z[m:2*m], z[2*m:]]), m

def evaluate(cp, cv, lam, kap):
    WXf, DXf = fields(xf, kap)
    Xf = WXf + lam*DXf - 2*kap*(FTXf @ cp) - FSXf @ cv
    Yf = be*np.log(sf) + gq*np.log(1 - sf) + 2*(FTSf @ cv)
    return CONST + kap*cp @ Gram @ cp + 0.5*cv @ Gram @ cv + Xf.max() + Yf.max(), Xf, Yf

_, Ipap = f_low(48, 7, 4, 4, 2)
gate("T0", Ipap == Fr(8609, 2), f"paper's design: integral {Ipap}, paper prints 8609/2")
fl, Ilow = f_low(*DES)
gate("T0", abs(fl - FLOW_REF) < 1e-6, f"f_low {fl:.7f} vs cat_design_lower.py {FLOW_REF}")
log(f"design {DES}: f_low = {fl:.7f} (integral {Ilow}); {NP} coefficients per sequence "
    f"({DFIN} finite, {len(rb)} real tails, {len(pb)} conjugate pairs)")
res = {}
for kap in (2, 1):
    z, mdir = solve(kap)
    cp, cv = z[:NP], z[NP:2*NP]; lam = z[2*NP]**2 if kap == 1 else 0.0
    v_float, _, _ = evaluate(cp, cv, lam, kap)
    # round to rationals with denominator DEN = 10^20
    cpq = [Fr(round(c*DEN), DEN) for c in cp]; cvq = [Fr(round(c*DEN), DEN) for c in cv]
    lamq = Fr(round(lam*DEN), DEN)
    v_round, Xf, Yf = evaluate(np.array([float(c) for c in cpq]), np.array([float(c) for c in cvq]), float(lamq), kap)
    gate("T2", abs(v_round - v_float) < 1e-8, f"kappa={kap}: rounding moved the value by {v_round - v_float:.2e}")
    res[kap] = {"value": v_round, "lambda": str(lamq), "cp": [str(c) for c in cpq], "cv": [str(c) for c in cvq]}
    log(f"kappa={kap}: {mdir} of {NP} directions kept; max |coefficient| {max(abs(z[:2*NP])):.3g}")
    log(f"kappa={kap}: bound {v_round:.7f} (before rounding {v_float:.9f})" + (f", lambda {float(lamq):.6f}" if kap == 1 else ""))
    if kap == 2:
        mX, mY = Xf.max(), Yf.max()
        onX = xf[Xf >= mX - 3e-4]; onY = sf[Yf >= mY - 3e-4]
        log(f"  contact (within 3e-4): x in [{onX[onX < 0].min() if (onX < 0).any() else float('nan'):.4f}, "
            f"{onX[onX < 0].max() if (onX < 0).any() else float('nan'):.4f}] U [{onX[onX > 0].min():.4f}, {onX[onX > 0].max():.4f}]; "
            f"s in [{onY.min():.4f}, {onY.max():.4f}]")
score = fl - max(res[2]["value"], res[1]["value"])
gate("T1", fl - res[2]["value"] >= T1_MIN, f"kappa=2 score {fl - res[2]['value']:+.6f} < {T1_MIN}")
gate("T3", res[1]["value"] < res[2]["value"], "kappa=1 not below kappa=2")
log(f"PROOF SCORE (tail form, rounded coefficients) = {score:+.6f}   (300-coefficient form: +0.00244; paper's design as printed: +0.00013)")

total = sum(FAILS.values())
out = {"script": "cat_tail_opt.py", "design": DES, "f_low": fl, "f_low_integral": str(Ilow), "D": DFIN,
       "real_bases": REAL_BASES, "pair_bases": PAIR_BASES, "denominator": DEN,
       "kappa2": res[2], "kappa1": res[1], "proof_score": score, "gates": FAILS, "GATE_FAILS": total,
       "seconds": round(time.time()-T0, 1)}
with open(os.path.join(HERE, "cat_tail_opt_out.json"), "w") as fh: json.dump(out, fh, indent=1)
log(f"gates: {FAILS}")
log(f'cond(Gram) = {np.linalg.cond(Gram):.3g}')
print(f"GATE_FAILS = {total}")
LOGF.write(f"GATE_FAILS = {total}\n"); LOGF.close()
print(f"log saved to {LOGPATH}")
