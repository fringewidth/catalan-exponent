"""
cat_design_search.py -- local search for the design of the OpenAI Catalan construction with the
largest PROVABLE margin (proof score = f_low - f_up, as in cat_design_upper.py).

SEARCH
  phase 1, n = 48: start at (48, 7, 3, 4, 2) (best of the nine designs of cat_design_upper.py);
    evaluate the 8 one-step neighbours (b, q, g, h each -/+ 1), move to the best one if it improves
    the score, repeat until no neighbour improves (at most 40 evaluations).
  phase 2, n = 96 (half steps): start at the doubled phase-1 optimum, same rule (at most 40 evaluations).
  During the search only kappa = 2 is optimised (kappa = 1 never bound in cat_design_upper.py);
  the final best design of each phase is re-evaluated with kappa = 1 as well.
  Admissible designs: b >= 0, q >= 1, g >= 1, h >= 0, g <= (n-1)/2.

GATES (pre-registered)
  S1 the start design reproduces its cat_design_upper.py score +0.001676 within 5e-5
     (the optimiser reproduces to about 2e-5 across machines: sandbox gave +0.001659)        [0 fails]
  S2 for each phase optimum, the kappa = 1 value lies below the kappa = 2 value (kappa = 2 binds) [0 fails]
REPORT: search path, every evaluated design with its score, the two optima.
Numerical (grid suprema, optimised trial sequences), not certified.

RUN: python3 cat_design_search.py   (no arguments; needs numpy and scipy; BLAS threads limited to 2)
TIME (from the record run of cat_design_upper.py: about 5 s per kappa=2 evaluation at n = 48):
  up to 80 evaluations, expected 5-15 min.
OUTPUT: screen + cat_design_search_log_<YYYYMMDD_HHMMSS>.txt + cat_design_search_out.json, next to the script.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"
import sys, json, time, math
from fractions import Fraction as Fr
import numpy as np
from scipy.optimize import minimize

PAPER_PRINTED = {2: -2.290939875, 1: -2.296789875}
LOG2 = math.log(2)

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
STAMP = time.strftime('%Y%m%d_%H%M%S')
LOGPATH = os.path.join(HERE, 'cat_design_search_log_' + STAMP + '.txt')
LOGF = None
def log(s):
    line = f"[{time.time()-T0:7.1f}s] {s}"
    pass
FAILS = {}
def gate(name, ok, info=""):
    FAILS[name] = FAILS.get(name, 0) + (0 if ok else 1)
    if not ok: log(f"FAIL {name} {info}")

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
    """candidate kinks/jumps: all crossings among the linear forms that appear in D."""
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

def integrate(n, b, q, g, h, exact=True):
    """integral of D over [0, H]. D is piecewise linear and may jump at isolated points.
    On each piece the interior is tested for linearity at 7 points; a linear piece contributes
    width * D(midpoint) (endpoint jumps do not matter). Pieces that are not linear are bisected;
    after 60 levels a piece is accepted with its midpoint value and counted as unresolved."""
    a, L, C, H, A = derived(n, b, q, g, h)
    num = (lambda v: Fr(v)) if exact else (lambda v: float(v))
    f = lambda x: num(Dfun(x, n, b, q, g, h))
    pts = [num(p) for p in breakpoints(n, b, q, g, h)]
    total = num(0); pieces = []; unresolved = 0
    stack = [(pts[i], pts[i+1], 0) for i in range(len(pts)-1)][::-1]
    while stack:
        lo, hi, depth = stack.pop()
        if hi <= lo: continue
        w = hi - lo
        xs = [lo + w*num(k)/8 for k in range(1, 8)]
        ys = [f(x) for x in xs]
        slope = (ys[6] - ys[0])/(xs[6] - xs[0])
        lin = all(abs(ys[k] - (ys[0] + slope*(xs[k]-xs[0]))) <= (0 if exact else 1e-9) for k in range(7))
        if not lin and depth < 60:
            mid = (lo + hi)/2; stack.append((mid, hi, depth+1)); stack.append((lo, mid, depth+1)); continue
        if not lin: unresolved += 1
        ymid = ys[3]
        total += w*ymid
        pieces.append((lo, hi, ys[0] - slope*(xs[0]-lo), ys[0] + slope*(hi-xs[0])))
    return total, pieces, unresolved

def f_low(n, b, q, g, h, exact=False):
    I, _, unres = integrate(n, b, q, g, h, exact=exact)
    gate("G1", unres == 0, f"{(n,b,q,g,h)}: {unres} unresolved pieces")
    delta = Fr(q + g + h, n)
    val = -float(I)/n**2 - (0.5 + float(delta)/2 + float(delta)**2/8)*LOG2
    return val, I


K = 300                     # number of trial coefficients per sequence
NX = 6000                   # optimisation grid
def design_ratios(n, b, q, g, h):
    a = b + q
    return a/n, b/n, q/n, g/n, h/n
th = np.linspace(0, np.pi, NX+2)[1:-1]
X = np.cos(th); X = X[np.abs(X) > 1e-6]; X = np.concatenate([[-1.0], X])
S = (1 - np.cos(th))/2
kk = np.arange(1, K+1)
def cheb(x):
    T = np.zeros((K+1, len(x))); T[0] = 1; T[1] = x
    for k in range(2, K+1): T[k] = 2*x*T[k-1] - T[k-2]
    return T[1:]
TX, TS = cheb(X), cheb(S)
PX = np.array([X**k for k in kk])
def setup(des, kap):
    al, be, gq, gg, et = design_ratios(*des)
    W = (al+2*gg)*np.log(np.abs(X)) + 2*et*np.log(1-X) - (kap/2+al+et+gg)*np.log(1+X**2)
    Dx = 2*gg - 2*X**2/(1+X**2)
    WS = be*np.log(S) + gq*np.log(1-S)
    const = (-1+al+gg)*np.log(2)
    return W, Dx, WS, const
def lse(z, tau):
    m = z.max(); w = np.exp((z-m)/tau); s = w.sum()
    return m + tau*np.log(s), w/s
def solve(des, kap, verbose=False):
    W, Dx, WS, const = setup(des, kap)
    nv = 2*K + (1 if kap == 1 else 0)
    def unpack(z):
        p, v = z[:K], z[K:2*K]; lam = z[2*K]**2 if kap == 1 else 0.0
        return p, v, lam
    def F(z, tau):
        p, v, lam = unpack(z)
        Xv = W + lam*Dx - 2*kap*(p/kk) @ TX - (v/kk) @ PX
        Yv = WS + 2*(v/kk) @ TS
        sx, wx = lse(Xv, tau); sy, wy = lse(Yv, tau)
        val = const + kap*np.sum(p*p/kk) + 0.5*np.sum(v*v/kk) + sx + sy
        gp = 2*kap*p/kk - 2*kap*(TX @ wx)/kk
        gv = v/kk - (PX @ wx)/kk + 2*(TS @ wy)/kk
        g = np.concatenate([gp, gv])
        if kap == 1: g = np.concatenate([g, [2*z[2*K]*(Dx @ wx)]])
        return val, g
    z = np.zeros(nv)
    if kap == 1: z[2*K] = 1.5   # lambda = 2.25 start (lambda = z^2 has zero gradient at z = 0)
    for tau in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5):
        r = minimize(F, z, args=(tau,), jac=True, method='L-BFGS-B', options={'maxiter': 3000})
        z = r.x
    p, v, lam = unpack(z)
    # exact-ish evaluation of the objective on a fine grid
    xf = np.cos(np.linspace(0, np.pi, 400002)[1:-1]); xf = xf[np.abs(xf) > 1e-7]; xf = np.concatenate([[-1.0], xf])
    sf = (1-np.cos(np.linspace(0, np.pi, 400002)[1:-1]))/2
    al, be, gq, gg, et = design_ratios(*des)
    def chebv(x, c):
        t0, t1 = np.ones_like(x), x; out = np.zeros_like(x)
        for k in range(1, K+1):
            out += c[k-1]*t1/k; t0, t1 = t1, 2*x*t1 - t0
        return out
    Xf = (al+2*gg)*np.log(np.abs(xf)) + 2*et*np.log(1-xf) - (kap/2+al+et+gg)*np.log(1+xf**2) \
         + lam*(2*gg - 2*xf**2/(1+xf**2)) - 2*kap*chebv(xf, p) - sum(v[k-1]*xf**k/k for k in kk)
    Yf = be*np.log(sf) + gq*np.log(1-sf) + 2*chebv(sf, v)
    sig = Xf.max() + Yf.max()
    val = (-1+al+gg)*np.log(2) + kap*np.sum(p*p/kk) + 0.5*np.sum(v*v/kk) + sig
    return val, lam, sig

