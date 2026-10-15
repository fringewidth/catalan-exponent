"""
cat_cert7_new.py -- rigorous Section-7-type certificate for the real-place bound (Prop. 6.1, extended
to a general design) of the design (n, b, q, g, h) = (96, 15, 6, 8, 5), with the exact rational trial
sequences (finite part + geometric tails) read from the files in INFILES next to this script, each
certified in turn: cat_tail_opt_out.json (cat_tail_opt.py, 46 coefficients per sequence) and
cat_tail_edges_out.json (cat_tail_edges.py, bases placed at the support edges of the equilibrium
measures, 72 coefficients per sequence). A missing file is reported and skipped.

QUANTITY (kappa in {2, 1}; alpha = (b+q)/n, beta = b/n, gq = q/n, gg = g/n, eta = h/n):
  U_kappa = (-1 + alpha + gg) log 2 + kappa ||p||^2 + ||v||^2/2 + sup_{[-1,1]} X + sup_{(0,1)} Y,
  X(x) = (alpha + 2gg) log|x| + 2 eta log(1-x) - (kappa/2 + alpha + eta + gg) log(1+x^2)
         + lambda (2gg - 2x^2/(1+x^2)) - 2 kappa T(p,x) - S(v,x),
  Y(s) = beta log s + gq log(1-s) + 2 T(v,s),
  u_k = l_k [k <= D] + sum_z r_z z^k,  T(u,x) = sum_k u_k T_k(x)/k,  S(u,x) = sum_k u_k x^k/k,
  ||u||^2 = sum_k u_k^2/k.
  Closed forms: T(u,x) = sum_k l_k T_k(x)/k - (1/2) sum_z r_z Log(1 - 2xz + z^2),
                S(u,x) = sum_k l_k x^k/k - sum_z r_z Log(1 - xz),
                ||u||^2 = sum_{k<=D} (l_k^2 + 2 l_k t_k)/k - sum_{z,z'} r_z r_z' Log(1 - z z'),
                t_k = sum_z r_z z^k.
  A conjugate pair with coefficients (a, b) contributes a Re(z^k) + b Im(z^k): r_z = (a - ib)/2,
  r_conj(z) = conj(r_z).

METHOD (as in Section 7 of the paper, with interval arithmetic for every value)
  1. Exact derivative numerators: X' = A_X/Q_X, Y' = A_Y/Q_Y with A, Q polynomials over Q(i)
     (all divisions exact, imaginary parts cancel). Q_X = x(1-x)(1+x^2)^(3-kappa) prod(quadratic
     factors of the p-tails) prod(linear factors of the v-tails), Q_Y = s(1-s) prod(quadratic factors
     of the v-tails); every factor other than x(1-x) is positive on [-1,1], so Q has no zero in
     (-1,0), (0,1).
  2. All real roots of A_X in (-1,0) and (0,1) and of A_Y in (0,1) are isolated exactly (integer
     Descartes / Vincent-Collins-Akritas bisection) and refined by exact sign bisection to brackets
     of width 2^-50.
  3. X -> -infinity at 0 and at 1 (coefficients alpha+2gg > 0 of log|x| and 2 eta > 0 of log(1-x),
     all other terms bounded), Y -> -infinity at 0 and 1. Hence sup X = max(X(-1), max over the
     critical points) and sup Y = max over the critical points.
  4. On a bracket [lo, hi] containing a critical point, X <= X(lo) + w sup|X'|, w = hi - lo, with
     X' enclosed on [lo, hi] by interval arithmetic in its closed form (checked against A/Q in C2).
  5. X(lo), X(-1), Y(lo), the norms and log 2 are enclosed with mpmath interval arithmetic
     (iv.log, iv.atan2; 60 digits). The final constant uses upper endpoints throughout.
  6. f_low = -(1/n^2) int_0^H D - (1/2 + delta/2 + delta^2/8) log 2 is recomputed exactly
     (breakpoint integrator of cat_design_lower.py) and enclosed.

GATES (pre-registered)
  C1 exact numerators: divisions exact, imaginary parts zero                      [0 fails]
  C2 A/Q equals the numerical derivative of X, Y (mpmath, 40 digits) at 5 points per function,
     relative error < 1e-25; and equals the closed-form derivative, relative error < 1e-35                                                      [0 fails]
  C3 root isolation: the bisection ends with Descartes counts 0 or 1 on every piece (so all real
     roots are simple and each count-1 piece holds exactly one); every such piece, refined to
     width 2^-50, shows a strict sign change of A at its ends                     [0 fails]
  C4 interval widths: every enclosure (values, norms) narrower than 1e-30;
     every mean-value allowance w sup|X'| < 1e-9                                       [0 fails]
  C5 positive control: dense float grid (200,000 points) max of X, Y <= certified sup + 1e-12,
     and certified sup <= grid max + 1e-6                                          [0 fails]
  C6 f_low: the paper's design gives the integral 8609/2; DES gives f_low = -2.275548 +- 1e-6 [0 fails]
  C7 THE CLAIM: max(U_2, U_1) < f_low (upper endpoint < lower endpoint), for each input [0 fails]
REPORT: U_2, U_1, f_low, margin, number of critical points, numerator degrees.

SCOPE: this certifies the inequality for the stated design and sequences. That the bound of
Prop. 6.1 and the lower bound of Prop. 3.4 hold for this design in the form used here rests on
the generalisation worked out in the design study (cat_design_lower.py, cat_design_upper.py,
cat_newdesign_check.py) and still needs a written proof.

RUN: python3 cat_cert7_new.py   (no arguments; needs mpmath and numpy; single process)
TIME: about 10-20 s per input file.
OUTPUT: screen + cat_cert7_new_log_<YYYYMMDD_HHMMSS>.txt + cat_cert7_new_out.json, next to the script.
"""
import os, sys, json, time, math
from fractions import Fraction as Fr
try:
    sys.set_int_max_str_digits(0)
except AttributeError:
    pass
import mpmath as mpm
import numpy as np

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
STAMP = time.strftime('%Y%m%d_%H%M%S')
LOGPATH = os.path.join(HERE, 'cat_cert7_new_log_' + STAMP + '.txt')
LOGF = open(LOGPATH, 'w')
def log(s):
    line = f"[{time.time()-T0:6.1f}s] {s}"
    print(line, flush=True); LOGF.write(line + '\n'); LOGF.flush()
FAILS = {}
def gate(name, ok, info=""):
    FAILS[name] = FAILS.get(name, 0) + (0 if ok else 1)
    if not ok: log(f"FAIL {name} {info}")

INFILES = ['cat_tail_opt_out.json', 'cat_tail_edges_out.json']   # certified in turn; a missing file is reported
DES = (36, 6, 2, 3, 2)
FLOW_REF = -2.2787357
BRACKET_BITS = 50
C2DPS = 150
IVDPS = 60
NGRID = 200000

n0, b0, q0, g0, h0 = DES
ALPHA, BETA, GQ_, GG, ETA = Fr(b0 + q0, n0), Fr(b0, n0), Fr(q0, n0), Fr(g0, n0), Fr(h0, n0)

# ---------------------------------------------------------------- Q(i) arithmetic
class GQ:
    __slots__ = ('re', 'im')
    def __init__(s, re, im=Fr(0)): s.re = Fr(re); s.im = Fr(im)
    def __add__(s, o): o = o if isinstance(o, GQ) else GQ(o); return GQ(s.re+o.re, s.im+o.im)
    __radd__ = __add__
    def __neg__(s): return GQ(-s.re, -s.im)
    def __sub__(s, o): return s + (-(o if isinstance(o, GQ) else GQ(o)))
    def __mul__(s, o):
        o = o if isinstance(o, GQ) else GQ(o)
        return GQ(s.re*o.re - s.im*o.im, s.re*o.im + s.im*o.re)
    __rmul__ = __mul__
    def inv(s):
        d = s.re*s.re + s.im*s.im; return GQ(s.re/d, -s.im/d)
    def conj(s): return GQ(s.re, -s.im)
    def iszero(s): return s.re == 0 and s.im == 0
ZERO, ONE = GQ(0), GQ(1)
def trim(a):
    while len(a) > 1 and a[-1].iszero(): a = a[:-1]
    return a
def pmul(a, b):
    out = [GQ(0) for _ in range(len(a)+len(b)-1)]
    for i, x in enumerate(a):
        if x.iszero(): continue
        for j, y in enumerate(b): out[i+j] = out[i+j] + x*y
    return trim(out)
def padd(a, b):
    m = max(len(a), len(b))
    return trim([(a[i] if i < len(a) else ZERO) + (b[i] if i < len(b) else ZERO) for i in range(m)])
def pscale(a, c): return trim([c*x for x in a])
def pdiv_exact(a, d, tag):
    a = list(a); d = trim(d); q = [GQ(0) for _ in range(max(1, len(a)-len(d)+1))]
    li = d[-1].inv()
    for k in range(len(a)-len(d), -1, -1):
        coef = a[k+len(d)-1]*li; q[k] = coef
        for j, y in enumerate(d): a[k+j] = a[k+j] - coef*y
    gate("C1", all(x.iszero() for x in a), f"inexact division {tag}")
    return trim(q)
X1, OMX, OPX2 = [ZERO, ONE], [ONE, GQ(-1)], [ONE, ZERO, ONE]
def quad_fac(z): return [ONE + z*z, GQ(-2)*z]          # 1 - 2xz + z^2
def lin_fac(z): return [ONE, -z]                       # 1 - xz
def chebU(nmax):
    U = [[ONE], [ZERO, GQ(2)]]
    for k in range(2, nmax+1): U.append(padd(pmul([ZERO, GQ(2)], U[-1]), pscale(U[-2], GQ(-1))))
    return U

def seq(kap, which):
    """(finite l_1..l_D, [(z, r_z)] with conjugates expanded), exact."""
    c = [Fr(s) for s in DATA[f"kappa{kap}"]["cp" if which == 'p' else "cv"]]
    l = c[:DFIN]; terms = []
    for j, z in enumerate(RB): terms.append((GQ(z), GQ(c[DFIN + j])))
    for j, (zr, zi) in enumerate(PB):
        a, b = c[DFIN + len(RB) + 2*j], c[DFIN + len(RB) + 2*j + 1]
        z = GQ(zr, zi); r = GQ(a/2, -b/2)
        terms.append((z, r)); terms.append((z.conj(), r.conj()))
    return l, terms

def build(kap):
    lp, tp = seq(kap, 'p'); lv, tv = seq(kap, 'v'); lam = LAM[kap]
    QX = pmul(X1, OMX)
    for _ in range(3 - kap): QX = pmul(QX, OPX2)
    for z, _ in tp: QX = pmul(QX, quad_fac(z))
    for z, _ in tv: QX = pmul(QX, lin_fac(z))
    A = pscale(pdiv_exact(QX, X1, "QX/x"), GQ(ALPHA + 2*GG))
    A = padd(A, pscale(pdiv_exact(QX, OMX, "QX/(1-x)"), GQ(-2*ETA)))
    A = padd(A, pscale(pmul(X1, pdiv_exact(QX, OPX2, "QX/(1+x^2)")), GQ(-(kap + 2*ALPHA + 2*ETA + 2*GG))))
    if lam != 0:
        A = padd(A, pscale(pmul(X1, pdiv_exact(pdiv_exact(QX, OPX2, "a"), OPX2, "QX/(1+x^2)^2")), GQ(-4*lam)))
    polyt = [ZERO]
    for k, l in enumerate(lp, start=1): polyt = padd(polyt, pscale(UP[k-1], GQ(l)))
    A = padd(A, pscale(pmul(polyt, QX), GQ(-2*kap)))
    for z, r in tp: A = padd(A, pscale(pdiv_exact(QX, quad_fac(z), "QX/quad"), GQ(-2*kap)*r*z))
    polyh = [GQ(l) for l in lv]
    A = padd(A, pscale(pmul(polyh, QX), GQ(-1)))
    for z, r in tv: A = padd(A, pscale(pdiv_exact(QX, lin_fac(z), "QX/lin"), -(r*z)))
    QY = pmul(X1, OMX)
    for z, _ in tv: QY = pmul(QY, quad_fac(z))
    B = pscale(pdiv_exact(QY, X1, "QY/s"), GQ(BETA))
    B = padd(B, pscale(pdiv_exact(QY, OMX, "QY/(1-s)"), GQ(-GQ_)))
    polytv = [ZERO]
    for k, l in enumerate(lv, start=1): polytv = padd(polytv, pscale(UP[k-1], GQ(l)))
    B = padd(B, pscale(pmul(polytv, QY), GQ(2)))
    for z, r in tv: B = padd(B, pscale(pdiv_exact(QY, quad_fac(z), "QY/quad"), GQ(2)*r*z))
    out = {}
    for name, P, Q in (('X', A, QX), ('Y', B, QY)):
        gate("C1", all(c.im == 0 for c in P) and all(c.im == 0 for c in Q), f"imaginary part k={kap} {name}")
        out[name] = ([c.re for c in P], [c.re for c in Q])
    return out

def peval(P, x):
    acc = Fr(0)
    for c in reversed(P): acc = acc*x + c
    return acc

# ---------------------------------------------------------------- interval evaluation
iv = mpm.iv; iv.dps = IVDPS
def ivq(q): return iv.mpf(q.numerator)/q.denominator
LOG2 = iv.log(iv.mpf(2))
class IC:
    """complex interval as a pair of real intervals."""
    __slots__ = ('re', 'im')
    def __init__(s, re, im=None): s.re = re; s.im = iv.mpf(0) if im is None else im
    def __add__(s, o): return IC(s.re + o.re, s.im + o.im)
    def __mul__(s, o): return IC(s.re*o.re - s.im*o.im, s.re*o.im + s.im*o.re)
    def inv(s):
        d = s.re*s.re + s.im*s.im
        return IC(s.re/d, -s.im/d)
    def log(s):
        """principal Log; used only where the argument has positive real part or does not cross
        the negative real axis (see the docstring of featT)."""
        return IC(iv.log(s.re*s.re + s.im*s.im)/2, iv.atan2(s.im, s.re))
def icq(z): return IC(ivq(z.re), ivq(z.im))
def hi(v):
    """upper endpoint as an mpf (exact: converted at a precision above the interval precision)."""
    with mpm.workdps(IVDPS + 30): return +mpm.mpf(v.b)
def lo_(v):
    with mpm.workdps(IVDPS + 30): return +mpm.mpf(v.a)
def ivwidth(v):
    with mpm.workdps(IVDPS + 30): return mpm.mpf(v.b) - mpm.mpf(v.a)

def make_funcs(kap, with_norm=True):
    lp, tp = seq(kap, 'p'); lv, tv = seq(kap, 'v')
    lpi, lvi = [ivq(l) for l in lp], [ivq(l) for l in lv]
    tpi = [(icq(z), icq(r)) for z, r in tp]; tvi = [(icq(z), icq(r)) for z, r in tv]
    lam = ivq(LAM[kap]); al, be, gq, gg, et = (ivq(q) for q in (ALPHA, BETA, GQ_, GG, ETA))
    one = iv.mpf(1); two = iv.mpf(2); half = one/2
    def T(l, terms, x):
        t0, t1 = one, x; s = iv.mpf(0)
        for k in range(1, len(l)+1):
            s += l[k-1]*t1/k; t0, t1 = t1, 2*x*t1 - t0
        for z, r in terms:
            # 1 - 2xz + z^2 = (1 - z e^{i th})(1 - z e^{-i th}); each factor has positive real part,
            # so the principal Log of the product equals the sum of the two principal Logs.
            w = IC(one) + IC(-2*x)*z + z*z
            s += -half*(r*w.log()).re
        return s
    def S(l, terms, x):
        s = iv.mpf(0); xp = one
        for k in range(1, len(l)+1):
            xp = xp*x; s += l[k-1]*xp/k
        for z, r in terms:
            w = IC(one) + IC(-x)*z                       # real part 1 - x Re z > 0
            s += -(r*w.log()).re
        return s
    def X(x):
        x = iv.mpf(x) if not isinstance(x, type(one)) else x
        W = (al + 2*gg)*iv.log(abs(x)) + 2*et*iv.log(1 - x) - (iv.mpf(kap)/2 + al + et + gg)*iv.log(1 + x*x)
        return W + lam*(2*gg - 2*x*x/(1 + x*x)) - 2*kap*T(lpi, tpi, x) - S(lvi, tvi, x)
    def Y(x):
        x = iv.mpf(x) if not isinstance(x, type(one)) else x
        return be*iv.log(x) + gq*iv.log(1 - x) + 2*T(lvi, tvi, x)
    def dT(l, terms, x):
        """d/dx T(u,x) = sum_k l_k U_{k-1}(x) + sum_z Re(r_z z/(1 - 2xz + z^2))."""
        u0, u1 = one, 2*x; s = l[0]*u0 if l else iv.mpf(0)
        for k in range(2, len(l)+1):
            s += l[k-1]*u1; u0, u1 = u1, 2*x*u1 - u0
        for z, r in terms:
            w = IC(one) + IC(-2*x)*z + z*z
            s += (r*z*w.inv()).re
        return s
    def dS(l, terms, x):
        s = iv.mpf(0); xp = one
        for k in range(1, len(l)+1):
            s += l[k-1]*xp; xp = xp*x
        for z, r in terms:
            w = IC(one) + IC(-x)*z
            s += (r*z*w.inv()).re
        return s
    def dX(x):
        return (al + 2*gg)/x - 2*et/(1 - x) - (iv.mpf(kap)/2 + al + et + gg)*2*x/(1 + x*x) \
               - lam*4*x/((1 + x*x)*(1 + x*x)) - 2*kap*dT(lpi, tpi, x) - dS(lvi, tvi, x)
    def dY(x):
        return be/x - gq/(1 - x) + 2*dT(lvi, tvi, x)
    if not with_norm: return X, Y, dX, dY
    def norm2(l, terms):
        s = iv.mpf(0)
        for k in range(1, len(l)+1):
            tk = IC(iv.mpf(0))
            for z, r in terms:
                zk = IC(one)
                for _ in range(k): zk = zk*z
                tk = tk + r*zk
            s += (l[k-1]*l[k-1] + 2*l[k-1]*tk.re)/k
        for z, r in terms:
            for zz, rr in terms:
                w = IC(one) + IC(-one)*(z*zz)              # |z z'| < 1: positive real part
                s += -((r*rr)*w.log()).re
        return s
    NP2 = kap*norm2(lpi, tpi) + norm2(lvi, tvi)/2
    return X, Y, dX, dY, NP2

# ---------------------------------------------------------------- exact real-root isolation
def to_int(P):
    den = 1
    for c in P: den = den*c.denominator//math.gcd(den, c.denominator)
    return [int(c*den) for c in P]
def shift(c, s):
    """coefficients of P(y + s) from those of P(y) (integer s)."""
    c = list(c); m = len(c)
    for i in range(m):
        for j in range(m-2, i-1, -1): c[j] += s*c[j+1]
    return c
def var01(c):
    """Descartes sign variations of P on (0, 1)."""
    r = shift(c[::-1], 1); sg = [1 if v > 0 else -1 for v in r if v != 0]
    return sum(1 for i in range(1, len(sg)) if sg[i] != sg[i-1])
def isolate(Pint, a, b):
    """isolating intervals (Fractions) for the real roots of Pint in the open interval (a, b),
    a, b integers with b - a = 1; exact roots returned as degenerate intervals."""
    c = shift(Pint, a); out = []
    stack = [(c, Fr(a), Fr(1))]
    while stack:
        c, lo, w = stack.pop()
        while len(c) > 1 and c[0] == 0:          # root at the left end (only at interior midpoints)
            if lo != a: out.append((lo, lo))
            c = c[1:]
        v = var01(c)
        if v == 0: continue
        if v == 1: out.append((lo, lo + w)); continue
        d = len(c) - 1
        left = [c[i] << (d - i) for i in range(d + 1)]      # 2^d P(y/2)
        right = shift(left, 1)                               # 2^d P((y+1)/2)
        if right[0] == 0:                                    # exact root at the midpoint
            out.append((lo + w/2, lo + w/2)); right = right[1:]
        stack.append((left, lo, w/2)); stack.append((right, lo + w/2, w/2))
    return sorted(out)
def sgn_at(Pint, x):
    """sign of P(x) for rational x: sign of sum c_i num^i den^(d-i) (den > 0)."""
    num, den = x.numerator, x.denominator; d = len(Pint) - 1
    acc = 0
    for i in range(d + 1): acc += Pint[i]*num**i*den**(d - i)
    return (acc > 0) - (acc < 0)
def refine(Pint, lo, hi, bits):
    """returns (lo, hi, ok): bracket of width <= 2^-bits with a strict sign change (ok), or an
    exact root (lo = hi, ok)."""
    if lo == hi: return lo, hi, sgn_at(Pint, lo) == 0
    slo, shi = sgn_at(Pint, lo), sgn_at(Pint, hi)
    if slo*shi >= 0: return lo, hi, False
    while hi - lo > Fr(1, 2**bits):
        mid = (lo + hi)/2; sm = sgn_at(Pint, mid)
        if sm == 0: return mid, mid, True
        if sm == slo: lo = mid
        else: hi = mid
    return lo, hi, True

# ---------------------------------------------------------------- f_low (exact)
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
def integral_D(n, b, q, g, h):
    a, L, C, H, A = derived(n, b, q, g, h)
    f = lambda x: Fr(Dfun(x, n, b, q, g, h))
    pts = breakpoints(n, b, q, g, h); I = Fr(0); unresolved = 0
    stack = [(pts[i], pts[i+1], 0) for i in range(len(pts)-1)][::-1]
    while stack:
        lo, hi, depth = stack.pop()
        if hi <= lo: continue
        w = hi - lo; xs = [lo + w*Fr(k, 8) for k in range(1, 8)]; ys = [f(x) for x in xs]
        slope = (ys[6] - ys[0])/(xs[6] - xs[0])
        lin = all(ys[k] == ys[0] + slope*(xs[k] - xs[0]) for k in range(7))
        if not lin and depth < 60:
            mid = (lo + hi)/2; stack.append((mid, hi, depth+1)); stack.append((lo, mid, depth+1)); continue
        if not lin: unresolved += 1
        I += w*ys[3]
    gate("C6", unresolved == 0, f"{(n,b,q,g,h)}: {unresolved} unresolved pieces")
    return I
Ipap = integral_D(48, 7, 4, 4, 2)
gate("C6", Ipap == Fr(8609, 2), f"paper's design: integral {Ipap}")
Ides = integral_D(*DES)
dl = Fr(q0 + g0 + h0, n0)
FLOW = -ivq(Ides)/n0**2 - ivq(Fr(1, 2) + dl/2 + dl*dl/8)*LOG2
gate("C6", abs(float(lo_(FLOW)) - FLOW_REF) < 1e-6, f"f_low {FLOW}")
log(f"C6 f_low: paper's integral {Ipap}; design {DES}: integral {Ides}, f_low in [{mpm.nstr(lo_(FLOW), 12)}, {mpm.nstr(hi(FLOW), 12)}]")

# ---------------------------------------------------------------- main
def certify(infile):
    """certificate for the sequences in the JSON file `name` (output of cat_tail_opt.py or cat_tail_edges.py)."""
    global DATA, DFIN, RB, PB, UP, LAM
    with open(os.path.join(HERE, infile)) as fh: DATA = json.load(fh)
    gate("C6", tuple(DATA["design"]) == DES, f"{infile}: input design {DATA['design']}")
    DFIN = DATA["D"]
    RB = [Fr(z) for z in DATA["real_bases"]]
    PB = [(Fr(a), Fr(b)) for a, b in DATA["pair_bases"]]
    UP = chebU(DFIN + 1)
    LAM = {k: Fr(DATA[f"kappa{k}"]["lambda"]) for k in (2, 1)}
    gate("C1", LAM[1] >= 0 and LAM[2] == 0, f"{infile}: lambda {LAM}")
    log(f"== input {infile}: design {DES}, D = {DFIN}, {len(RB)} real tails, {len(PB)} pairs")
    RES = {}
    for kap in (2, 1):
        tk = time.time()
        NUM = build(kap)
        log(f"kappa={kap}: exact numerators built: degrees A_X {len(NUM['X'][0])-1}, Q_X {len(NUM['X'][1])-1}, "
            f"A_Y {len(NUM['Y'][0])-1}, Q_Y {len(NUM['Y'][1])-1} ({time.time()-tk:.1f}s)")
        # C2 derivative check: (i) exact A/Q against the numerical derivative of X, Y (mpmath diff at
        # 40 digits on functions evaluated at 150 digits); (ii) exact A/Q against the closed-form
        # derivative used for the mean-value allowances.
        iv.dps = C2DPS; Xh, Yh, dXh, dYh = make_funcs(kap, with_norm=False)
        mpm.mp.dps = 40
        mid = lambda v: (mpm.mpf(v.a) + mpm.mpf(v.b))/2
        worst1 = worst2 = mpm.mpf(0)
        for name, pts, F, dF in (('X', ['-0.73', '-0.21', '0.17', '0.52', '0.91'], Xh, dXh),
                                 ('Y', ['0.07', '0.33', '0.5', '0.68', '0.97'], Yh, dYh)):
            P, Q = NUM[name]
            for st in pts:
                x = Fr(st); e = peval(P, x)/peval(Q, x)
                exact = mpm.mpf(e.numerator)/e.denominator
                num = mpm.diff(lambda t: mid(F(iv.mpf(t))), mpm.mpf(x.numerator)/x.denominator)
                rel1 = abs(exact - num)/max(abs(num), mpm.mpf(1)); worst1 = max(worst1, rel1)
                gate("C2", rel1 < mpm.mpf('1e-25'), f"k={kap} {name} x={st} diff rel={mpm.nstr(rel1, 3)}")
                cf = mid(dF(ivq(x)))
                rel2 = abs(exact - cf)/max(abs(cf), mpm.mpf(1)); worst2 = max(worst2, rel2)
                gate("C2", rel2 < mpm.mpf('1e-35'), f"k={kap} {name} x={st} closed-form rel={mpm.nstr(rel2, 3)}")
        iv.dps = IVDPS; mpm.mp.dps = IVDPS + 30
        X, Y, dX, dY, NP2 = make_funcs(kap)
        log(f"  C2 A/Q vs numerical derivative: worst rel {mpm.nstr(worst1, 3)}; vs closed-form derivative: {mpm.nstr(worst2, 3)}")
        # C3 critical points
        sups = {}; ncrit = {}; allow_max = 0; width_max = mpm.mpf(0)
        for name, F, dF, ranges in (('X', X, dX, [(-1, 0), (0, 1)]), ('Y', Y, dY, [(0, 1)])):
            P, Q = NUM[name]; Pint = to_int(P)
            best = None; cnt = 0
            if name == 'X':
                v = F(iv.mpf(-1)); best = v; width_max = max(width_max, ivwidth(v))
            for a, b in ranges:
                ivs = isolate(Pint, a, b)
                for blo, bhi in ivs:
                    blo, bhi, ok = refine(Pint, blo, bhi, BRACKET_BITS)
                    gate("C3", ok, f"k={kap} {name}: no sign change on [{float(blo)}, {float(bhi)}]")
                    cnt += 1
                    w = bhi - blo
                    # mean value: X(x) <= X(blo) + w sup|X'| on [blo, bhi]; X' enclosed in interval arithmetic
                    d = dF(iv.mpf([ivq(blo).a, ivq(bhi).b]))
                    allow = ivq(w)*iv.mpf(max(abs(d.a), abs(d.b)))
                    allow_max = max(allow_max, float(hi(allow)))
                    v = F(ivq(blo)); width_max = max(width_max, ivwidth(v))
                    up = v + allow
                    if best is None or hi(up) > hi(best): best = up
            sups[name] = best; ncrit[name] = cnt
        gate("C4", width_max < mpm.mpf('1e-30'), f"k={kap}: enclosure width {mpm.nstr(width_max, 3)}")
        gate("C4", allow_max < 1e-9, f"k={kap}: mean-value allowance {allow_max:.3e}")
        gate("C4", ivwidth(NP2) < mpm.mpf('1e-30'), f"k={kap}: norm enclosure width {mpm.nstr(ivwidth(NP2), 3)}")
        U = (-1 + ivq(ALPHA + GG))*LOG2 + NP2 + sups['X'] + sups['Y']
        log(f"  C3 critical points: X {ncrit['X']}, Y {ncrit['Y']}; max mean-value allowance {float(allow_max):.2e}; "
            f"max enclosure width {mpm.nstr(width_max, 3)}")
        log(f"  sup X <= {mpm.nstr(hi(sups['X']), 15)}, sup Y <= {mpm.nstr(hi(sups['Y']), 15)}, "
            f"kappa||p||^2 + ||v||^2/2 <= {mpm.nstr(hi(NP2), 15)}")
        log(f"  U_{kap} <= {mpm.nstr(hi(U), 15)}   (cat_tail_opt.py grid value {DATA[f'kappa{kap}']['value']:.9f})")
        # C5 dense float grid control
        xs = np.cos(np.linspace(0, np.pi, NGRID + 2)[1:-1]); xs = xs[np.abs(xs) > 1e-7]
        ss = (1 - np.cos(np.linspace(0, np.pi, NGRID + 2)[1:-1]))/2
        lp, tp = seq(kap, 'p'); lv, tv = seq(kap, 'v')
        def Tf(l, terms, x):
            t0, t1 = np.ones_like(x), x.copy(); s = np.zeros_like(x)
            for k in range(1, len(l)+1): s += float(l[k-1])*t1/k; t0, t1 = t1, 2*x*t1 - t0
            for z, r in terms:
                zc = complex(float(z.re), float(z.im)); rc = complex(float(r.re), float(r.im))
                s += (-0.5*rc*np.log(1 - 2*x*zc + zc*zc + 0j)).real
            return s
        def Sf(l, terms, x):
            s = np.zeros_like(x)
            for k in range(1, len(l)+1): s += float(l[k-1])*x**k/k
            for z, r in terms:
                zc = complex(float(z.re), float(z.im)); rc = complex(float(r.re), float(r.im))
                s += (-rc*np.log(1 - x*zc + 0j)).real
            return s
        al, be, gq, gg, et = (float(q) for q in (ALPHA, BETA, GQ_, GG, ETA)); lam = float(LAM[kap])
        Xg = (al + 2*gg)*np.log(np.abs(xs)) + 2*et*np.log(1 - xs) - (kap/2 + al + et + gg)*np.log(1 + xs*xs) \
             + lam*(2*gg - 2*xs*xs/(1 + xs*xs)) - 2*kap*Tf(lp, tp, xs) - Sf(lv, tv, xs)
        Yg = be*np.log(ss) + gq*np.log(1 - ss) + 2*Tf(lv, tv, ss)
        for name, g in (('X', Xg.max()), ('Y', Yg.max())):
            cs = float(hi(sups[name]))
            gate("C5", g <= cs + 1e-12 and cs <= g + 1e-6, f"k={kap} {name}: grid {g:.12f} certified {cs:.12f}")
        log(f"  C5 grid maxima X {Xg.max():.12f}, Y {Yg.max():.12f}  ({time.time()-tk:.1f}s for kappa={kap})")
        RES[kap] = {"U_upper": mpm.nstr(hi(U), 20), "supX_upper": mpm.nstr(hi(sups['X']), 20),
                    "supY_upper": mpm.nstr(hi(sups['Y']), 20), "norms_upper": mpm.nstr(hi(NP2), 20),
                    "critical_points": ncrit, "max_allowance": float(allow_max),
                    "degrees": {k: [len(NUM[k][0]) - 1, len(NUM[k][1]) - 1] for k in NUM}, "_U": U}

    Umax = RES[2]["_U"] if hi(RES[2]["_U"]) >= hi(RES[1]["_U"]) else RES[1]["_U"]
    margin = lo_(FLOW) - hi(Umax)
    gate("C7", margin > 0, f"max U {mpm.nstr(hi(Umax), 15)} not below f_low {mpm.nstr(lo_(FLOW), 15)}")
    log(f"C7 CLAIM ({infile}): max(U_2, U_1) <= {mpm.nstr(hi(Umax), 15)} < f_low >= {mpm.nstr(lo_(FLOW), 15)}; margin {mpm.nstr(margin, 8)}")

    for k in RES: RES[k].pop("_U")
    return {"input": infile, "kappa2": RES[2], "kappa1": RES[1], "margin_lower": mpm.nstr(margin, 12)}

OUTS = []
for name in INFILES:
    if not os.path.exists(os.path.join(HERE, name)):
        log(f"== input {name}: not found next to the script, skipped"); continue
    OUTS.append(certify(name))
gate("C6", len(OUTS) > 0, "no input file found")
total = sum(FAILS.values())
out = {"script": "cat_cert7_new.py", "design": DES, "f_low_integral": str(Ides),
       "f_low_lower": mpm.nstr(lo_(FLOW), 20), "certificates": OUTS, "gates": FAILS, "GATE_FAILS": total,
       "seconds": round(time.time() - T0, 1)}
with open(os.path.join(HERE, "cat_cert7_new_out.json"), "w") as fh: json.dump(out, fh, indent=1)
log(f"gates: {FAILS}")
print(f"GATE_FAILS = {total}")
LOGF.write(f"GATE_FAILS = {total}\n"); LOGF.close()
print(f"log saved to {LOGPATH}")
