"""
cat_newdesign_check.py -- checks 1a and 1b for the design with the largest provable margin found by
cat_design_search.py: (n, b, q, g, h) = (96, 15, 6, 8, 5) per unit N
(a = b + q = 21, L = 117, C = 125, H = 130, A = a + 2g = 37).

GATES (pre-registered)
  S  structural conditions used in Sections 3-4 of the paper hold for this design:
     g <= (n-1)/2; b < A; L < H; b < H/2; row supports inside [A, H-1] (rows 0..n-1, N = 1);
     auxiliary row n: contact order C + n - g >= L; counting identity R0 = R (Lemma 3.3) for
     N = 1..60 and every prime H/2 < p <= H                                                  [0 fails]
  Z  zeta(2) coefficient of every raw entry is exactly 0, rows 0..n (including the auxiliary
     row used in Section 4), columns b <= j < L, N = 1                                       [0 fails]
  B  Section 4 certificate for this design: the (n+1) x n rational matrix B of eq. (47),
     rebuilt for (96,15,6,8,5); every denominator prime <= H = 130; B0, B0 + B1, B0 - B1 have full
     rank modulo 131, hence nonzero rational determinants (the paper's argument with 131 for 101) [0 fails]
  V  end-to-end per-prime bound at N = 1 (Delta of size 96) with the hypothetical value G = 1/3:
     for every prime 2 sqrt(H) < p <= H, -v_p(Delta) <= D(p) with D the generalised loss of
     cat_design_lower.py (eqs. 38, 40); primes H < p < H + 60: v_p >= 0                       [0 fails]
REPORT: denominators' largest prime in B; determinant sizes; slack per prime.

RUN: python3 cat_newdesign_check.py   (no arguments; single process; needs mpmath only)
TIME (estimate): S, Z < 1 s; B ~10 s; V ~6 min (cat_padic.py took 5.5 min per G value at size 96).
OUTPUT: screen + cat_newdesign_check_log_<YYYYMMDD_HHMMSS>.txt + cat_newdesign_check_out.json, next to the script.
"""
import sys, json, time, math
from fractions import Fraction as Fr
from math import comb, gcd
try:
    sys.set_int_max_str_digits(0)
except AttributeError:
    pass
import mpmath as mpm




DES = (36, 6, 2, 3, 2)
GV = Fr(1, 3)
PCERT = 53
LOG2 = math.log(2)
T0 = time.time()
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
STAMP = time.strftime('%Y%m%d_%H%M%S')
LOGPATH = _os.path.join(HERE, 'cat_newdesign_check_log_' + STAMP + '.txt')
LOGF = open(LOGPATH, 'w')
def log(s):
    line = f"[{time.time()-T0:7.1f}s] {s}"
    print(line, flush=True)
    LOGF.write(line + '\n'); LOGF.flush()
FAILS = {}
def gate(name, ok, info=""):
    FAILS[name] = FAILS.get(name, 0) + (0 if ok else 1)
    if not ok: log(f"FAIL {name} {info}")

def c(l): return Fr(comb(2*l, l), 4**l)
def vp(x, p):
    if x == 0: return 10**9
    v = 0; a, b = x.numerator, x.denominator
    while a % p == 0: a //= p; v += 1
    while b % p == 0: b //= p; v -= 1
    return v
def primes_upto(m):
    s = [True]*(m+1); s[0] = s[1] = False
    for i in range(2, int(m**0.5)+1):
        if s[i]:
            for j in range(i*i, m+1, i): s[j] = False
    return [i for i in range(m+1) if s[i]]

def arrays(Imax, Jmax):
    """m_i (i <= Imax), k-_d (d <= Imax), k+_d (d <= Jmax+1), M0[i][j], Z0 via B1, B2."""
    top = max(Imax, Jmax) + 2
    m = [Fr(0)]*(top+1); m[0] = Fr(2)
    for i in range(2, top+1, 2): m[i] = i*m[i-2]/(i+1)
    km = [Fr(0)]*(top+1); km[1] = Fr(2)
    for d in range(2, top+1): km[d] = ((d-1)*km[d-2] + m[d-2] + m[d-1])/d
    kp = [Fr(0)]*(top+1)
    for d in range(2, top+1): kp[d] = ((d-2)*kp[d-2] + Fr(2, d-1))/(d-1)
    M = [[Fr(0)]*(Jmax+1) for _ in range(Imax+1)]
    for i in range(Imax+1): M[i][0] = km[i]
    for j in range(Jmax+1): M[0][j] = kp[j]
    for i in range(1, Imax+1):
        Mi, Mp, mi = M[i], M[i-1], m[i-1]
        for j in range(1, Jmax+1): Mi[j] = Mp[j-1] - mi/j
    B1 = [Fr(0)]*(top+1); B2 = [Fr(0)]*(top+1)
    for i in range(1, top+1): B1[i] = B1[i-1] + Fr(1, i); B2[i] = B2[i-1] + Fr(1, i*i)
    def Z0(i, j): return -B2[i] if i == j else (B1[i]-B1[j])/(i-j)
    return m, km, kp, M, Z0


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


def build(N, des):
    n0, b0, q0, g0, h0 = des
    n, b, q, g, h = n0*N, b0*N, q0*N, g0*N, h0*N
    a = b + q; L = n + a; C = L + g; H = C + h; A = a + 2*g
    m, km, kp, M, Z0 = arrays(H, L)
    # Chebyshev T_d, U_{d-1} as coefficient lists in x
    T = [[1], [0, 1]]; U = [[1], [0, 2]]
    for k in range(2, n+2):
        for S in (T, U):
            a1 = [0] + [2*v for v in S[-1]]; a0 = S[-2]
            S.append([(a1[i] if i < len(a1) else 0) - (a0[i] if i < len(a0) else 0) for i in range(len(a1))])
    binh = [(-1)**v*comb(h, v) for v in range(h+1)]
    def rowpoly(coeffs_x, sign):      # (1-t)^h t^(C-1) Poly(1/t)
        base = {}
        for k, v in enumerate(coeffs_x):
            if v: base[C-1-k] = sign*v
        out = {}
        for e, v in base.items():
            for s, bv in enumerate(binh):
                out[e+s] = out.get(e+s, 0) + v*bv
        return {e: v for e, v in out.items() if v}
    Pr, Dr = [], []
    for r in range(n):
        d = abs(r-g); sg = (r > g) - (r < g)
        Pr.append(rowpoly(T[d], 1))
        Dr.append(rowpoly(U[d-1], sg) if d >= 1 else {})
    # raw columns j in [b, L): rational part and G-coefficient
    Zc = {}
    def Z0c(i, j):
        key = (i, j)
        if key not in Zc: Zc[key] = Z0(i, j)
        return Zc[key]
    raw_rat, raw_G = [], []
    for r in range(n):
        rr, rg = [], []
        for j in range(b, L):
            s = Fr(0); sG = Fr(0)
            for i, v in Pr[r].items():
                s += v*M[i][j]
                if (i-j) % 2 == 0 and i >= j: sG += v*c((i-j)//2)
            for i, v in Dr[r].items():
                s -= Fr(3, 2)*v*Z0c(i, j)
            rr.append(s); rg.append(4*sG)
        raw_rat.append(rr); raw_G.append(rg)
    # filter: column k = sum_j (-1)^(j-b-k) C(q, j-b-k) raw_j
    Frat = [[sum(((-1)**t)*comb(q, t)*raw_rat[r][k+t] for t in range(q+1)) for k in range(n)] for r in range(n)]
    FG = [[sum(((-1)**t)*comb(q, t)*raw_G[r][k+t] for t in range(q+1)) for k in range(n)] for r in range(n)]
    return dict(n=n, a=a, b=b, q=q, g=g, h=h, L=L, C=C, H=H, A=A, Frat=Frat, FG=FG, Pr=Pr, Dr=Dr)




def vp_det(Mq, p):
    """exact p-adic valuation of det of a rational matrix (full pivoting over Z/p^K)."""
    n = len(Mq)
    s = -min(min(vp(x, p) for x in row) for row in Mq)
    s = max(s, 0)
    K = n*s + 3*n + 20; MOD = p**K
    A = []
    for row in Mq:
        rr = []
        for x in row:
            if x == 0: rr.append(0); continue
            num, den = x.numerator*(p**s), x.denominator
            while den % p == 0:
                den //= p; num //= p   # num divisible by p here since x*p^s is p-integral
            rr.append(num*pow(den, -1, MOD) % MOD)
        A.append(rr)
    def val(x):
        if x == 0: return K
        v = 0
        while x % p == 0: x //= p; v += 1
        return v
    tot = 0
    rows = list(range(n)); cols = list(range(n))
    for step in range(n):
        best = (K+1, None, None)
        for i in range(step, n):
            Ai = A[i]
            for j in range(step, n):
                x = Ai[j]
                if x:
                    v = val(x)
                    if v < best[0]:
                        best = (v, i, j)
                        if v == 0: break
            if best[0] == 0: break
        v, bi, bj = best
        if bi is None:                       # remaining block is 0 mod p^K
            return tot + (n-step)*K - n*s, True
        A[step], A[bi] = A[bi], A[step]
        for row in A: row[step], row[bj] = row[bj], row[step]
        tot += v
        piv = A[step][step]; u = piv // p**v; uinv = pow(u, -1, MOD)
        prow = A[step]
        for i in range(step+1, n):
            x = A[i][step]
            if x:
                f = (x // p**v) * uinv % MOD
                Ai = A[i]
                A[i] = [(Ai[k] - f*prow[k]) % MOD for k in range(n)]
    return tot - n*s, False




n0, b0, q0, g0, h0 = DES
a0, L0, C0, H0, A0 = derived(*DES)
log(f"design {DES}: a={a0}, L={L0}, C={C0}, H={H0}, A={A0}")

# ------------------------------------------------------------------ S structural
gate("S", 2*g0 <= n0 - 1, "g > (n-1)/2")
gate("S", b0 < A0, "b >= A")
gate("S", L0 < H0, "L >= H")
gate("S", 2*b0 < H0, "b >= H/2")
gate("S", C0 + n0 - g0 >= L0, "auxiliary row contact")
nB = 0
for N in range(1, 61):
    n, b, q, g, h = (x*N for x in DES)
    a, L, C, H, A = derived(n, b, q, g, h)
    for p in primes_upto(H):
        if 2*p <= H: continue
        K, J = L-p, H-p
        Bp = pos(min(A, K) - b)
        R0 = pos(K) + pos(J-b) - Bp
        R = pos(K) + pos(K-A) + pos(J - max(b, K))
        gate("S", R0 == R, f"N={N} p={p} R0={R0} R={R}"); nB += 1
log(f"S: basic conditions checked; R0 = R in {nB} (N,p) cases")

# ------------------------------------------------------------------ rows 0..n (N = 1) and arrays
m, km, kp, M, Z0 = arrays(H0, L0)
T = [[1], [0, 1]]; U = [[1], [0, 2]]
for k in range(2, n0 + 3):
    for Sq in (T, U):
        a1 = [0] + [2*v for v in Sq[-1]]; a_ = Sq[-2]
        Sq.append([(a1[i] if i < len(a1) else 0) - (a_[i] if i < len(a_) else 0) for i in range(len(a1))])
binh = [(-1)**v*comb(h0, v) for v in range(h0+1)]
def rowpoly(coeffs_x, sign):
    out = {}
    for k, v in enumerate(coeffs_x):
        if v:
            for s, bv in enumerate(binh):
                e = C0 - 1 - k + s
                out[e] = out.get(e, 0) + sign*v*bv
    return {e: v for e, v in out.items() if v}
Pr, Dr = [], []
for r in range(n0 + 1):
    d = abs(r - g0); sg = (r > g0) - (r < g0)
    Pr.append(rowpoly(T[d], 1)); Dr.append(rowpoly(U[d-1], sg) if d >= 1 else {})
for r in range(n0):
    degs = list(Pr[r]) + list(Dr[r])
    gate("S", min(degs) >= A0 and max(degs) <= H0 - 1, f"row {r} support [{min(degs)},{max(degs)}]")
log(f"S: row supports inside [A, H-1] = [{A0}, {H0-1}] for rows 0..{n0-1}; auxiliary row {n0}: "
    f"[{min(list(Pr[n0])+list(Dr[n0]))}, {max(list(Pr[n0])+list(Dr[n0]))}]")

# ------------------------------------------------------------------ Z contact
bad = 0
for r in range(n0 + 1):
    for j in range(b0, L0):
        s = Fr(0)
        for i, v in Pr[r].items():
            if i < j and (j-i-1) % 2 == 0: s += v*c((j-i-1)//2)
        s -= Dr[r].get(j, 0)
        if s != 0: bad += 1
gate("Z", bad == 0, f"{bad} entries with nonzero zeta(2) coefficient")
log(f"Z: zeta(2) coefficient exactly 0 in all {(n0+1)*(L0-b0)} raw entries (rows 0..{n0})" if bad == 0 else f"Z: {bad} failures")

# ------------------------------------------------------------------ B certificate
B1h = {}
def Z0c(i, j):
    if (i, j) not in B1h: B1h[(i, j)] = Z0(i, j)
    return B1h[(i, j)]
Bm = []
for r in range(n0 + 1):
    raw = []
    for j in range(b0, b0 + n0 + q0):
        s = Fr(0)
        for i, v in Pr[r].items(): s += v*M[i][j]
        for i, v in Dr[r].items(): s -= Fr(3, 2)*v*Z0c(i, j)
        raw.append(s)
    Bm.append([sum(((-1)**t)*comb(q0, t)*raw[k+t] for t in range(q0+1)) for k in range(n0)])
log(f"B: ({n0+1} x {n0}) matrix built")
def pf(nn):
    nn = abs(nn); ps = set(); p = 2
    while p*p <= nn:
        while nn % p == 0: ps.add(p); nn //= p
        p += 1
    if nn > 1: ps.add(nn)
    return ps
maxp = max((max(pf(x.denominator)) if x.denominator > 1 else 1) for row in Bm for x in row)
gate("B", maxp <= H0, f"denominator prime {maxp} > H")
gate("B", PCERT > H0, "certificate prime not above H")
log(f"B: largest denominator prime {maxp} (<= H = {H0})")
def det_exact(Mq):
    nn = len(Mq); A_ = []; scale = Fr(1)
    for row in Mq:
        l = 1
        for x in row: l = l*x.denominator//gcd(l, x.denominator)
        A_.append([int(x*l) for x in row]); scale /= l
    sign = 1; prev = 1
    for k in range(nn-1):
        if A_[k][k] == 0:
            sw = next((i for i in range(k+1, nn) if A_[i][k] != 0), None)
            if sw is None: return Fr(0)
            A_[k], A_[sw] = A_[sw], A_[k]; sign = -sign
        for i in range(k+1, nn):
            for j in range(k+1, nn):
                A_[i][j] = (A_[i][j]*A_[k][k] - A_[i][k]*A_[k][j])//prev
        prev = A_[k][k]
    return sign*A_[nn-1][nn-1]*scale
def rank_mod(Mq, p):
    R = [[x.numerator*pow(x.denominator, -1, p) % p for x in row] for row in Mq]
    nn = len(R); rk = 0
    for col in range(nn):
        piv = next((i for i in range(rk, nn) if R[i][col]), None)
        if piv is None: continue
        R[rk], R[piv] = R[piv], R[rk]; inv = pow(R[rk][col], -1, p)
        for i in range(rk+1, nn):
            f = R[i][col]*inv % p
            if f: R[i] = [(R[i][k]-f*R[rk][k]) % p for k in range(nn)]
        rk += 1
    return rk
resB = {}
MS = {sg: [[Bm[i][k] + sg*Bm[i+1][k] for k in range(n0)] for i in range(n0)] for sg in (0, 1, -1)}
for sg in (0, 1, -1):
    rk = rank_mod(MS[sg], PCERT)
    gate("B", rk == n0, f"sigma={sg}: rank mod {PCERT} = {rk}")
    resB[str(sg)] = {"rank_mod_p": rk}
    log(f"B: sigma={sg:+d}: rank mod {PCERT} = {rk}/{n0}")
# Full rank modulo 131, a prime above every denominator prime of B, implies the three rational
# determinants are nonzero (exactly the paper's argument with 101 replaced by 131).

# ------------------------------------------------------------------ V per-prime valuations, N = 1
D1 = build(1, DES)
n = D1['n']
Mq = [[D1['Frat'][r][k] + GV*D1['FG'][r][k] for k in range(n)] for r in range(n)]
log(f"V: Delta_1 built (n = {n}); valuations for primes 2 sqrt(H) < p < H + 60")
rows = []
for p in primes_upto(H0 + 60):
    if p == 2 or p*p <= 4*H0 or GV.denominator % p == 0: continue
    v, lb = vp_det(Mq, p)
    if p > H0:
        gate("V", v >= 0, f"p={p} > H: v_p = {v}")
        rows.append((p, v, None)); continue
    bound = -Dfun(p, n0, b0, q0, g0, h0)
    gate("V", v >= bound, f"p={p}: v_p = {v} < bound {bound}")
    rows.append((p, v, bound))
claimed = [r for r in rows if r[2] is not None]
log("V: p : v_p (bound)  " + "  ".join(f"{p}:{v}({bd})" for p, v, bd in claimed))
log(f"V: {len(claimed)} primes in range; attained exactly at {sum(1 for p,v,bd in claimed if v == bd)}; "
    f"max slack {max(v-bd for p,v,bd in claimed)}; primes above H: all v_p >= 0: {all(v >= 0 for p,v,bd in rows if bd is None)}")

total = sum(FAILS.values())
out = {"script": "cat_newdesign_check.py", "design": DES, "gates": FAILS, "B": resB, "B_max_den_prime": maxp,
       "V": [{"p": p, "vp": v, "bound": bd} for p, v, bd in rows], "GATE_FAILS": total,
       "seconds": round(time.time()-T0, 1)}
with open(_os.path.join(HERE, "cat_newdesign_check_out.json"), "w") as fh: json.dump(out, fh, indent=1)
log(f"gates: {FAILS}")
print(f"GATE_FAILS = {total}")
LOGF.write(f"GATE_FAILS = {total}\n"); LOGF.close()
print(f"log saved to {LOGPATH}")
