"""
cat_cert4_new.py -- Section 4.4 certificate (nonvanishing, Prop. 4.1) for a general design, with the
Gaussian pivots printed in the form of the paper's Table 1.

For a design (n0, b0, q0, g0, h0) (per unit N; a0 = b0 + q0, L0 = n0 + a0, C0 = L0 + g0, H0 = C0 + h0)
the fixed (n0+1) x n0 rational matrix B of eq. (47) is built by the recipe of Section 4.4:
  m_0 = 2, m_odd = 0, m_i = i m_{i-2}/(i+1);  k-, k+ boundary recurrences;  Y_ij = M0(i,j) by
  Y_ij = Y_{i-1,j-1} - m_{i-1}/j;  J_ij = (3/2) Z0(i,j);
  E_0 = t^(C0-1), E_1 = t^(C0-2), I_0 = 0, I_1 = t^(C0-1), S_m = 2 S_{m-1}/t - S_{m-2};
  rows r = 0..n0 (r = n0 is the auxiliary row): u = |r - g0|,
  y_r = (1-t)^h0 E_u = P_r, z_r = sign(r - g0)(1-t)^h0 I_u = D_r;
  v_r(j) = sum_i [t^i]y_r Y_ij - [t^i]z_r J_ij, j = b0 .. L0-1; q0 successive differences
  (v(j) - v(j+1)) give row r of B (length n0).
Then B0 = rows 0..n0-1, B1 = rows 1..n0, and for sigma in {0, 1, -1} the paper's elimination rule
(rows R_i = B(i,.) + sigma B(i+1,.); at step i swap with the first later row with a nonzero entry
in column i if needed; record the pivot d_i; clear column i below) is run modulo a prime P > H0.
All pivots nonzero <=> the three matrices are invertible mod P <=> (denominators being units mod P)
their rational determinants are nonzero, which is eq. (48) for the design.

DESIGNS
  paper (48, 7, 4, 4, 2) modulo 101  -- positive control: must reproduce the paper's Table 1
                                        (144 pivots) and swaps (30,31), (45,46) for sigma = +1
  new   (96, 15, 6, 8, 5) modulo 131 -- the certificate for the candidate design

GATES (pre-registered)
  P1 paper design: rows y_r, z_r equal definition (3) (Chebyshev T_d, U_{d-1}), rows 0..n0  [0 fails]
  P2 paper design: zeta(2) coefficient of every raw entry is 0 (rows 0..n0, j = b0..L0-1)    [0 fails]
  P3 paper design: pivots and swaps mod 101 equal Table 1 exactly                            [0 fails]
  N1 new design: rows equal definition (3), rows 0..n0; supports of rows 0..n0-1 in [A0, H0-1] [0 fails]
  N2 new design: zeta(2) coefficient of every raw entry is 0                                [0 fails]
  N3 new design: every denominator prime of B is <= H0 = 130 < 131                          [0 fails]
  N4 new design: all 3 x 96 pivots mod 131 nonzero                                          [0 fails]
REPORT: pivots and swaps for the new design (also written to cat_cert4_new_table.tex as a LaTeX
table in the layout of the paper's Table 1, 16 pivots per line).

RUN: python3 cat_cert4_new.py   (no arguments; single process; standard library only)
TIME: about 15 s.
OUTPUT: screen + cat_cert4_new_log_<YYYYMMDD_HHMMSS>.txt + cat_cert4_new_out.json +
        cat_cert4_new_table.tex, next to the script.
"""
import sys, json, time, os
from fractions import Fraction as Fr
from math import comb
try:
    sys.set_int_max_str_digits(0)
except AttributeError:
    pass

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
STAMP = time.strftime('%Y%m%d_%H%M%S')
LOGPATH = os.path.join(HERE, 'cat_cert4_new_log_' + STAMP + '.txt')
LOGF = open(LOGPATH, 'w')
def log(s):
    line = f"[{time.time()-T0:6.1f}s] {s}"
    print(line, flush=True); LOGF.write(line + '\n'); LOGF.flush()
FAILS = {}
def gate(name, ok, info=""):
    FAILS[name] = FAILS.get(name, 0) + (0 if ok else 1)
    if not ok: log(f"FAIL {name} {info}")

PAPER = ((48, 7, 4, 4, 2), 101)
NEW = ((36, 6, 2, 3, 2), 53)
TABLE1 = {
 0: [60,68,62,79,47,32,69,57,30,72,35,66,43,20,85,48,
     88,4,77,54,60,79,26,68,83,39,40,65,1,68,78,24,
     15,98,32,22,94,9,99,10,15,75,4,2,25,53,90,79],
 1: [38,51,90,70,5,21,88,55,45,20,35,41,77,10,18,25,
     76,14,38,72,6,66,56,35,83,58,56,11,17,20,30,24,
     28,11,25,70,79,99,66,38,4,41,63,91,63,17,90,98],
 -1:[82,92,26,87,21,74,87,88,88,3,14,23,38,58,36,20,
     26,33,94,74,78,45,93,86,73,66,45,30,61,3,88,27,
     20,58,69,48,78,39,48,1,66,18,86,93,52,92,39,77]}
SWAPS1 = {0: [], 1: [(30, 31), (45, 46)], -1: []}

def c(l): return Fr(comb(2*l, l), 4**l)
def primefactors(nn):
    nn = abs(nn); ps = set(); p = 2
    while p*p <= nn:
        while nn % p == 0: ps.add(p); nn //= p
        p += 1
    if nn > 1: ps.add(nn)
    return ps

def build_B(des, tag):
    n0, b0, q0, g0, h0 = des
    a0 = b0 + q0; L0 = n0 + a0; C0 = L0 + g0; H0 = C0 + h0; A0 = a0 + 2*g0
    IMAX, JMAX = H0 - 1, L0 - 1
    m = [Fr(0)]*(IMAX + 1); m[0] = Fr(2)
    for i in range(2, IMAX + 1, 2): m[i] = i*m[i-2]/(i + 1)
    km = [Fr(0)]*(IMAX + 1); km[1] = Fr(2)
    for d in range(2, IMAX + 1): km[d] = ((d-1)*km[d-2] + m[d-2] + m[d-1])/d
    kp = [Fr(0)]*(JMAX + 1)
    for d in range(2, JMAX + 1): kp[d] = ((d-2)*kp[d-2] + Fr(2, d-1))/(d-1)
    Y = [[Fr(0)]*(JMAX + 1) for _ in range(IMAX + 1)]
    for i in range(IMAX + 1): Y[i][0] = km[i]
    for j in range(JMAX + 1): Y[0][j] = kp[j]
    for i in range(1, IMAX + 1):
        for j in range(1, JMAX + 1): Y[i][j] = Y[i-1][j-1] - m[i-1]/j
    B1h = [Fr(0)]*(IMAX + 1); B2h = [Fr(0)]*(IMAX + 1)
    for i in range(1, IMAX + 1): B1h[i] = B1h[i-1] + Fr(1, i); B2h[i] = B2h[i-1] + Fr(1, i*i)
    def J(i, j): return Fr(3, 2)*(-B2h[i] if i == j else (B1h[i] - B1h[j])/(i - j))
    # Chebyshev recipe of Section 4.4 (polynomials as coefficient lists in t)
    def mono(k): return [0]*k + [1]
    def divt(a):
        assert a[0] == 0; return a[1:]
    def sub(a, b):
        L_ = max(len(a), len(b))
        return [(a[k] if k < len(a) else 0) - (b[k] if k < len(b) else 0) for k in range(L_)]
    umax = n0 - g0
    E = [mono(C0-1), mono(C0-2)]; I = [[0], mono(C0-1)]
    for mi in range(2, umax + 1):
        for S in (E, I): S.append(sub([2*x for x in divt(S[mi-1])], S[mi-2]))
    binh = [(-1)**v*comb(h0, v) for v in range(h0 + 1)]
    def times_binh(a, sg):
        out = [0]*(len(a) + h0)
        for i, x in enumerate(a):
            if x:
                for v, bv in enumerate(binh): out[i+v] += sg*x*bv
        return out
    yr, zr = [], []
    for r in range(n0 + 1):
        u = abs(r - g0); sg = (r > g0) - (r < g0)
        yr.append(times_binh(E[u], 1)); zr.append(times_binh(I[u], sg))
    # independent check against definition (3): (1-t)^h0 t^(C0-1) T_d(1/t), sign (1-t)^h0 t^(C0-1) U_{d-1}(1/t)
    T = [[1], [0, 1]]; U = [[1], [0, 2]]
    for k in range(2, umax + 1):
        T.append(sub([0] + [2*x for x in T[-1]], T[-2])); U.append(sub([0] + [2*x for x in U[-1]], U[-2]))
    G_rows = "P1" if tag == "paper" else "N1"
    for r in range(n0 + 1):
        d = abs(r - g0); sg = (r > g0) - (r < g0)
        P = [0]*C0; D = [0]*C0
        for k, x in enumerate(T[d]): P[C0-1-k] += x
        if d >= 1:
            for k, x in enumerate(U[d-1]): D[C0-1-k] += sg*x
        strip = lambda a: a[:max([i for i, x in enumerate(a) if x] + [-1]) + 1]
        gate(G_rows, strip(times_binh(P, 1)) == strip(yr[r]), f"{tag} P row {r}")
        gate(G_rows, strip(times_binh(D, 1)) == strip(zr[r]), f"{tag} D row {r}")
        if r < n0:
            degs = [i for i, x in enumerate(yr[r]) if x] + [i for i, x in enumerate(zr[r]) if x]
            gate(G_rows, min(degs) >= A0 and max(degs) <= H0 - 1, f"{tag} row {r} support")
    # zeta(2) contact
    G_z = "P2" if tag == "paper" else "N2"; bad = 0
    for r in range(n0 + 1):
        for j in range(b0, L0):
            s = Fr(0)
            for i in range(0, min(j, len(yr[r]))):
                if (j - i - 1) % 2 == 0 and yr[r][i]: s += yr[r][i]*c((j - i - 1)//2)
            s -= zr[r][j] if j < len(zr[r]) else 0
            bad += (s != 0)
    gate(G_z, bad == 0, f"{tag}: {bad} raw entries with nonzero zeta(2) coefficient")
    # B
    B = []
    for r in range(n0 + 1):
        v = []
        for j in range(b0, L0):
            s = Fr(0)
            for i, x in enumerate(yr[r]):
                if x: s += x*Y[i][j]
            for i, x in enumerate(zr[r]):
                if x: s -= x*J(i, j)
            v.append(s)
        for _ in range(q0): v = [v[k] - v[k+1] for k in range(len(v) - 1)]
        assert len(v) == n0
        B.append(v)
    maxp = max((max(primefactors(x.denominator)) if x.denominator > 1 else 1) for row in B for x in row)
    log(f"{tag} design {des}: B ({n0+1} x {n0}) built; rows checked against (3); zeta(2) failures {bad}; "
        f"largest denominator prime {maxp} (H0 = {H0})")
    return B, maxp, H0

def pivots(B, n0, P):
    def modp(x): return x.numerator*pow(x.denominator, -1, P) % P
    res = {}
    for sg in (0, 1, -1):
        R = [[modp(B[i][k] + sg*B[i+1][k]) for k in range(n0)] for i in range(n0)]
        piv, swaps = [], []
        for i in range(n0):
            if R[i][i] == 0:
                jj = next((j for j in range(i + 1, n0) if R[j][i] != 0), None)
                if jj is None: piv.append(0); break
                R[i], R[jj] = R[jj], R[i]; swaps.append((i, jj))
            di = R[i][i]; piv.append(di); inv = pow(di, -1, P)
            for j in range(i + 1, n0):
                f = R[j][i]*inv % P
                if f: R[j] = [(R[j][k] - f*R[i][k]) % P for k in range(n0)]
        res[sg] = (piv, swaps)
    return res

# ---------------------------------------------------------------- positive control: paper
(des, P) = PAPER
B, maxp, H0 = build_B(des, "paper")
res = pivots(B, des[0], P)
for sg in (0, 1, -1):
    piv, sw = res[sg]
    gate("P3", piv == TABLE1[sg] and sw == SWAPS1[sg], f"paper sigma={sg}: pivots {piv} swaps {sw}")
    log(f"P3 paper sigma={sg:+d}: {sum(a == b for a, b in zip(piv, TABLE1[sg]))}/48 pivots equal Table 1; swaps {sw}")

# ---------------------------------------------------------------- new design
(des, P) = NEW
B, maxp, H0 = build_B(des, "new")
gate("N3", maxp <= H0 < P, f"largest denominator prime {maxp}, H0 {H0}, modulus {P}")
res = pivots(B, des[0], P)
out_piv = {}
for sg in (0, 1, -1):
    piv, sw = res[sg]
    gate("N4", len(piv) == des[0] and all(piv), f"new sigma={sg}: zero pivot at step {len(piv)-1}")
    out_piv[str(sg)] = {"pivots": piv, "swaps": sw}
    log(f"N4 new sigma={sg:+d}: {sum(1 for x in piv if x)}/{des[0]} nonzero pivots mod {P}; swaps {sw}")
    for k in range(0, len(piv), 16): log(f"    {', '.join(str(x) for x in piv[k:k+16])}")

# ---------------------------------------------------------------- LaTeX table
lines = [r"\begin{table}[ht]", r"\centering", r"\small",
         r"\begin{tabular}{rl}", r"\hline", r"$\sigma$ & Successive pivots in $\mathbb{F}_{131}$ \\", r"\hline"]
for sg in (0, 1, -1):
    piv = out_piv[str(sg)]["pivots"]
    chunks = [", ".join(str(x) for x in piv[k:k+16]) for k in range(0, len(piv), 16)]
    lab = {0: "$0$", 1: "$1$", -1: "$-1$"}[sg]
    for i, ch in enumerate(chunks):
        lines.append(f"{lab if i == 0 else ''} & {ch} \\\\")
    lines.append(r"\hline")
sw_text = "; ".join(f"$\\sigma={sg}$: " + (", ".join(f"$({a},{b})$" for a, b in out_piv[str(sg)]["swaps"]) or "none")
                    for sg in (0, 1, -1))
lines += [r"\end{tabular}",
          r"\caption{Gaussian pivots of $B_0+\sigma B_1$ modulo $131$ for the design $(96,15,6,8,5)$, "
          r"computed by the elimination rule of Section~4.4 of the preprint (rows in increasing order, "
          r"indices from zero). Row swaps: " + sw_text + r".}",
          r"\label{tab:pivots131}", r"\end{table}"]
with open(os.path.join(HERE, "cat_cert4_new_table.tex"), "w") as fh: fh.write("\n".join(lines) + "\n")
log("LaTeX table written to cat_cert4_new_table.tex")

total = sum(FAILS.values())
out = {"script": "cat_cert4_new.py", "design": NEW[0], "modulus": NEW[1], "largest_denominator_prime": maxp,
       "pivots": out_piv, "gates": FAILS, "GATE_FAILS": total, "seconds": round(time.time() - T0, 1)}
with open(os.path.join(HERE, "cat_cert4_new_out.json"), "w") as fh: json.dump(out, fh, indent=1)
log(f"gates: {FAILS}")
print(f"GATE_FAILS = {total}")
LOGF.write(f"GATE_FAILS = {total}\n"); LOGF.close()
print(f"log saved to {LOGPATH}")
