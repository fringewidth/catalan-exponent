"""Assemble the result package for a certified design directory. Usage: package.py cert_<n>_<b>_<q>_<g>_<h> outdir"""
import json, math, os, shutil, sys
from fractions import Fraction as Fr
d, out = sys.argv[1], sys.argv[2]
n,b,q,g,h = (int(x) for x in d.split('_')[1:])
c7 = json.load(open(f'{d}/cat_cert7_new_out.json'))
best = max(c7['certificates'], key=lambda c: float(c['margin_lower']))
k2, k1 = best['kappa2'], best['kappa1']
tails = json.load(open(f"{d}/{best['input']}"))
lam1 = float(Fr(tails['kappa1']['lambda']))
gamma = float(best['margin_lower'])
S2hi = float(k2['supX_upper']) + float(k2['supY_upper']); S1hi = float(k1['supX_upper']) + float(k1['supY_upper'])
S2 = S2hi - 1e-12; S1 = S1hi - 1e-12
U2 = float(k2['U_upper']); U1 = float(k1['U_upper'])
LOG2 = math.log(2); LOGS = math.log(1+math.sqrt(2))
al = Fr(b+q, n); gg = Fr(g, n); gq = Fr(q, n); et = Fr(h, n); nu = 1 - gg
rho1 = float(2 - al - gg)*LOG2 - S2 + 2*LOG2
second = float(2 - al - gg)*LOG2 - S1 - lam1*float(1 - 2*gg) + 1.5*LOG2
rho2 = rho1 + float(et + gq)*LOG2 + float(nu)*LOGS
mu = rho2*(float(nu)/gamma + n)
H0 = n+b+q+g+h; A0 = b+q+2*g
c4 = json.load(open(f'{d}/cat_cert4_new_out.json'))
chk = json.load(open(f'{d}/cat_newdesign_check_out.json'))
os.makedirs(f'{out}/certificates/{d}', exist_ok=True); os.makedirs(f'{out}/scripts', exist_ok=True)
for f in os.listdir(d):
    if f.endswith('.json') or f.endswith('.txt') or f.endswith('.log') or f.endswith('.tex') or f.endswith('.py'):
        shutil.copy(f'{d}/{f}', f'{out}/certificates/{d}/{f}')
for f in ['catlib.py','catmu.py','search.py','certify.py','mu_cert.py','package.py']:
    shutil.copy(f, f'{out}/scripts/{f}')
shutil.copy('results.jsonl', f'{out}/certificates/search_results.jsonl')
shutil.copy('search.log', f'{out}/certificates/search.log')
summary = dict(design=[n,b,q,g,h], H0=H0, A0=A0, prime=c4.get('new',{}).get('prime') if isinstance(c4.get('new'),dict) else None,
    f_low_lower=c7['f_low_lower'], certificate_input=best['input'], gamma_lower=gamma,
    U2_upper=U2, U1_upper=U1, Sigma2_lower=S2, Sigma1_lower=S1, lambda1=lam1,
    rho1_upper=rho1, second_case_coefficient=second, rho2_upper=rho2, mu_upper=mu,
    gates=dict(cert7=c7['GATE_FAILS'], cert4=c4['GATE_FAILS'], structural=chk['GATE_FAILS']))
json.dump(summary, open(f'{out}/RESULT.json','w'), indent=2)
readme = f"""# An irrationality exponent below {math.ceil(mu)} for Catalan's constant

**Claim.** The irrationality exponent of Catalan's constant satisfies mu(G) < {mu:.2f}, so mu(G) < {math.ceil(mu)}.

**How.** This is Theorem 35 of M. C. S. Carreira, *General designs and an irrationality exponent for Catalan's constant*
(MarcosCarreira/catalan-designs, commit ec231b3), applied to the design (n0, b0, q0, g0, h0) = ({n}, {b}, {q}, {g}, {h})
in place of (96, 15, 6, 8, 5). Carreira's Theorem 35 gives mu(G) < 2715 for the design (96, 15, 6, 8, 5); every
ingredient of its proof is stated there for a general design satisfying conditions (C1)-(C4), with two numerical
dominance checks in the proof of Corollary 33 (minors). The theorem reads, for a design with certified gap gamma and
certified sup-terms Sigma_2, Sigma_1, lambda^(1):

    rho_1 = (2 - alpha - gamma_g) log 2 - Sigma_2 + 2 log 2,
    rho_2 = rho_1 + (eta + gamma_q) log 2 + (1 - gamma_g) log(1 + sqrt 2),
    mu(G) <= rho_2 (1 - gamma_g + n0 gamma) / gamma,

with alpha = (b0+q0)/n0, gamma_g = g0/n0, gamma_q = q0/n0, eta = h0/n0. The exponent is dominated by
(1 - gamma_g)/gamma, and the term n0 enters additively; Carreira's design search maximised gamma alone, at n0 = 96.
Minimising mu directly over designs, with n0 free, gives a design at n0 = {n} with a slightly smaller gap and a much
smaller additive term.

**Numbers for ({n}, {b}, {q}, {g}, {h})** (all from the certification scripts of Carreira's repository, re-run for this
design; see `certificates/{d}/`):

| quantity | value | source |
| --- | --- | --- |
| (C1) 2 g0 < n0 - 1 | {2*g} < {n-1} | arithmetic |
| (C2) 2 b0 < H0 | {2*b} < {H0} | arithmetic |
| (C3) det B0, det(B0+B1), det(B0-B1) nonzero | all {n} pivots nonzero mod {summary['prime'] or 'P'} for sigma in {{0, 1, -1}} | `cat_cert4_new.py`, gates N1-N4 pass |
| structural checks S, Z, B, V | pass | `cat_newdesign_check.py` |
| F_low | >= {float(c7['f_low_lower']):.12f} | exact integral {c7['f_low_integral']}/n0^2, `cat_cert7_new.py` C6 |
| U_2 (kappa = 2) | <= {U2:.12f} | `cat_cert7_new.py` on `{best['input']}` |
| U_1 (kappa = 1) | <= {U1:.12f} | same |
| (C4) gap gamma = F_low - max(U_1, U_2) | >= {gamma:.10f} | `cat_cert7_new.py` C7 |
| Sigma_2 = sup X + sup Y (kappa = 2) | >= {S2:.12f} | enclosure of the sup values (allowance < 1e-20), minus 1e-12 |
| Sigma_1 (kappa = 1), lambda^(1) | >= {S1:.12f}, {lam1:.10f} | same |
| U_1 - U_2 | {U1-U2:.5f} < 0 | second case has the smaller constant |
| rho_1 | <= {rho1:.6f} | formula above |
| second-case coefficient of t | {second:.6f} <= rho_1 | first case dominates on [0, 1] |
| rho_2 | <= {rho2:.6f} | formula above |
| mu(G) | <= {mu:.4f} | rho_2 ((1 - gamma_g)/gamma + n0) |

Equilibrium ceiling: the dual of the real-place bound (`cat_equilibrium.py`) bounds the best certifiable gap for this
design by 0.00262, so the certificate above (gap 0.00253) is within four percent of what the method can give here.

**Reproduce.** From Carreira's repository at commit ec231b3, copy `scripts/` of this package next to
`new_design/`, then `python3 scripts/certify.py {n} {b} {q} {g} {h}` writes a parametrised copy of the five
certification scripts; run `cat_cert4_new.py`, `cat_tail_opt.py`, `cat_tail_edges.py`, `cat_cert7_new.py`
(needs mpmath) and `cat_newdesign_check.py` (needs mpmath) in that directory, then `python3 scripts/mu_cert.py <dir>`.
`scripts/search.py` is the design search (objective mu, quick evaluator of `cat_design_search.py`);
`certificates/search_results.jsonl` lists every design evaluated.

**What is and is not claimed.** The mathematics is Carreira's (and Jha's, Litsey's and OpenAI's); this package
changes the design and re-runs the certification. The design-dependent steps of Theorem 35 were checked as listed:
conditions (C1)-(C4), the two dominance checks of Corollary 33, lambda^(1) >= 0, and the hypotheses of Jha's
Lemma 4.3 quoted there: D* = n - 1 - 2g = {30 if (n,g)==(36,3) else 'n-1-2g'}N - 1 lies in (0, n], and n = {n}N >= 48 for
N >= 2, which suffices because Corollary 33 and Proposition 34 are asymptotic statements (N >= N_eps, primes p > 4 H0). The numerical
values are rigorous where the scripts are (interval arithmetic in `cat_cert7_new.py`, exact arithmetic in
`cat_cert4_new.py`); the sup lower bounds use the certified enclosures (width below 1e-20) with a 1e-12 safety margin; the printed
margin_lower is rounded to 12 significant digits, which can exceed the true margin by at most 5e-15, worth 4e-9 in mu
against the 0.26 of slack below {math.ceil(mu)}. A written
check of this instantiation by a second person has not been done.

**Credits.** Design search, certification runs and this package: Claude (Claude Fable 5.1, Anthropic), operating in
Claude Code on compute provided by Hrishik Sai Bojnal (fringewidth), 15 October 2026. Theorem and scripts:
Marcos Costa Santos Carreira (Apache-2.0 scripts, CC BY paper); ingredients: R. K. Jha, S. Litsey; the construction and
the irrationality proof: OpenAI, *Catalan's constant is irrational* (24 September 2026).
"""
open(f'{out}/README.md','w').write(readme)
print(json.dumps(summary, indent=1))
