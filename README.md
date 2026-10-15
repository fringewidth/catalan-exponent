# An irrationality exponent below 2402 for Catalan's constant

**Claim.** The irrationality exponent of Catalan's constant satisfies mu(G) < 2401.74, so mu(G) < 2402.

**How.** This is Theorem 35 of M. C. S. Carreira, *General designs and an irrationality exponent for Catalan's constant*
(MarcosCarreira/catalan-designs, commit ec231b3), applied to the design (n0, b0, q0, g0, h0) = (36, 6, 2, 3, 2)
in place of (96, 15, 6, 8, 5). Carreira's Theorem 35 gives mu(G) < 2715 for the design (96, 15, 6, 8, 5); every
ingredient of its proof is stated there for a general design satisfying conditions (C1)-(C4), with two numerical
dominance checks in the proof of Corollary 33 (minors). The theorem reads, for a design with certified gap gamma and
certified sup-terms Sigma_2, Sigma_1, lambda^(1):

    rho_1 = (2 - alpha - gamma_g) log 2 - Sigma_2 + 2 log 2,
    rho_2 = rho_1 + (eta + gamma_q) log 2 + (1 - gamma_g) log(1 + sqrt 2),
    mu(G) <= rho_2 (1 - gamma_g + n0 gamma) / gamma,

with alpha = (b0+q0)/n0, gamma_g = g0/n0, gamma_q = q0/n0, eta = h0/n0. The exponent is dominated by
(1 - gamma_g)/gamma, and the term n0 enters additively; Carreira's design search maximised gamma alone, at n0 = 96.
Minimising mu directly over designs, with n0 free, gives a design at n0 = 36 with a slightly smaller gap and a much
smaller additive term.

**Numbers for (36, 6, 2, 3, 2)** (all from the certification scripts of Carreira's repository, re-run for this
design; see `certificates/cert_36_6_2_3_2/`):

| quantity | value | source |
| --- | --- | --- |
| (C1) 2 g0 < n0 - 1 | 6 < 35 | arithmetic |
| (C2) 2 b0 < H0 | 12 < 49 | arithmetic |
| (C3) det B0, det(B0+B1), det(B0-B1) nonzero | all 36 pivots nonzero mod P for sigma in {0, 1, -1} | `cat_cert4_new.py`, gates N1-N4 pass |
| structural checks S, Z, B, V | pass | `cat_newdesign_check.py` |
| F_low | >= -2.278735682280 | exact integral 4825/2/n0^2, `cat_cert7_new.py` C6 |
| U_2 (kappa = 2) | <= -2.281264736129 | `cat_cert7_new.py` on `cat_tail_edges_out.json` |
| U_1 (kappa = 1) | <= -2.419237646550 | same |
| (C4) gap gamma = F_low - max(U_1, U_2) | >= 0.0025290538 | `cat_cert7_new.py` C7 |
| Sigma_2 = sup X + sup Y (kappa = 2) | >= -2.581907572299 | enclosure of the sup values (allowance < 1e-20), minus 1e-12 |
| Sigma_1 (kappa = 1), lambda^(1) | >= -2.992664633367, 2.7242538218 | same |
| U_1 - U_2 | -0.13797 < 0 | second case has the smaller constant |
| rho_1 | <= 5.142701 | formula above |
| second-case coefficient of t | 2.936673 <= rho_1 | first case dominates on [0, 1] |
| rho_2 | <= 6.027643 | formula above |
| mu(G) | <= 2401.7410 | rho_2 ((1 - gamma_g)/gamma + n0) |

Equilibrium ceiling: the dual of the real-place bound (`cat_equilibrium.py`) bounds the best certifiable gap for this
design by 0.00262, so the certificate above (gap 0.00253) is within four percent of what the method can give here.

**Reproduce.** From Carreira's repository at commit ec231b3, copy `scripts/` of this package next to
`new_design/`, then `python3 scripts/certify.py 36 6 2 3 2` writes a parametrised copy of the five
certification scripts; run `cat_cert4_new.py`, `cat_tail_opt.py`, `cat_tail_edges.py`, `cat_cert7_new.py`
(needs mpmath) and `cat_newdesign_check.py` (needs mpmath) in that directory, then `python3 scripts/mu_cert.py <dir>`.
`scripts/search.py` is the design search (objective mu, quick evaluator of `cat_design_search.py`);
`certificates/search_results.jsonl` lists every design evaluated (1,352 designs, 45 minutes on 11 cores; the
runner-up by the quick evaluator is (52, 8, 3, 4, 3), five percent worse).

**What is and is not claimed.** The mathematics is Carreira's (and Jha's, Litsey's and OpenAI's); this package
changes the design and re-runs the certification. The design-dependent steps of Theorem 35 were checked as listed:
conditions (C1)-(C4), the two dominance checks of Corollary 33, lambda^(1) >= 0, and the hypotheses of Jha's
Lemma 4.3 quoted there: D* = n - 1 - 2g = 30N - 1 lies in (0, n], and n = 36N >= 48 for
N >= 2, which suffices because Corollary 33 and Proposition 34 are asymptotic statements (N >= N_eps, primes p > 4 H0). The numerical
values are rigorous where the scripts are (interval arithmetic in `cat_cert7_new.py`, exact arithmetic in
`cat_cert4_new.py`); the sup lower bounds use the certified enclosures (width below 1e-20) with a 1e-12 safety margin; the printed
margin_lower is rounded to 12 significant digits, which can exceed the true margin by at most 5e-15, worth 4e-9 in mu
against the 0.26 of slack below 2402. A written
check of this instantiation by a second person has not been done.

**Credits.** Design search, certification runs and this package: Claude (Claude Fable 5.1, Anthropic), operating in
Claude Code on compute provided by Hrishik Sai Bojnal (fringewidth), 15 October 2026. Theorem and scripts:
Marcos Costa Santos Carreira (Apache-2.0 scripts, CC BY paper); ingredients: R. K. Jha, S. Litsey; the construction and
the irrationality proof: OpenAI, *Catalan's constant is irrational* (24 September 2026).
