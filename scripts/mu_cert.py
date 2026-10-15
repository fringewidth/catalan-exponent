"""Certified-level mu for a design from the cat_cert7_new output (Theorem 35's formula and its second-case check)."""
import json, math, sys
d = sys.argv[1]; n,b,q,g,h = (int(x) for x in d.split('_')[1:])
out = json.load(open(f'{d}/cat_cert7_new_out.json'))
LOG2 = math.log(2); LOGS = math.log(1+math.sqrt(2))
al=(b+q)/n; gg=g/n; gq=q/n; et=h/n; nu=1-gg
best=None
for c in out['certificates']:
    k2, k1 = c['kappa2'], c['kappa1']
    gamma = float(c['margin_lower'])
    S2 = float(k2['supX_upper'])+float(k2['supY_upper']) - 1e-12   # lower bound on Sigma_2 (allowances ~1e-23)
    S1 = float(k1['supX_upper'])+float(k1['supY_upper']) - 1e-12
    U2 = float(k2['U_upper']); U1 = float(k1['U_upper'])
    # lambda^(1) from the tail file used
    tails = json.load(open(f"{d}/{c['input']}"))
    lam = None
    for key in ('lambda','lambda1','lam'):
        if key in tails: lam = float(eval(tails[key]) if isinstance(tails[key],str) else tails[key])
    if lam is None and 'kappa1' in tails:
        for key in ('lambda','lam'):
            if key in tails['kappa1']: v=tails['kappa1'][key]; lam=float(eval(v) if isinstance(v,str) else v)
    rho1 = (2-al-gg)*LOG2 - S2 + 2*LOG2
    second = (2-al-gg)*LOG2 - S1 - (lam if lam is not None else float('nan'))*(1-2*gg) + 1.5*LOG2
    rho2 = rho1 + (et+gq)*LOG2 + nu*LOGS
    mu = rho2*(nu/gamma + n)
    print(f"{c['input']}: gamma={gamma:.10f} Sigma2>={S2:.12f} Sigma1>={S1:.12f} lambda1={lam} U1-U2={U1-U2:.5f}")
    print(f"   rho1={rho1:.6f} second-case coeff={second:.6f} (must be <= rho1)  rho2={rho2:.6f}  mu={mu:.2f}")
    if best is None or mu < best: best = mu
print("BEST certified-level mu for", (n,b,q,g,h), "=", round(best,2))
