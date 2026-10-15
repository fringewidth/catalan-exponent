"""mu(design): the irrationality-exponent bound of Theorem 35 of Carreira's note, with the numerical gap
and sup-terms of the quick evaluator (cat_design_search.py) in place of the certified ones."""
import math, sys, json
sys.path.insert(0, __file__.rsplit('/',1)[0])
import catlib
LOG2 = math.log(2); LOGS = math.log(1+math.sqrt(2))

def evaluate(d):
    n, b, q, g, h = d
    fl, _ = catlib.f_low(n, b, q, g, h)
    u2, lam, sig = catlib.solve(d, 2)
    gamma = fl - float(u2)
    al = (b+q)/n; gg = g/n; gq = q/n; et = h/n; nu = 1 - gg
    rho1 = (2 - al - gg)*LOG2 - float(sig) + 2*LOG2
    rho2 = rho1 + (et + gq)*LOG2 + nu*LOGS
    mu = rho2*(nu/gamma + n) if gamma > 0 else float('inf')
    return dict(design=d, f_low=fl, U2=float(u2), gamma=gamma, Sigma2=float(sig), rho1=rho1, rho2=rho2, mu=mu)

if __name__ == '__main__':
    for d in [(96,15,6,8,5), (48,8,3,4,3), (48,7,4,4,2)]:
        r = evaluate(d)
        print({k:(round(v,7) if isinstance(v,float) else v) for k,v in r.items()})
