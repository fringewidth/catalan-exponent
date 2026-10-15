"""Parametrize Carreira's certification scripts for a design and run them in a fresh directory.
Usage: certify.py n b q g h  -> directory cert_<n>_<b>_<q>_<g>_<h>/ with logs and out.json files."""
import os, re, sys, shutil, subprocess, math
from fractions import Fraction as Fr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import catlib
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'new_design')
des = tuple(int(x) for x in sys.argv[1:6])
n,b,q,g,h = des
H0 = n+b+q+g+h
def isprime(m):
    if m<2: return False
    for p in range(2,int(m**0.5)+1):
        if m%p==0: return False
    return True
P = H0+1
while not isprime(P): P+=1
I,_,unres = catlib.integrate(n,b,q,g,h,exact=True)
delta = Fr(q+g+h, n)
flow = -float(I)/n**2 - (0.5+float(delta)/2+float(delta)**2/8)*math.log(2)
d = f"cert_{n}_{b}_{q}_{g}_{h}"
os.makedirs(d, exist_ok=True)
print(f"design {des} H0={H0} prime={P} f_low={flow:.7f} integral={I}")
subs = {
 'cat_tail_opt.py':   [("DES = (96, 15, 6, 8, 5)", f"DES = {des}"), ("FLOW_REF = -2.275548", f"FLOW_REF = {flow:.7f}"), ("T1_MIN = 0.0020", "T1_MIN = 0.0")],
 'cat_tail_edges.py': [("DES = (96, 15, 6, 8, 5)", f"DES = {des}"), ("FLOW_REF = -2.275548", f"FLOW_REF = {flow:.7f}"), ("T1_MIN = 0.0025", "T1_MIN = 0.0")],
 'cat_cert7_new.py':  [("DES = (96, 15, 6, 8, 5)", f"DES = {des}"), ("FLOW_REF = -2.275548", f"FLOW_REF = {flow:.7f}")],
 'cat_cert4_new.py':  [("NEW = ((96, 15, 6, 8, 5), 131)", f"NEW = ({des}, {P})")],
 'cat_newdesign_check.py': [("DES = (96, 15, 6, 8, 5)", f"DES = {des}"), ("PCERT = 131", f"PCERT = {P}")],
}
for f, pairs in subs.items():
    s = open(os.path.join(SRC, f)).read()
    for old, new in pairs:
        assert s.count(old) == 1, (f, old, s.count(old))
        s = s.replace(old, new)
    open(os.path.join(d, f), 'w').write(s)
print("scripts written to", d)
