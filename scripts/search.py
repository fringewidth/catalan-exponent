"""Parallel hill-climb over designs (n,b,q,g,h) minimising mu. Results appended to results.jsonl."""
import os, sys, json, random, time
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"): os.environ[v]="1"
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def admissible(d):
    n,b,q,g,h = d
    a=b+q; L=n+a; C=L+g; H=C+h
    return n>=16 and b>=1 and q>=1 and g>=1 and h>=0 and 2*g < n-1 and 2*b < H

def evalone(d):
    import catmu
    try:
        r = catmu.evaluate(d); r['design']=list(d); return r
    except Exception as e:
        return dict(design=list(d), mu=float('inf'), error=str(e))

def neighbours(d):
    n,b,q,g,h = d
    out=set()
    for i in range(1,5):
        for s in (-2,-1,1,2):
            e=list(d); e[i]+=s; out.add(tuple(e))
    for s in (-8,-4,4,8):   # scale n with ratios kept roughly
        m=n+s
        if m>=16:
            out.add((m, round(b*m/n), round(q*m/n), round(g*m/n), round(h*m/n)))
    return [e for e in out if admissible(e)]

if __name__=='__main__':
    budget_s = float(sys.argv[1]) if len(sys.argv)>1 else 2400
    workers = 11
    t0=time.time(); cache={}; log=open('results.jsonl','a')
    def record(rs):
        for r in rs:
            cache[tuple(r['design'])]=r; log.write(json.dumps(r)+"\n"); log.flush()
    rng=random.Random(20261015)
    starts=[(96,15,6,8,5),(48,8,3,4,3),(48,7,3,4,2),(72,11,4,6,4),(64,10,4,5,3),(40,6,3,3,2),(32,5,2,3,2),(56,9,3,5,3),(80,13,5,7,4),(120,19,8,10,6),(36,6,2,3,2),(24,4,2,2,1)]
    with Pool(workers) as pool:
        record(pool.map(evalone, starts))
        frontier=sorted(cache.values(), key=lambda r:r['mu'])
        while time.time()-t0 < budget_s:
            # pick a few current bests plus a random explored point, expand their unexplored neighbours
            picks=[tuple(r['design']) for r in frontier[:4]]
            picks.append(tuple(rng.choice(frontier[:30])['design']))
            todo=[]
            for p in picks:
                for e in neighbours(p):
                    if e not in cache and e not in todo: todo.append(e)
            if not todo:
                # random restart
                n=rng.choice([32,40,48,56,64,72,80,96]); 
                d=(n, rng.randint(max(1,n//12), n//4), rng.randint(1,n//8), rng.randint(1,n//6), rng.randint(0,n//8))
                todo=[d] if admissible(d) and d not in cache else []
                if not todo: continue
            rng.shuffle(todo)
            record(pool.map(evalone, todo[:workers*2]))
            frontier=sorted(cache.values(), key=lambda r:r['mu'])
            b=frontier[0]
            print(f"[{time.time()-t0:6.0f}s] evaluated {len(cache)} best mu={b['mu']:.1f} design={b['design']} gamma={b['gamma']:.6f} rho2={b['rho2']:.4f}", flush=True)
    print("DONE")
