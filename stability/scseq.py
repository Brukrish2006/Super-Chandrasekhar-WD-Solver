from self_consistent import *
import json,sys,time
from scipy.optimize import brentq
B0=float(sys.argv[1])
rhos=np.logspace(np.log10(2e9),np.log10(6e10),19)
rows=[]
for rc in rhos:
    k,m,R,_=closure(B0,rc)
    d=dict(rhoc=rc,kstar=k,M=m,R=R)
    for md in ['eq','toroidal','frozen43']:
        d[md]=omega0sq(Model(B0=B0,kappa=k,bmode=md),rc,lo=-8,hi=8,n=81)[0]
    rows.append(d); print(json.dumps(d),flush=True)
json.dump(rows,open(f'scseq_{B0:.3e}.json','w'))
