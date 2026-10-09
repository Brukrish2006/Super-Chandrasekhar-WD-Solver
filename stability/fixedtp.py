from self_consistent import *
from scipy.optimize import minimize_scalar
import json
out=[]
for B0 in [1e13,3e13,1e14,1.5e14,2e14,3e14,3.79e14,5e14]:
    M=Model(B0=B0,kappa=0.15)
    ls=np.linspace(9.3,10.8,16); ms=[equilibrium(M,10**l)[0] for l in ls]; i=int(np.argmax(ms))
    r=minimize_scalar(lambda l:-equilibrium(M,10**l)[0],bounds=(ls[max(i-1,0)],ls[min(i+1,15)]),method='bounded',options={'xatol':1e-3})
    d=dict(B0=B0,rc_tp=10**r.x,M_tp=-r.fun,M_1e10=equilibrium(M,1e10)[0]);out.append(d);print(d,flush=True)
json.dump(out,open('fixedtp.json','w'))
