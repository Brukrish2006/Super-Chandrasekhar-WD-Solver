from radial_pulsation import *
from scipy.optimize import brentq, minimize_scalar
import json,sys
def w2(M,rc): return omega0sq(M,rc,lo=-4,hi=4,n=41)[0]
def crit(M,a=1e9,b=3e11):
    ls=np.linspace(np.log10(a),np.log10(b),13); v=[w2(M,10**l) for l in ls]
    for i in range(len(ls)-1):
        if np.isfinite(v[i]) and np.isfinite(v[i+1]) and v[i]>0>v[i+1]:
            l=brentq(lambda l:w2(M,10**l),ls[i],ls[i+1],xtol=2e-3); return 10**l, equilibrium(M,10**l)[0]
    return (None, None) if v[-1]>0 else ('unstable_all',None)
def tp(M,a=1e9,b=3e11):
    ls=np.linspace(np.log10(a),np.log10(b),25); ms=[equilibrium(M,10**l)[0] for l in ls]; i=int(np.argmax(ms))
    if i in (0,len(ls)-1): return None,None
    r=minimize_scalar(lambda l:-equilibrium(M,10**l)[0],bounds=(ls[i-1],ls[i+1]),method='bounded',options={'xatol':1e-3})
    return 10**r.x,-r.fun
cases=[(0,0),(0,0.15),(1e13,0),(1e13,0.15),(1e14,0),(1e14,0.15),(3.79e14,0),(3.79e14,0.15),(3.79e14,0.45)]
res=[]
for B0,k in cases:
    d=dict(B0=B0,kappa=k)
    d['tp']=tp(Model(B0=B0,kappa=k,mag=B0>0))
    for md in (['eq','toroidal','frozen43'] if B0>0 else ['eq']):
        d[md]=crit(Model(B0=B0,kappa=k,mag=B0>0,bmode=md))
    print(json.dumps(d,default=str),flush=True); res.append(d)
json.dump(res,open('crit.json','w'),default=str)
