"""Referee 3.3: magnetic-to-gravitational energy ratio for the adopted field profile."""
from self_consistent import *
from topologies import deb_B0, make, closure_top, RHOC
import json
def ratio(M,rhoc):
    s=integrate(M,rhoc,0.0,dense=True); R=s.t[-1]; r=np.linspace(1e-4*R,R,6000); Y=s.sol(r)
    rho=np.maximum(Y[0],1.0); m=Y[1]/fM   # grams
    PB=np.array([M.thermo(x)[6] for x in rho]); eps=np.array([M.thermo(x)[1] for x in rho])
    Em=np.trapezoid(PB*4*np.pi*r*r,r)
    Wg=np.trapezoid(6.6743e-8*m*(eps/2.99792458e10**2)*4*np.pi*r,r)
    return Em/Wg
out=[]
for top in ['deb','frozen','core']:
    for Bc in [1e13,3e13,1e14,2e14,3.5e14]:
        if top=='core' and Bc>4e13: continue
        k,_,_=closure_top(top,Bc); r=ratio(make(top,Bc,k),RHOC)
        d=dict(top=top,Bc=Bc,kstar=k,Emag_over_Wg=r); out.append(d); print(json.dumps(d),flush=True)
json.dump(out,open('energy_ratio.json','w'))
