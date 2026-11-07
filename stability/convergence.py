"""Referee 3.10: separate numerical integration error from physical/model uncertainty."""
import radial_pulsation as rp
from self_consistent import *
import json
def eq_rtol(M,rhoc,rtol):
    P,e,*_=M.thermo(rhoc); GPP,_,X0,_=M.gammas(rhoc); r0=10.0
    y0=[rhoc,4/3*np.pi*r0**3*e*fP,0.0,1.0,(-3*GPP+X0)*fP]
    s=solve_ivp(rp.rhs,(r0,3e9),y0,args=(M,0.0),events=rp.surf,method='LSODA',rtol=rtol,atol=[1e-6*rhoc*rtol/1e-9,1e-30,1e-14,1e-12,1e-40])
    return s.y[1,-1]/fM/Msun
out={}
for lab,kw in [('GR',dict(mag=False)),('B13',dict(B0=1e13)),('B379_k015',dict(B0=3.79e14,kappa=0.15))]:
    M=Model(**kw)
    vals={rt:eq_rtol(M,1e10,rt) for rt in [1e-7,1e-9,1e-11]}
    out[lab]={str(k):v for k,v in vals.items()}; print(lab,vals,flush=True)
# eigenvalue: rtol 1e-9 vs 1e-11 by temporarily patching solve_ivp tolerance
import functools
orig=rp.solve_ivp
for rt in [1e-9,1e-11]:
    rp.solve_ivp=functools.partial(orig,rtol=rt) if False else (lambda *a,rt=rt,**k: orig(*a,**{**k,'rtol':rt}))
    w=rp.omega0sq(Model(B0=3.79e14,kappa=0.15,bmode='toroidal'),9e9,lo=-6,hi=6,n=61)[0]
    w2=rp.omega0sq(Model(mag=False),2e10,lo=-2,hi=2,n=21)[0]
    out[f'omega2_rtol{rt}']=(w,w2); print('rtol',rt,'omega2 toroidal ext 9e9:',w,' GR 2e10:',w2,flush=True)
rp.solve_ivp=orig
json.dump(out,open('convergence.json','w'))
