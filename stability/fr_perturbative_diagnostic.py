import os, sys, numpy as np, json
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
from constants import G,c
from eos import EOS
from tov_solver import TOVSolver
eos=EOS(mode='chandra',magnetic_tov=False,N_points=2000)
sb=TOVSolver(eos,alpha=0.0,kappa=0.0,compute_tidal=False)
def diag(a,rc):
    s=TOVSolver(eos,alpha=a,kappa=0.0,compute_tidal=False);rec=[];o=s.derivs
    s.derivs=lambda r,y:(rec.append((r,np.array(y,float))),o(r,y))[1]
    res=s.solve(rc); s.derivs=o
    R=res['r_profile'][-1]; M=res['M_profile'][-1]/1.989e33
    fr=[abs(o(r,y)[2]-sb.derivs(r,y)[2])/abs(sb.derivs(r,y)[2]) for r,y in rec if r>1e3 and y[2]>3e21 and r<0.99*R]
    R0c=eos.get_R0_derivs(eos.get_P_eps_from_rho(rc)[0])[0]
    return dict(alpha=a,rhoc=rc,M=M,R=R/1e5,eps_alpha=max(fr),aR0=abs(a*R0c),ell_km=np.sqrt(6*abs(a))/1e5,tau_ms=np.sqrt(6*abs(a))/c*1e3)
out=[]
for a in [-3e12,-1e13,-2e13,-3.5e13,2e13]:
    for rc in [1e10,1e11]:
        d=diag(a,rc); out.append(d); print(json.dumps(d),flush=True)
M0=TOVSolver(eos,compute_tidal=False).solve(1e11)['M_profile'][-1]/1.989e33
print('GR at 1e11',M0)
json.dump(out,open('fr_diag.json','w'))
