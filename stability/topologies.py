"""Referee 2.2: robustness of the self-consistent anisotropy kappa* to the field topology.
Profiles are compared at equal *central* field B_c = B(rho_c), rho_c = 1e10 g/cm^3.
  deb        : Deb et al. (2022) profile, eta=0.2, gamma=0.9 (fiducial)
  deb_eta01  : same with eta=0.1 (value quoted in Deb et al. text)
  frozen     : flux-freezing scaling B = Bs + B1 (rho/rho_c)^(2/3)
  core       : field confined to the core, B = Bs + B1*0.5*[1+tanh(ln(rho/1e8))]
  transverse : fiducial profile with Delta = B^2/4pi (kappa_B doubled)
"""
from self_consistent import *
import json
RHOC=1e10
class FrozenModel(Model):
    def __init__(self,Bc,rhoc=RHOC,**kw):
        super().__init__(B0=1.0,**kw); self.B1=Bc-self.Bs; self.rhoc=rhoc
    def B(self,rho):
        t=(rho/self.rhoc)**(2/3); return self.Bs+self.B1*t, self.B1*(2/3)*t/rho
class CoreModel(Model):
    def __init__(self,Bc,rhoc=RHOC,rcore=1e8,**kw):
        super().__init__(B0=1.0,**kw); self.rcore=rcore
        self.B1=(Bc-self.Bs)/(0.5*(1+np.tanh(np.log(rhoc/rcore))))
    def B(self,rho):
        th=np.tanh(np.log(rho/self.rcore))
        return self.Bs+self.B1*0.5*(1+th), self.B1*0.5*(1-th*th)/rho
def deb_B0(Bc,eta=0.2,gam=0.9,Bs=1e9):
    x=eta*(RHOC/1e9)**gam; return (Bc-Bs)/(1-np.exp(-x))
def make(top,Bc,k):
    if top=='deb' or top=='transverse': return Model(B0=deb_B0(Bc),kappa=k)
    if top=='deb_eta01': return Model(B0=deb_B0(Bc,eta=0.1),kappa=k,eta=0.1)
    if top=='frozen': m=FrozenModel(Bc); m.k=k; return m
    if top=='core': m=CoreModel(Bc); m.k=k; return m
def closure_top(top,Bc,k0=0.15,tol=1e-5):
    fac=2.0 if top=='transverse' else 1.0
    k=k0
    for it in range(60):
        kn=fac*kappa_avg(make(top,Bc,k),RHOC)[0]
        if abs(kn-k)<tol: k=kn;break
        k=kn
    m,R,_=equilibrium(make(top,Bc,k),RHOC)
    return k,m,R
if __name__=='__main__':
    out=[]
    for top in ['deb','deb_eta01','frozen','core','transverse']:
        for Bc in np.logspace(12,np.log10(3.5e14),12):
            k,m,R=closure_top(top,Bc)
            mf=equilibrium(make(top,Bc,0.15),RHOC)[0]
            d=dict(top=top,Bc=Bc,kstar=k,M_sc=m,R_sc=R,M_fixed=mf); out.append(d); print(json.dumps(d),flush=True)
    json.dump(out,open('topologies.json','w'))
