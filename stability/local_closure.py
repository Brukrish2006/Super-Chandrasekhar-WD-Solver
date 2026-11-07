"""Referees 2.1/3.1: point-by-point closure kappa(r) = kappa_B(r) during integration.
The local magnetic anisotropy Delta = B^2/8pi does not vanish at the centre, so
kappa_B = 6 P_B/(r Q a) diverges as r^-2 there (main text Sect. 2.2). We therefore cap
kappa(r) = min(kappa_B(r), kappa_max); kappa_max = 3 is the largest value for which the
effective gravity (1 - kappa/3) stays non-negative. Equilibrium only (alpha = 0)."""
from self_consistent import *
from topologies import deb_B0
import json
def rhs_local(r,y,M,kmax):
    rho,m=y; rho=max(rho,1.0)
    P,e,dP,de,Pf,ef,PB,dPf=M.thermo(rho)
    P*=fP;e*=fP;dPdrho=dP*fP;PB*=fP
    Q=e+P; el=1/(1-2*m/r); a=(m+4*np.pi*r**3*P)*el/r**2
    kB=6*PB/(r*Q*a) if a>0 else kmax
    k=min(kB,kmax)
    return [-(1-k/3)*Q*a/dPdrho, 4*np.pi*r*r*e]
def surf2(r,y,M,kmax): return y[0]-1e4
surf2.terminal=True; surf2.direction=-1
def eq_local(M,rhoc,kmax,r0=10.0,rtol=1e-9):
    P,e,*_=M.thermo(rhoc)
    s=solve_ivp(rhs_local,(r0,3e9),[rhoc,4/3*np.pi*r0**3*e*fP],args=(M,kmax),events=surf2,method='LSODA',rtol=rtol,atol=[1e-6*rhoc,1e-30],dense_output=True)
    return s.y[1,-1]/fM/Msun, s.t[-1]/1e5, s
def capped_fraction(M,s,kmax,N=4000):
    """volume fraction where kappa_B exceeds the cap"""
    R=s.t[-1]; r=np.linspace(1e-4*R,R*0.999,N); Y=s.sol(r); cnt=0
    for ri,rh,mi in zip(r,np.maximum(Y[0],1),Y[1]):
        P,e,dP,de,Pf,ef,PB,dPf=M.thermo(rh); P*=fP;e*=fP;PB*=fP
        a=(mi+4*np.pi*ri**3*P)/(ri*ri*(1-2*mi/ri))
        if 6*PB/(ri*(e+P)*a)>kmax: cnt+=ri*ri
    return cnt/np.sum(r*r)
if __name__=='__main__':
    out=[]
    for Bc in [8e12,8e13,1.2e14,3.03e14]:
        B0=deb_B0(Bc); M0=Model(B0=B0)
        kv,mv,Rv,_=closure(B0,1e10)
        for kmax in [0.5,1.0,3.0]:
            m,R,s=eq_local(M0,1e10,kmax)
            d=dict(Bc=Bc,B0=B0,kmax=kmax,M_local=m,R_local=R,capped_vol=capped_fraction(M0,s,kmax),kstar_vol=kv,M_vol=mv,R_vol=Rv)
            out.append(d); print(json.dumps(d),flush=True)
    json.dump(out,open('local_closure.json','w'))
