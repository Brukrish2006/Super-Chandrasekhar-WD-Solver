"""Referee 3.1: geometrically consistent magnetic closures.
Spherically averaged Maxwell stress for three field geometries (energy density B^2/8pi in all):
  TO (toroidal/transverse): P_r^B = +B^2/8pi, <P_t^B> = 0        -> Delta_B = -B^2/8pi
  RO (radial):              P_r^B = -B^2/8pi, P_t^B = +B^2/8pi   -> Delta_B = +B^2/4pi
  tangled (isotropic):      P^B   = B^2/24pi                     -> Delta_B = 0
Self-consistent kappa*: fixed point of kappa = <6 Delta_B / (r (eps+P_r) a)>_V.
Also 'magnitude': P_r^B=+B^2/8pi with Delta=+B^2/8pi (the v3 closure, an upper bound)."""
from radial_pulsation import Model, c, G, fP, fM, Msun
from topologies import deb_B0
import numpy as np, json
from scipy.integrate import solve_ivp
GEO={'TO':(1.0,-1.0),'RO':(-1.0,2.0),'tangled':(1/3,0.0),'magnitude':(1.0,1.0)}  # (s_r, f_Delta) in units of B^2/8pi
def thermo(M,rho,sr):
    P,e,dP,de,Pf,ef,PB,dPf=M.thermo(rho)
    B,dB=M.B(rho); dPB=B*dB/(4*np.pi)
    Pr=Pf+sr*PB; dPr=dPf+sr*dPB
    return Pr,e,dPr,PB
def rhs(r,y,M,sr,k):
    rho,m=y; rho=max(rho,1.0)
    Pr,e,dPr,PB=thermo(M,rho,sr); Pr*=fP;e*=fP;dPr*=fP
    Q=e+Pr; el=1/(1-2*m/r); a=(m+4*np.pi*r**3*Pr)*el/r**2
    return [-(1-k/3)*Q*a/dPr, 4*np.pi*r*r*e]
def ev_rho(r,y,M,sr,k): return y[0]-1e4
ev_rho.terminal=True; ev_rho.direction=-1
def ev_P(r,y,M,sr,k): return thermo(M,max(y[0],1.0),sr)[0]
ev_P.terminal=True; ev_P.direction=-1
def ev_dP(r,y,M,sr,k): return thermo(M,max(y[0],1.0),sr)[2]   # dP_r/drho -> 0: model breaks down
ev_dP.terminal=True; ev_dP.direction=-1
def solve(M,rhoc,sr,k):
    e=M.thermo(rhoc)[1]; r0=10.0
    s=solve_ivp(rhs,(r0,3e9),[rhoc,4/3*np.pi*r0**3*e*fP],args=(M,sr,k),events=[ev_rho,ev_P,ev_dP],method='LSODA',rtol=1e-9,atol=[1e-6*rhoc,1e-30],dense_output=True)
    return s
def kavg(M,s,sr,fD,cut=0.01,N=3000):
    R=s.t[-1]; r=np.linspace(cut*R,(1-cut)*R,N); Y=s.sol(r); kb=[]
    for ri,rh,mi in zip(r,np.maximum(Y[0],1.0),Y[1]):
        Pr,e,dPr,PB=thermo(M,rh,sr); Pr*=fP;e*=fP;PB*=fP
        a=(mi+4*np.pi*ri**3*Pr)/(ri*ri*(1-2*mi/ri)); kb.append(6*fD*PB/(ri*(e+Pr)*a))
    w=r*r; return np.trapezoid(np.array(kb)*w,r)/np.trapezoid(w,r)
def closure(Bc,geo,rhoc=1e10):
    sr,fD=GEO[geo]; M=Model(B0=deb_B0(Bc)); k=0.0
    for it in range(80):
        s=solve(M,rhoc,sr,k); kn=kavg(M,s,sr,fD)
        if abs(kn-k)<1e-6: k=kn; break
        k=0.5*k+0.5*kn
    s=solve(M,rhoc,sr,k); ended='rho' if s.t_events[0].size else ('P' if s.t_events[1].size else ('dP' if s.t_events[2].size else 'none'))
    s0=solve(M,rhoc,sr,0.0)
    return dict(geo=geo,Bc=Bc,kstar=k,M=s.y[1,-1]/fM/Msun,R=s.t[-1]/1e5,M_k0=s0.y[1,-1]/fM/Msun,end=ended)
if __name__=='__main__':
    out=[]
    for geo in ['magnitude','TO','RO','tangled']:
        for Bc in [1e12,1e13,3e13,1e14,2e14,3.03e14]:
            d=closure(Bc,geo); out.append(d); print(json.dumps(d),flush=True)
    json.dump(out,open('geometry_closure.json','w'))
