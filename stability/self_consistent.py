"""Self-consistent kappa closure (GR sector) on top of puls.py."""
from radial_pulsation import *
def profile(M,rhoc):
    s=integrate(M,rhoc,0.0,dense=True)
    return s
def kappa_avg(M,rhoc,cut=0.01,N=4000,s=None):
    if s is None: s=profile(M,rhoc)
    R=s.t[-1]; r=np.linspace(cut*R,(1-cut)*R,N)
    Y=s.sol(r); rho=np.maximum(Y[0],1.0); m=Y[1]
    kB=np.empty_like(r)
    for i,(ri,rh,mi) in enumerate(zip(r,rho,m)):
        P,e,dP,de,Pf,ef,PB,dPf=M.thermo(rh)
        P*=fP;e*=fP;PB*=fP
        a=(mi+4*np.pi*ri**3*P)/(ri*ri*(1-2*mi/ri))
        kB[i]=6*PB/(ri*(e+P)*a)
    w=r*r
    return np.trapezoid(kB*w,r)/np.trapezoid(w,r), r/R, kB
def closure(B0,rhoc,k0=0.15,tol=1e-4,itmax=40,**kw):
    k=k0; hist=[]
    for it in range(itmax):
        M=Model(B0=B0,kappa=k,**kw)
        kn=kappa_avg(M,rhoc)[0]; hist.append(kn)
        if abs(kn-k)<tol: k=kn; break
        k=kn
    M=Model(B0=B0,kappa=k,**kw)
    m,R,_=equilibrium(M,rhoc)
    return k,m,R,it+1
if __name__=='__main__':
    import json,time
    out=[]
    for B0 in np.logspace(12,np.log10(5e14),16):
        t=time.time()
        k,m,R,n=closure(B0,1e10)
        k2,_,_,_=closure(B0,1e10,k0=0.0)
        mf,Rf,_=equilibrium(Model(B0=B0,kappa=0.15),1e10)
        d=dict(B0=B0,kstar=k,kstar_from0=k2,M_sc=m,R_sc=R,iters=n,M_fixed=mf,R_fixed=Rf)
        print(json.dumps(d),'%.0fs'%(time.time()-t),flush=True); out.append(d)
    json.dump(out,open('kstar_vs_B0.json','w'))
