"""GR radial pulsations of magnetized, Bowers-Liang anisotropic white dwarfs (alpha=0).
Geometric units internally: lengths cm, energy densities/pressures in cm^-2 (x G/c^4), masses in cm (x G/c^2).
Equilibrium: P = P_fl + B^2/8pi, eps = eps_fl + B^2/8pi, dP/dr = -(1-k/3)(eps+P) a,
             a = (m+4 pi r^3 P)/(r^2 (1-2m/r)) = nu'/2.
Perturbation (Lagrangian xi = Delta r / r, DP = Lagrangian radial-pressure perturbation):
  Dn/n = -(r xi' + 3 xi) + a r xi
  DP   = GP * Dn/n,  De = GE * Dn/n
  w^2 e^{lam-nu} Q u = dP_E' + g[(de+dP_E) a + Q da],  u = r xi, Q = eps+P, g = 1-k/3
  (anisotropy perturbed as the same Bowers-Liang functional -> factor g on gravity terms)
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
c=2.99792458e10; G=6.67430e-8; hbar=1.054571817e-27; h=2*np.pi*hbar
m_e=9.1093837e-28; m_u=1.66053906660e-24; mu_e=2.0
K_rho=(8*np.pi*mu_e*m_u*(m_e*c)**3)/(3*h**3); K_P=(np.pi*m_e**4*c**5)/(3*h**3)
Msun=1.989e33
fP=G/c**4   # pressure/energy-density -> cm^-2
fM=G/c**2

class Model:
    def __init__(self,B0=0.0,kappa=0.0,eta=0.2,gam=0.9,rho0=1e9,Bs=1e9,mag=True,bmode='eq'):
        self.B0,self.k,self.eta,self.gam,self.rho0,self.Bs=B0,kappa,eta,gam,rho0,(Bs if mag else 0.0)
        self.mag=mag; self.bmode=bmode   # 'eq' (follow B(rho)), 'frozen43' (B~n^2/3), 'frozen2' (B~n)
    def B(self,rho):
        if not self.mag: return 0.0,0.0
        x=self.eta*(rho/self.rho0)**self.gam
        B=self.Bs+self.B0*(1-np.exp(-x))
        dB=self.B0*np.exp(-x)*self.gam*x/rho      # dB/drho
        return B,dB
    def thermo(self,rho):
        """returns P, eps, dP/drho, deps/drho (cgs), and fluid parts"""
        x=(rho/K_rho)**(1/3); s=np.sqrt(1+x*x)
        Pf=K_P*(x*(2*x*x-3)*s+3*np.arcsinh(x))
        ef=rho*c*c+3*K_P*(x*(2*x*x+1)*s-np.arcsinh(x))   # K_eps = 3 K_P
        dxdr=x/(3*rho)
        dPf=K_P*8*x**4/s*dxdr
        def_=c*c+3*K_P*8*x*x*s*dxdr
        B,dB=self.B(rho); PB=B*B/(8*np.pi); dPB=B*dB/(4*np.pi)
        return Pf+PB, ef+PB, dPf+dPB, def_+dPB, Pf, ef, PB, dPf
    def gammas(self,rho):
        """Adiabatic response per Dn/n (n ~ rho): returns (GP*P, GE) in cgs"""
        P,e,dP,de,Pf,ef,PB,dPf=self.thermo(rho)
        if self.bmode=='eq':
            # perturbation follows the equilibrium barotrope; first law defines n: De=(e+P)Dn/n
            GPP=(e+P)*dP/de; GE=(e+P)
            X=Y=0.0
        elif self.bmode=='toroidal':
            # flux freezing of a toroidal field in radial flow: B/(rho r) conserved
            # => DB/B = Dn/n + xi ;  DP_B = De_B = 2 P_B (Dn/n + xi)
            Ge=rho*dPf/Pf
            GPP=Ge*Pf+2*PB; GE=(ef+Pf)+2*PB; X=Y=2*PB
        else:
            gb={'frozen43':4/3,'frozen2':2.0}[self.bmode]
            Ge=rho*dPf/Pf
            GPP=Ge*Pf+gb*PB
            GE=(ef+Pf)+gb*PB
            X=Y=0.0
        return GPP,GE,X,Y

def rhs(r,y,M,w2):
    rho,m,nu,xi,DP=y
    rho=max(rho,1.0)
    P,e,dP,de,*_=M.thermo(rho)
    P*=fP;e*=fP;dPdrho=dP*fP;dedrho=de*fP
    GPP,GE,X,Y=M.gammas(rho); GPP*=fP; GE*=fP; X*=fP; Y*=fP
    g=1-M.k/3
    Q=e+P
    el=1/(1-2*m/r)
    a=(m+4*np.pi*r**3*P)*el/r**2
    Pp=-g*Q*a
    rhop=Pp/dPdrho
    ep=dedrho*rhop
    mp=4*np.pi*r*r*e
    # a'
    num=m+4*np.pi*r**3*P; den=r*r-2*m*r
    nump=mp+12*np.pi*r*r*P+4*np.pi*r**3*Pp; denp=2*r-2*m-2*mp*r
    ap=(nump*den-num*denp)/den**2
    Qp=ep+Pp
    Ppp=-g*(Qp*a+Q*ap)
    # perturbations
    dnn=(DP-X*xi)/GPP
    xip=(-3*xi-dnn+a*r*xi)/r
    u=r*xi; up=xi+r*xip
    De=GE*dnn+Y*xi
    dPE=DP-Pp*u; deE=De-ep*u
    lamp=8*np.pi*r*el*e-(el-1)/r
    dlam=-(lamp+2*a)*u
    dnup=dlam*(2*a+1/r)+8*np.pi*r*el*dPE
    da=dnup/2
    # w^2 e^{lam-nu} Q u = dPE' + g[(deE+dPE)a + Q da]; dPE' = DP' - (Pp u)'
    DPp=w2*el*np.exp(-nu)*Q*u - g*((deE+dPE)*a+Q*da) + Ppp*u+Pp*up
    nup=2*a
    return [rhop,mp,nup,xip,DPp]

def surf(r,y,M,w2): return y[0]-1e4
surf.terminal=True; surf.direction=-1

def integrate(M,rhoc,w2,r0=10.0,dense=False):
    P,e,*_=M.thermo(rhoc); GPP,_,X0,_=M.gammas(rhoc)
    m0=4/3*np.pi*r0**3*e*fP
    y0=[rhoc,m0,0.0,1.0,(-3*GPP+X0)*fP*1.0]
    sol=solve_ivp(rhs,(r0,3e9),y0,args=(M,w2),events=surf,method='LSODA',rtol=1e-9,atol=[1e-6*rhoc,1e-30,1e-14,1e-12,1e-40],dense_output=dense)
    return sol

def equilibrium(M,rhoc):
    s=integrate(M,rhoc,0.0)
    R=s.t[-1]; m=s.y[1,-1]
    return m/fM/Msun, R/1e5, s

def shoot(M,rhoc,w2):
    s=integrate(M,rhoc,w2)
    xi=s.y[3]; DP=s.y[4]
    P_c=M.thermo(rhoc)[0]*fP
    return DP[-1]/P_c, np.sum(np.diff(np.sign(xi))!=0)

def omega0sq(M,rhoc,lo=-5.0,hi=5.0,n=41):
    """fundamental eigenvalue in units of G*rho_c ... returned as w^2 [s^-2] (coordinate time at infinity)."""
    Mm,R,s=equilibrium(M,rhoc)
    nu_R=s.y[2,-1]; mR=s.y[1,-1]; Rr=s.t[-1]
    C=np.log(1-2*mR/Rr)-nu_R      # nu_true = nu_int + C
    scale=G*rhoc/c**2               # cm^-2
    xs=np.linspace(lo,hi,n); fs=[]
    for x in xs:
        f,_=shoot(M,rhoc,x*scale); fs.append(f)
    fs=np.array(fs)
    for i in range(n-1):
        if np.sign(fs[i])!=np.sign(fs[i+1]):
            x0=brentq(lambda x:shoot(M,rhoc,x*scale)[0],xs[i],xs[i+1],xtol=1e-10)
            # Omega^2 = w2 e^{-C}  => physical w^2 (geom) = x*scale*e^{C}
            w2=x0*scale*np.exp(C)*c**2
            return w2,Mm,R
    return np.nan,Mm,R
