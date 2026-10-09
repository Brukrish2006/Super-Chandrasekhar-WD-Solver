from self_consistent import *
import json
out={}
# (eta,gamma) bracket for kstar at rho_c=1e10
for B0 in [1e13,1e14,1.5e14,3.79e14]:
    vals=[]
    for eta in [0.1,0.2,0.4]:
        for gam in [0.72,0.9,1.08]:
            k,m,R,_=closure(B0,1e10,eta=eta,gam=gam); vals.append((eta,gam,k,m))
    out[f'{B0:.2e}']=vals
    ks=[v[2] for v in vals]; print(B0,'kstar range',min(ks),max(ks),flush=True)
# inner-cut sensitivity at 3.79e14
for cut in [0.005,0.01,0.02]:
    M=Model(B0=3.79e14,kappa=0.146)
    print('cut',cut,kappa_avg(M,1e10,cut=cut)[0])
# Delta=B^2/4pi variant: kappa_B doubles -> closure with factor 2
def closure2(B0,rc,k0=0.15):
    k=k0
    for it in range(40):
        kn=2*kappa_avg(Model(B0=B0,kappa=k),rc)[0]
        if abs(kn-k)<1e-4: k=kn;break
        k=kn
    return k, equilibrium(Model(B0=B0,kappa=k),rc)[0]
for B0 in [1e13,1e14,1.5e14,3.79e14]:
    print('B^2/4pi variant',B0,closure2(B0,1e10),flush=True)
json.dump(out,open('sens.json','w'))
