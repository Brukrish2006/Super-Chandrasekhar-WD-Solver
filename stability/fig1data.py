from self_consistent import *
import pickle
rhos=np.logspace(7,np.log10(5e10),36)
cases={'GR':dict(mag=False),'kappa':dict(mag=False,kappa=0.15),'mag':dict(B0=3.79e14),'magk':dict(B0=3.79e14,kappa=0.15)}
out={}
for k,kw in cases.items():
    out[k]=np.array([equilibrium(Model(**kw),r)[:2] for r in rhos])
# self-consistent at 3.79e14 along rho
sc=[]
for r in rhos[rhos>=1e8]:
    kk,m,R,_=closure(3.79e14,r); sc.append((r,m,R,kk))
out['sc']=np.array(sc)
prof={}
for B0,k in [(3.79e14,0.1459),(1e14,0.0111),(1e13,0.0001)]:
    a,x,kB=kappa_avg(Model(B0=B0,kappa=k),1e10,cut=0.002)
    prof[B0]=(x,kB,a)
pickle.dump((rhos,out,prof),open('fig1data.pkl','wb'))
print('done')
