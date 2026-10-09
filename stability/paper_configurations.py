from radial_pulsation import *
import json
cfg=[('stable-regime SC',1e13,0.0,1e10),('fixed-k conservative',1e13,0.15,1e10),
     ('1.60 pair (a=0 analog)',1.5e14,0.025,1e10),('1.60 pair (a=0 analog)',1e14,0.012,1e10),
     ('extreme fixed-k',3.79e14,0.15,1e10),
     ('carbon window (a=0 analog)',5e13,0.0026,4e10),('carbon window (a=0 analog)',1e14,0.012,4e10),
     ('carbon window (a=0 analog)',1.5e14,0.025,4e10),('carbon window (a=0 analog)',2e14,0.040,4e10)]
out=[]
for lab,B0,k,rc in cfg:
    d=dict(label=lab,B0=B0,kappa=k,rhoc=rc)
    for md in ['eq','toroidal','frozen43']:
        w2,m,R=omega0sq(Model(B0=B0,kappa=k,bmode=md),rc,lo=-6,hi=6,n=61)
        d[md]=w2; d['M']=m; d['R']=R
    print(json.dumps(d),flush=True); out.append(d)
json.dump(out,open('configs.json','w'))
