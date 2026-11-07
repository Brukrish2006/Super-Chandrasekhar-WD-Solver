from topologies import *
out=[]
for top in ['deb','deb_eta01','frozen','core','transverse']:
    for Bc in [1e13]:
        k,m,R=closure_top(top,Bc); mf=equilibrium(make(top,Bc,0.15),RHOC)[0]
        d=dict(top=top,Bc=Bc,kstar=k,M_sc=m,M_fixed=mf); out.append(d); print(json.dumps(d),flush=True)
json.dump(out,open('topologies_1e13.json','w'))
