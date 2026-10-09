from radial_pulsation import *
import pickle
rhos=np.logspace(9,np.log10(5e10),22)
out={}
for lab,kw in [('GR',dict(mag=False)),('ext',dict(B0=3.79e14,kappa=0.15)),('1e14',dict(B0=1e14,kappa=0.15))]:
    for md in (['eq'] if lab=='GR' else ['eq','toroidal','frozen43']):
        row=[]
        for rc in rhos:
            w2,m,R=omega0sq(Model(bmode=md,**kw),rc,lo=-6,hi=6,n=61); row.append((rc,m,R,w2))
        out[(lab,md)]=np.array(row); print(lab,md,'done',flush=True)
pickle.dump(out,open('figseq.pkl','wb'))
