"""Summarize self-consistent sequences (scseq_*.json): turning point, onset of
radial instability for each field prescription, and maximum stable mass."""
import json,glob,numpy as np
from scipy.interpolate import CubicSpline
res=[]
for f in sorted(glob.glob('scseq_*.json'),key=lambda f:float(f[6:-5])):
    B0=float(f[6:-5]); D=json.load(open(f))
    l=np.log10([d['rhoc'] for d in D]); M=np.array([d['M'] for d in D]); k=np.array([d['kstar'] for d in D])
    sM=CubicSpline(l,M); lf=np.linspace(l[0],l[-1],4000); Mf=sM(lf)
    i=np.argmax(Mf); tp=(10**lf[i],Mf[i])
    row=dict(B0=B0,tp=tp,k_at_1e10=float(CubicSpline(l,k)(10.0)),M_at_1e10=float(sM(10.0)))
    for md in ['eq','toroidal','frozen43']:
        w=np.array([d[md] for d in D]); z=None
        for j in range(len(w)-1):
            if w[j]>0>=w[j+1]:
                z=l[j]+(l[j+1]-l[j])*w[j]/(w[j]-w[j+1]); break
        row[md]=(None,None,None) if z is None else (10**z,float(sM(z)),float(Mf[lf<=z].max()))
    res.append(row); print(row)
json.dump(res,open('scseq_summary.json','w'))
