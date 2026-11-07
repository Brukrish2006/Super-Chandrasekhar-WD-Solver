import json,numpy as np,matplotlib,os
matplotlib.use('Agg');import matplotlib.pyplot as plt
os.makedirs('figures',exist_ok=True)
T=json.load(open('results/topologies.json'))
lab={'deb':'Deb profile, $\\eta=0.2$ (fiducial)','deb_eta01':'Deb profile, $\\eta=0.1$','frozen':'flux-frozen, $B\\propto\\rho^{2/3}$','core':'core-confined (tanh)','transverse':'fiducial, $\\Delta=B^2/4\\pi$'}
col={'deb':'#0f766e','deb_eta01':'#1f4e9c','frozen':'#c2410c','core':'#7c3aed','transverse':'#6b7280'}
mk={'deb':'o','deb_eta01':'s','frozen':'^','core':'D','transverse':'v'}
fig,ax=plt.subplots(1,2,figsize=(11,4.3))
for top in lab:
    r=[d for d in T if d['top']==top and (top!='core' or d['Bc']<5e13)]
    B=[d['Bc'] for d in r]
    ax[0].loglog(B,[d['kstar'] for d in r],marker=mk[top],ms=4,color=col[top],lw=1.6,label=lab[top])
    ax[1].semilogx(B,[d['M_sc'] for d in r],marker=mk[top],ms=4,color=col[top],lw=1.6,label=lab[top])
for a in ax: a.axvline(1e13,color='0.5',ls=':',lw=1); a.grid(alpha=.3,ls='--',which='both'); a.set_xlabel('central field $B_c=B(\\rho_c)$ (G)')
ax[0].axhline(0.01,color='0.5',ls='--',lw=1); ax[0].axhline(0.15,color='#c2410c',ls='--',lw=1)
ax[0].set_ylabel('$\\kappa^*$'); ax[1].set_ylabel('$M$ at $\\rho_c=10^{10}$ g cm$^{-3}$ ($M_\\odot$)'); ax[1].set_ylim(1.4,2.2)
ax[0].legend(fontsize=7.5,loc='upper left'); ax[0].set_title('(a) Self-consistent anisotropy',fontsize=10); ax[1].set_title('(b) Self-consistent mass',fontsize=10)
plt.tight_layout(); plt.savefig('figures/topologies.png',dpi=200)
