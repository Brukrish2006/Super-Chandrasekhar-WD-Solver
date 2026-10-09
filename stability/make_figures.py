import pickle,json,numpy as np,matplotlib,shutil
import os; os.makedirs('figures',exist_ok=True)
matplotlib.use('Agg');import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10.5})
P='results/'
rhos,out,prof=pickle.load(open('fig1data.pkl','rb'))
fig,ax=plt.subplots(1,2,figsize=(11,4.3))
st={'GR':('k','GR, $B=0$'),'kappa':('0.55','$\\kappa=0.15$, $B=0$'),'mag':('#1f4e9c','$B_0=3.79\\times10^{14}$ G, $\\kappa=0$'),'magk':('#c2410c','$B_0=3.79\\times10^{14}$ G, $\\kappa=0.15$')}
for k,(c,l) in st.items():
    A=out[k]; sel=rhos<=1e10
    ax[0].plot(A[sel,0],A[sel,1],color=c,lw=2,label=l); ax[1].semilogx(rhos,A[:,0],color=c,lw=2,label=l)
S=out['sc']; sel=S[:,0]<=1e10
ax[0].plot(S[sel,1],S[sel,2],color='#0f766e',lw=2,ls='--',label='$B_0=3.79\\times10^{14}$ G, $\\kappa=\\kappa^*$')
ax[1].semilogx(S[:,0],S[:,1],color='#0f766e',lw=2,ls='--')
for a in ax: a.grid(alpha=.3,ls='--')
ax[1].axvline(1e10,color='0.4',ls=':');ax[1].axvline(2.06e10,color='0.4',ls='-.')
ax[0].set_xlabel('$M$ ($M_\\odot$)');ax[0].set_ylabel('$R$ (km)');ax[0].set_ylim(0,8000);ax[0].legend(fontsize=8)
ax[1].set_xlabel('$\\rho_c$ (g cm$^{-3}$)');ax[1].set_ylabel('$M$ ($M_\\odot$)')
ax[0].set_title('(a) Mass–radius, $\\rho_c\\leq10^{10}$ g cm$^{-3}$');ax[1].set_title('(b) Mass vs central density')
plt.tight_layout();plt.savefig('figures/mass_radius.png',dpi=200);plt.close()
K=json.load(open(P+'kstar_vs_B0.json'))
B=np.array([d['B0'] for d in K]);ks=np.array([d['kstar'] for d in K]);Ms=np.array([d['M_sc'] for d in K]);Mf=np.array([d['M_fixed'] for d in K])
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
ax[0].loglog(B,ks,'o-',color='#0f766e',lw=2,ms=4)
ax[0].axhline(0.15,color='#c2410c',ls='--',lw=1.2,label='conventional $\\kappa=0.15$')
ax[0].axhline(0.01,color='0.5',ls=':',lw=1.2,label='$\\kappa^*=0.01$')
ax[0].axvspan(1e12,1e13,color='0.88',zorder=0,label='$B_0\\leq10^{13}$ G (stability bound)')
ax[0].set_xlabel('$B_0$ (G)');ax[0].set_ylabel('$\\kappa^*=\\langle\\kappa_B\\rangle_V$');ax[0].legend(fontsize=8,loc='upper left');ax[0].set_title('(a) Self-consistent anisotropy, $\\rho_c=10^{10}$ g cm$^{-3}$',fontsize=10)
ax[1].semilogx(B,Mf,'-',color='#c2410c',lw=2,label='fixed $\\kappa=0.15$')
ax[1].semilogx(B,Ms,'-',color='#0f766e',lw=2,label='self-consistent $\\kappa^*$')
ax[1].axvspan(1e12,1e13,color='0.88',zorder=0)
ax[1].set_xlabel('$B_0$ (G)');ax[1].set_ylabel('$M$ at $\\rho_c=10^{10}$ g cm$^{-3}$ ($M_\\odot$)');ax[1].legend(fontsize=8);ax[1].set_title('(b) Mass at $\\rho_c=10^{10}$ g cm$^{-3}$',fontsize=10)
for a in ax:a.grid(alpha=.3,ls='--',which='both')
plt.tight_layout();plt.savefig('figures/kstar.png',dpi=200);plt.close()
Ssum=json.load(open(P+'scseq_summary.json'));F=json.load(open(P+'fixedtp.json'))
fig,ax=plt.subplots(figsize=(6.4,4.4))
ax.semilogx([d['B0'] for d in F],[d['M_tp'] for d in F],'s-',color='#c2410c',lw=2,label='fixed $\\kappa=0.15$ (turning point)')
Bs=[d['B0'] for d in Ssum]
ax.semilogx(Bs,[d['tp'][1] for d in Ssum],'o-',color='#0f766e',lw=2,label='self-consistent $\\kappa^*$ (turning point)')
for md,c,mk,l in [('toroidal','#7c3aed','^','self-consistent, toroidal flux frozen'),('frozen43','#0369a1','v','self-consistent, tangled flux frozen')]:
    ax.semilogx(Bs,[d[md][2] for d in Ssum],mk,color=c,ms=7,mfc='none',mew=1.6,label=l)
ax.axvspan(1e12,1e13,color='0.88',zorder=0);ax.set_xlim(7e12,6e14)
ax.set_xlabel('$B_0$ (G)');ax.set_ylabel('maximum stable mass ($M_\\odot$)');ax.grid(alpha=.3,ls='--',which='both');ax.legend(fontsize=8)
plt.tight_layout();plt.savefig('figures/max_mass.png',dpi=200);plt.close()
fig,ax=plt.subplots(figsize=(6.4,4.2))
labs={3.79e14:'$3.79\\times10^{14}$',1e14:'$10^{14}$',1e13:'$10^{13}$'}
for (B0,(x,kB,a)),c in zip(prof.items(),['#c2410c','#7c3aed','#0f766e']):
    ax.semilogy(x,kB,color=c,lw=2,label=f'$B_0=${labs[B0]} G, '+'$\\langle\\kappa_B\\rangle_V=$'+f'{a:.2g}')
ax.axhline(0.15,color='0.4',ls='--',lw=1);ax.set_xlabel('$r/R$');ax.set_ylabel('$\\kappa_B(r)$');ax.grid(alpha=.3,ls='--',which='both');ax.legend(fontsize=8);ax.set_ylim(1e-7,1e3)
plt.tight_layout();plt.savefig('figures/kappa_profile.png',dpi=200);plt.close()
pass
print('ok')
# Radial-pulsation figure (needs figseq.pkl from pulsation_sequences.py)
if os.path.exists('figseq.pkl'):
    D=pickle.load(open('figseq.pkl','rb'))
    col={'eq':'#1f4e9c','toroidal':'#c2410c','frozen43':'#0f766e'}
    lab={'eq':'field follows $B(\\rho)$','toroidal':'toroidal flux frozen','frozen43':'tangled flux frozen ($\\Gamma_B=4/3$)'}
    fig,ax=plt.subplots(2,2,figsize=(11,7),sharex=True,gridspec_kw={'height_ratios':[1,1.3]})
    for j,(key,title) in enumerate([('ext',r'$B_0=3.79\times10^{14}$ G, $\kappa=0.15$'),('1e14',r'$B_0=10^{14}$ G, $\kappa=0.15$')]):
        A=D[(key,'eq')]; G=D[('GR','eq')]
        ax[0,j].semilogx(A[:,0],A[:,1],'k-',lw=2); ax[0,j].semilogx(G[:,0],G[:,1],color='0.6',ls='--',lw=1.5,label='GR, $B=0$')
        ax[0,j].set_ylabel(r'$M$ ($M_\odot$)');ax[0,j].set_title(title)
        for md in ['eq','toroidal','frozen43']:
            B=D[(key,md)]; ax[1,j].semilogx(B[:,0],B[:,3],color=col[md],lw=2,label=lab[md])
        ax[1,j].semilogx(G[:,0],G[:,3],color='0.6',ls='--',lw=1.5,label='GR, $B=0$')
        ax[1,j].set_yscale('symlog',linthresh=1);ax[1,j].axhline(0,color='k',lw=.8)
        ax[1,j].set_ylabel(r'$\omega_0^2$ (s$^{-2}$)');ax[1,j].set_xlabel(r'$\rho_c$ (g cm$^{-3}$)')
        for a in ax[:,j]:
            a.axvline(1e10,color='0.3',ls=':',lw=1);a.axvline(4e10,color='0.3',ls='-.',lw=1);a.grid(alpha=.25,ls='--')
        ax[0,j].legend(fontsize=8,loc='lower right')
    ax[1,0].legend(fontsize=8,loc='lower left')
    plt.tight_layout();plt.savefig('figures/pulsation.png',dpi=200);plt.close()
print('figures written to stability/figures/')
