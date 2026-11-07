import pickle,numpy as np,matplotlib,os
matplotlib.use('Agg');import matplotlib.pyplot as plt
os.makedirs('figures',exist_ok=True)
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
        a.axvline(1e10,color='0.3',ls=':',lw=1); a.axvline(2.06e10,color='#b91c1c',ls='--',lw=1); a.axvline(4.16e10,color='0.3',ls='-.',lw=1); a.grid(alpha=.25,ls='--')
    ax[0,j].legend(fontsize=8,loc='lower right')
ax[1,0].legend(fontsize=8,loc='lower left')
plt.tight_layout();plt.savefig('figures/pulsation.png',dpi=200)
