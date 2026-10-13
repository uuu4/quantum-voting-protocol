import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
r=json.load(open('results.json')); r2=json.load(open('results2.json'))
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.8,"axes.grid":True,"grid.alpha":.25,"grid.linewidth":.5,"legend.frameon":False,"lines.linewidth":1.6,"lines.markersize":5})
C1,C2,C3,C4="#0072B2","#D55E00","#009E73","#CC79A7"
fig,ax=plt.subplots(1,3,figsize=(7.2,2.5))
# (a) depolarizing after voting: retained margin vs p for K
for K,c,m in ((2,C1,"o"),(3,C2,"s"),(5,C3,"^"),(8,C4,"D")):
    pts=[b for b in r['B'] if b['K']==K]; x=[b['p'] for b in pts]; y=[b['gap_ratio']/K for b in pts]
    ax[0].plot(x,y,color=c,marker=m,label=f"K={K}")
ax[0].set_xlabel("depolarizing $p$ per candidate qubit"); ax[0].set_ylabel("retained margin $\\rho$"); ax[0].set_ylim(0,1.05); ax[0].legend(loc="lower left",fontsize=7); ax[0].set_title("(a) after voting",fontsize=9)
# (b) memory noise during voting: rho vs N (leader on last label)
for kind,val,c,m,lab in (("dephase",1e-4,C1,"o","dephasing $10^{-4}$"),("dephase",1e-3,C2,"s","dephasing $10^{-3}$"),("damp",1e-4,C3,"^","damping $10^{-4}$"),("damp",1e-3,C4,"D","damping $10^{-3}$")):
    pts=[b for b in r2['C'] if b['kind']==kind and b['val']==val]; ax[1].plot([b['N'] for b in pts],[b['rho'] for b in pts],color=c,marker=m,label=lab)
ax[1].axhline(0,color="k",lw=.8); ax[1].set_xscale("log",base=2); ax[1].set_xlabel("number of voters $N$"); ax[1].set_ylabel("retained margin $\\rho$"); ax[1].set_ylim(-1.2,1.1); ax[1].legend(loc="lower left",fontsize=6.5); ax[1].set_title("(b) memory error per voting step",fontsize=9)
# (c) gate errors during voting: total-variation distortion with 95% CI vs errors per run
allE=[(e['N'],e['errors_per_run'],e['tv'],e['tv_ci']) for e in r['E']]+[(e['N'],e['errors_per_run'],e['tv'],e['tv_ci']) for e in r2['E128']]
for N,c,m in ((8,C1,"o"),(16,C2,"s"),(32,C3,"^"),(64,C4,"D"),(128,"k","v")):
    pts=[p for p in allE if p[0]==N]; x=np.array([p[1] for p in pts]); y=np.array([p[2] for p in pts])
    lo=y-np.array([p[3][0] for p in pts]); hi=np.array([p[3][1] for p in pts])-y
    ax[2].errorbar(x,y,yerr=[lo,hi],color=c,marker=m,capsize=2,lw=1.2,label=f"N={N}")
ax[2].set_xscale("log"); ax[2].set_xlabel("expected gate errors per run"); ax[2].set_ylabel("tally distortion (TV)"); ax[2].legend(loc="upper left",fontsize=7); ax[2].set_title("(c) gate errors during voting",fontsize=9)
plt.tight_layout(); plt.savefig("noise_margin.png",dpi=300)
# Figure 2: winner identification error vs bound (CI)
fig,ax=plt.subplots(1,2,figsize=(7.2,2.4))
labs=[f"N={a['N']},K={a['K']}" for a in r['A']]; x=np.arange(len(labs)); w=.38
for i,(key,ci,c,lab) in enumerate((("err_hoeff","ci_hoeff",C1,"Hoeffding $R$"),("err_bern","ci_bern",C2,"Bernstein $R$"))):
    y=np.array([a[key] for a in r['A']]); lo=y-np.array([a[ci][0] for a in r['A']]); hi=np.array([a[ci][1] for a in r['A']])-y
    ax[0].bar(x+(i-.5)*w,y,w*.9,color=c,label=lab); ax[0].errorbar(x+(i-.5)*w,y,yerr=[lo,hi],fmt="none",ecolor="k",lw=.8,capsize=2)
ax[0].axhline(0.05,color="k",ls="--",lw=.8); ax[0].text(-0.4,0.052,"target $\\delta=0.05$",ha="left",fontsize=7)
ax[0].set_xticks(x); ax[0].set_xticklabels(labs,rotation=35,ha="right",fontsize=7); ax[0].set_ylabel("empirical error (95% CI)"); ax[0].set_ylim(0,0.075); ax[0].legend(fontsize=7,loc="upper right",ncol=2); ax[0].set_title("(a) winner identification, $\\Delta=0.1$",fontsize=9)
ex=[(a['K'],a['exec_mean'],a['exec_pred'],a['exec_q99'],a['exec_bound']) for a in r['A'] if a['N']==1000]
ax[1].plot([e[0] for e in ex],[e[1] for e in ex],color=C1,marker="o",label="mean executions"); ax[1].plot([e[0] for e in ex],[e[2] for e in ex],color="k",ls=":",label="$K R_{\\rm winner}$")
ax[1].plot([e[0] for e in ex],[e[3] for e in ex],color=C2,marker="s",label="99th percentile"); ax[1].plot([e[0] for e in ex],[e[4] for e in ex],color="k",ls="--",label="$2KR_{\\rm winner}$")
ax[1].set_yscale("log"); ax[1].set_xlabel("candidates $K$"); ax[1].set_ylabel("protocol executions"); ax[1].legend(fontsize=7,loc="lower right"); ax[1].set_title("(b) cost of post-selection",fontsize=9)
plt.tight_layout(); plt.savefig("sampling_check.png",dpi=300); print("figs ok")
