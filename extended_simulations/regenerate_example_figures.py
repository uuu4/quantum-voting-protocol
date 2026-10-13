"""Re-plot the Section 9 example figures at single-column size.
Data: values printed in the notebook outputs of quantum_voting_experiments.ipynb (20,000 shots;
convergence sweep 500..100,000 shots), so the figures show exactly the same results as before.
Style matches the revision figures (serif, 9 pt base, light grid, no top/right spines).
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.titlesize":9,"axes.labelsize":9,
  "xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":7.5,"axes.spines.top":False,"axes.spines.right":False,
  "axes.linewidth":.8,"axes.grid":True,"grid.alpha":.25,"grid.linewidth":.5,"legend.frameon":False,
  "lines.linewidth":1.6,"lines.markersize":4.5,"savefig.dpi":300,"savefig.bbox":"tight"})
W=3.45   # single-column width in inches
SC=["#2C3E50","#E74C3C","#F39C12","#9B59B6"]            # scenario colours
CC=["#E74C3C","#3498DB","#2ECC71","#F39C12","#9B59B6"]   # candidate colours
TC=["#C0392B","#2980B9","#27AE60","#E67E22","#8E44AD"]

ex1=["Blue","Red"]; ex2=["Cand. 0","Cand. 1","Cand. 2"]
# ---------------- data (percent) ----------------
ideal1={"theory":[50,50],"sim":[50.4620,49.5380]}
ideal2={"theory":[37.5,37.5,25.0],"sim":[38.1509,37.1186,24.7305]}
noise1=[("Ideal",[50.4620,49.5380],0),("$p$=5%",[46.35,47.19],6.46),("$p$=10%",[43.96,43.90],12.13),("$p$=20%",[38.61,38.37],23.02)]
noise2=[("Ideal",[38.1509,37.1186,24.7305],0),("$p$=5%",[37.90,35.45,24.73],1.92),("$p$=10%",[36.51,34.50,24.71],4.28),("$p$=20%",[36.21,32.27,23.93],7.60)]
dish1=[("Ideal",[50.4620,49.5380],0),("1 faulty (25%)",[32.53,67.15],0.32),("2 faulty (50%)",[50.46,48.59],0.96),("3 faulty (75%)",[0.00,97.97],2.03)]
dish2=[("Ideal",[38.1509,37.1186,24.7305],0),("1 faulty (12%)",[30.29,39.99,26.75],2.97),("2 faulty (25%)",[35.64,28.59,28.83],6.95),("4 faulty (50%)",[33.06,33.69,16.67],16.57)]
shots=[500,2000,5000,20000,100000]
conv1={"Blue":[53.36,49.60,50.46,49.79,50.26],"Red":[46.64,50.40,49.54,50.21,49.74]}; th1={"Blue":50,"Red":50}
conv2={"Cand. 0":[33.33,38.42,37.95,39.28,37.78],"Cand. 1":[37.91,36.62,35.41,35.70,37.31],"Cand. 2":[28.76,24.96,26.64,25.03,24.91]}
th2={"Cand. 0":37.5,"Cand. 1":37.5,"Cand. 2":25.0}

def ideal(d,names,path,title):
    fig,ax=plt.subplots(figsize=(W,2.5)); x=np.arange(len(names)); w=.36
    b1=ax.bar(x-w/2,d["theory"],w,color=SC[0],label="Statevector (exact)",edgecolor="white",lw=.6,zorder=3)
    b2=ax.bar(x+w/2,d["sim"],w,color=SC[1],label="Sampling (20,000 shots)",edgecolor="white",lw=.6,zorder=3)
    for bars,c in ((b1,SC[0]),(b2,SC[1])):
        for b in bars: ax.text(b.get_x()+b.get_width()/2,b.get_height()+.8,f"{b.get_height():.1f}",ha="center",va="bottom",fontsize=7,color=c)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("Probability (%)"); ax.set_ylim(0,max(max(d["theory"]),max(d["sim"]))*1.45)
    ax.grid(axis="x",visible=False); ax.legend(loc="upper right",fontsize=7); ax.set_title(title); fig.tight_layout(); fig.savefig(path); plt.close(fig)

def scen(rows,names,path,title,tall=3.2):
    fig,ax=plt.subplots(figsize=(W,tall)); x=np.arange(len(names)); n=len(rows); tw=.8; bw=tw/n
    for i,(lab,vals,sp) in enumerate(rows):
        off=(i-(n-1)/2)*bw
        l=lab+(f" (spoiled {sp:.1f}%)" if sp>0 else "")
        bars=ax.bar(x+off,vals,bw*.92,color=SC[i],label=l,edgecolor="white",lw=.5,zorder=3)
        for b in bars:
            h=b.get_height()
            if h>0.5: ax.text(b.get_x()+b.get_width()/2,h+.6,f"{h:.0f}",ha="center",va="bottom",fontsize=6.5,color=SC[i])
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("Probability (%)"); ax.grid(axis="x",visible=False)
    top=max(max(r[1]) for r in rows); ax.set_ylim(0,min(100,top*1.15)); ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.14),ncol=1,fontsize=7,handlelength=1.2)
    ax.set_title(title); fig.tight_layout(); fig.savefig(path); plt.close(fig)

def conv(c,th,names,path,title):
    fig,ax=plt.subplots(figsize=(W,2.7))
    for i,nm in enumerate(names):
        ax.axhline(th[nm],color=TC[i],ls="--",lw=.9,alpha=.6)
        ax.plot(shots,c[nm],marker="o",color=CC[i],label=f"{nm} (theory {th[nm]:.1f}%)",zorder=4)
    ax.set_xscale("log"); ax.set_xlabel("Number of shots"); ax.set_ylabel("Probability (%)")
    ax.legend(loc="best",fontsize=7); ax.set_title(title); fig.tight_layout(); fig.savefig(path); plt.close(fig)

ideal(ideal1,ex1,"ex1_s0_ideal.png","Example 1: exact vs sampled tally")
ideal(ideal2,ex2,"ex2_s0_ideal.png","Example 2: exact vs sampled tally")
scen(noise1,ex1,"ex1_s1_noise.png","Example 1: depolarizing noise")
scen(noise2,ex2,"ex2_s1_noise.png","Example 2: depolarizing noise")
scen(dish1,ex1,"ex1_s2_dishonest.png","Example 1: faulty ballots")
scen(dish2,ex2,"ex2_s2_dishonest.png","Example 2: faulty ballots")
conv(conv1,th1,ex1,"ex1_s3_convergence.png","Example 1: convergence with shots")
conv(conv2,th2,ex2,"ex2_s3_convergence.png","Example 2: convergence with shots")
print("done")
