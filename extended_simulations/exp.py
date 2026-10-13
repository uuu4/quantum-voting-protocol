import numpy as np, json, math, time, sys
from scipy.stats import beta
from qv import *
from fullstate import run_full, summarize
rng=np.random.default_rng(2026)
def cp(k,n,a=0.05):
    lo=0 if k==0 else beta.ppf(a/2,k,n-k+1); hi=1 if k==n else beta.ppf(1-a/2,k+1,n-k); return lo,hi
def counts_for(N,K,D):
    dN=round(D*N); nr=round((N-dN)/K); lead=N-(K-1)*nr; return [lead]+[nr]*(K-1)
res={}
# ---------- A: finite-sample winner identification (sufficient bound check, with CIs)
A=[]; delta=0.05; Dl=0.1
for N in (100,1000):
    for K in (2,5,10):
        c=counts_for(N,K,Dl); p=np.array(c)/N; gap=p[0]-p[1]
        Rh=math.ceil(2/gap**2*math.log(max(K-1,1)/delta)); s=2*(p[0]+p[1])
        Rb=math.ceil((s+2/3*(1+gap)*gap)/gap**2*math.log(max(K-1,1)/delta))
        T=3000; out={}
        for name,R in (("hoeffding",Rh),("bernstein",Rb)):
            X=rng.multinomial(R,p,size=T); wrong=int(np.sum(X[:,1:].max(1)>=X[:,0])); out[name]=(R,wrong,cp(wrong,T))
        # executions to get Rh acceptances: negative binomial (K per accept)
        Texec=rng.negative_binomial(Rh,1/K,size=T)+Rh
        # adaptive anytime rule
        R0=50; stops=[]; wr=0
        for _ in range(600):
            cnt=np.zeros(K); R=0; t=0; done=False
            while True:
                t+=1; Rt=R0*2**(t-1); cnt+=rng.multinomial(Rt-R,p); R=Rt
                eta=math.sqrt(2/Rt*math.log(K*(K-1)*t*(t+1)/delta))
                lead=int(cnt.argmax()); gaps=(cnt[lead]-np.delete(cnt,lead))/Rt
                if np.all(gaps>eta): wr+= (lead!=0); stops.append(Rt); break
                if Rt>2e6: break
        A.append(dict(N=N,K=K,gap=gap,delta=delta,R_hoeff=out['hoeffding'][0],err_hoeff=out['hoeffding'][1]/T,ci_hoeff=out['hoeffding'][2],
             R_bern=out['bernstein'][0],err_bern=out['bernstein'][1]/T,ci_bern=out['bernstein'][2],
             exec_mean=float(Texec.mean()),exec_pred=K*Rh,exec_q99=float(np.quantile(Texec,.99)),exec_bound=2*K*Rh,
             adaptive_mean_R=float(np.mean(stops)),adaptive_err=wr/len(stops),adaptive_n=len(stops),adaptive_wrong=wr))
res['A']=A; print("A done",flush=True)
# exact tally check
E=[]
for N,K in ((10,3),(20,3),(20,5)):
    c=counts_for(N,K,0.2); p=np.array(c)/N; R=math.ceil(2*N*N*math.log(2*K/delta)); T=3000
    X=rng.multinomial(R,p,size=T); ok=np.all(np.rint(X/R*N)==np.array(c)[None,:],axis=1); wrong=int((~ok).sum())
    E.append(dict(N=N,K=K,R=R,err=wrong/T,ci=cp(wrong,T)))
res['A_exact']=E
# ---------- B: depolarizing after voting (paper's model), N-independent
B=[]
for K in (2,3,5,8):
    S=general(K)
    for N in (100,):
        c=counts_for(N,K,0.1); p=np.array(c)/N
        for pd in (0,.01,.02,.05,.1,.2):
            r=tally_distribution(S,c,p_depol=pd); q=r['valid']; qn=q/q.sum()
            B.append(dict(K=K,p=pd,spoiled=float(r['spoiled']),gap_ratio=float((q[0]-q[1:].max())/ (1/K*(p[0]-p[1:].max()))), 
                          tv=float(0.5*np.abs(qn-p).sum()),correct=bool(q.argmax()==0),accept=float(r['accept'])))
res['B']=B; print("B done",flush=True)
# ---------- C: memory noise (dephasing / amplitude damping per voting step on control+candidate), scaling with N
C=[]
S=general(3)
for N in (4,16,64,256,1024,4096):
    c=counts_for(N,3,0.1); p=np.array(c)/N
    for kind,val in (("dephase",1e-4),("dephase",1e-3),("damp",1e-4),("damp",1e-3)):
        kw=dict(q_dephase=val) if kind=="dephase" else dict(g_damp=val)
        r=tally_distribution(S,c,mem_steps=N,cand_only=False,**kw); q=r['valid']
        C.append(dict(N=N,kind=kind,val=val,spoiled=float(r['spoiled']),gap_ratio=float((q[0]-q[1:].max())*3/(p[0]-p[1:].max())),
                      accept=float(r['accept']),correct=bool(q.argmax()==0),tv=float(0.5*np.abs(q/q.sum()-p).sum())))
res['C']=C; print("C done",flush=True)
# ---------- D: readout error
D=[]
S=general(3); c=counts_for(100,3,0.1); p=np.array(c)/100
for ro in (0,.005,.01,.02,.05):
    r=tally_distribution(S,c,p_ro=ro); q=r['valid']
    D.append(dict(ro=ro,spoiled=float(r['spoiled']),gap_ratio=float((q[0]-q[1:].max())*3/(p[0]-p[1:].max())),accept=float(r['accept']),correct=bool(q.argmax()==0)))
res['D']=D
# ---------- E: gate/idle Pauli noise during voting (trajectories), full state
Etr=[]; S=general(3); K=3
for N in (8,16,32,64):
    n=int(np.ceil(np.log2(N))); g=12*n+6
    votes=sum(([k]*c for k,c in enumerate(counts_for(N,K,0.1))),[]); rng.shuffle(votes); p=np.bincount(votes,minlength=K)/N
    for p2 in (1e-4,3e-4,1e-3,3e-3):
        T=300; posts=np.array([run_full(S,votes,n,gate_p=p2,g=g,rng=rng) for _ in range(T)])
        def metr(P):
            m=P.mean(0); acc=m.sum(); v=np.array([m[b] for b in S.basis])/acc; sp=(m.sum()-sum(m[b] for b in S.basis))/acc
            return acc,v,sp
        acc,v,sp=metr(posts); tv=0.5*np.abs(v/v.sum()-p).sum()
        bs=[]; 
        for _ in range(200):
            idx=rng.integers(0,T,T); a,vv,s=metr(posts[idx]); bs.append((0.5*np.abs(vv/vv.sum()-p).sum(),s,(vv[0]-vv[1:].max())*K/(p[0]-p[1:].max())))
        bs=np.array(bs)
        gr=(v[0]-v[1:].max())*K/(p[0]-p[1:].max())
        Etr.append(dict(N=N,p2=p2,g=g,errors_per_run=float(g*p2*N),spoiled=float(sp),spoiled_ci=[float(x) for x in np.quantile(bs[:,1],[.025,.975])],
            tv=float(tv),tv_ci=[float(x) for x in np.quantile(bs[:,0],[.025,.975])],gap_ratio=float(gr),gap_ci=[float(x) for x in np.quantile(bs[:,2],[.025,.975])],
            accept=float(acc),correct=bool(v.argmax()==0)))
    print("E N",N,flush=True)
res['E']=Etr
json.dump(res,open('results.json','w'),indent=1)
print("saved")
