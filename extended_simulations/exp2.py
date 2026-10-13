import numpy as np, json
from qv import *
from fullstate import run_full
rng=np.random.default_rng(77)
def counts_for(N,K,D):
    dN=round(D*N); nr=round((N-dN)/K); lead=N-(K-1)*nr; return [lead]+[nr]*(K-1)
out={}
# C': memory noise, leader placed on the LAST candidate label (conservative: amplitude damping drives to |0..0> = candidate 0)
C=[]; S=general(3)
for N in (4,16,64,256,1024,4096):
    c=counts_for(N,3,0.1)[::-1]; p=np.array(c)/N     # leader is index 2
    for kind,val in (("dephase",1e-4),("dephase",1e-3),("damp",1e-4),("damp",1e-3)):
        kw=dict(q_dephase=val) if kind=="dephase" else dict(g_damp=val)
        r=tally_distribution(S,c,mem_steps=N,cand_only=False,**kw); q=np.real(r['valid'])
        lead=2; rho=(q[lead]-np.delete(q,lead).max())/(p[lead]-np.delete(p,lead).max())
        C.append(dict(N=N,kind=kind,val=val,rho=float(rho),accept=float(np.real(r['accept'])),correct=bool(q.argmax()==lead),tv=float(0.5*np.abs(q/q.sum()-p).sum())))
out['C']=C
# E': add N=128, with retained-margin metric rho and CIs; leader first (gate errors are label-symmetric on average)
Etr=[]; K=3
for N in (128,):
    n=int(np.ceil(np.log2(N))); g=12*n+6
    votes=sum(([k]*c for k,c in enumerate(counts_for(N,K,0.1))),[]); rng.shuffle(votes); p=np.bincount(votes,minlength=K)/N
    for p2 in (1e-4,3e-4,1e-3,3e-3):
        T=300; posts=np.array([run_full(S,votes,n,gate_p=p2,g=g,rng=rng) for _ in range(T)])
        def metr(P):
            m=P.mean(0); acc=m.sum(); v=np.array([m[b] for b in S.basis])/acc; sp=(m.sum()-sum(m[b] for b in S.basis))/acc; return acc,v,sp
        acc,v,sp=metr(posts); tv=0.5*np.abs(v/v.sum()-p).sum(); bs=[]
        for _ in range(200):
            idx=rng.integers(0,T,T); a,vv,s=metr(posts[idx]); bs.append((0.5*np.abs(vv/vv.sum()-p).sum(),s))
        bs=np.array(bs)
        Etr.append(dict(N=N,p2=p2,g=g,errors_per_run=float(g*p2*N),spoiled=float(sp),spoiled_ci=[float(x) for x in np.quantile(bs[:,1],[.025,.975])],
           tv=float(tv),tv_ci=[float(x) for x in np.quantile(bs[:,0],[.025,.975])],rho=float((v[0]-v[1:].max())/(p[0]-p[1:].max())),accept=float(acc),correct=bool(v.argmax()==0)))
    print("E",N,flush=True)
out['E128']=Etr
json.dump(out,open('results2.json','w'),indent=1)
for c in C: print(c)
for e in Etr: print(e)
