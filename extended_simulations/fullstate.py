import numpy as np
from qv import *
def init_state(S,N,n):
    st=np.zeros((2**n,2,S.D),complex)
    st[:N,:,:]=S.psi[None,None,:]/np.sqrt(2*N)
    return st
def vote_op(st,j,S,k): st[j,0,S.basis[k]]*=-1
def final_probs(st):
    a=(st[:,0,:]-st[:,1,:])/np.sqrt(2)      # control=1 after H
    return (np.abs(a)**2).sum(axis=0)       # over candidate basis index, summed over ID
def run_full(S,votes,n,gate_p=None,g=None,rng=None,idle_p=0.0,ro=0.0):
    """One trajectory. votes: list length N of candidate idx. gate_p: per-elementary-gate failure prob; g gates per voting op."""
    N=len(votes); nq=1+S.nc  # control + cand qubits
    st=init_state(S,N,n); size=2**n
    idbit=[1<<(n-1-m) for m in range(n)]
    jj=np.arange(size)
    nonid=n+nq   # error locations: n ID qubits, control, cand qubits
    for j,k in enumerate(votes):
        vote_op(st,j,S,k)
        if gate_p:
            ne=rng.binomial(g,gate_p)
            if idle_p: ne+=rng.binomial(nonid,idle_p)
            for _ in range(ne):
                loc=rng.integers(0,nonid); pa=rng.integers(0,3)   # 0=X,1=Y,2=Z
                if loc<n:                       # ID qubit: X/Y permute, Z harmless
                    if pa!=2: st=st[jj^idbit[loc]]
                elif loc==n:                    # control
                    if pa in (0,1): st=st[:,::-1,:].copy()
                    if pa in (1,2): st[:,1,:]*=-1
                else:                           # candidate qubit q
                    q=loc-n-1; bit=1<<(S.nc-1-q); xs=np.arange(S.D)
                    if pa in (0,1): st=st[:,:,xs^bit]
                    if pa in (1,2): st=st*(1-2*((xs&bit)>0))[None,None,:]
    return final_probs(st)   # unnormalised joint probs over candidate basis (control=1)
def summarize(S,post):
    acc=post.sum(); valid=np.array([post[b] for b in S.basis]); return acc,valid/acc,(post.sum()-valid.sum())/acc
if __name__=="__main__":
    rng=np.random.default_rng(1)
    # ideal check vs closed form, N=64, K=5, and vs branch-class
    for N,K in ((8,3),(64,5),(256,8)):
        S=general(K); n=int(np.ceil(np.log2(N))); votes=list(rng.integers(0,K,N)); counts=np.bincount(votes,minlength=K)
        acc,v,sp=summarize(S,run_full(S,votes,n))
        r=tally_distribution(S,list(counts))
        print(N,K,"max|full-n_k/N/K..|",np.abs(v-counts/N/K*K*1.0 /1).max() if False else np.abs(v*acc*K-counts/N).max(),
              "accept",round(acc,5),"1/K",1/K,"branch-class diff",np.abs(r['valid']-v).max(),"spoiled",sp)
