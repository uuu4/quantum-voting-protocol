"""Independent NumPy simulator for the phase-flip voting protocol (branch-class method).
After voting, the tally depends on the ID register only through the per-voter pure states
|chi_k> on (control, candidate); noise channels acting on control/candidate commute with the
trace over the ID register, so the post-selected statistics are sum_k (n_k/N) * E(|chi_k><chi_k|).
"""
import numpy as np, itertools
I2=np.eye(2); X=np.array([[0,1],[1,0]],complex); Y=np.array([[0,-1j],[1j,0]]); Zm=np.diag([1.,-1]).astype(complex)
H=np.array([[1,1],[1,-1]],complex)/np.sqrt(2)

def kron(*a):
    out=np.array([[1]],complex)
    for x in a: out=np.kron(out,x)
    return out

class Setup:
    def __init__(self,psi,basis,names=None):
        self.psi=np.array(psi,complex); self.D=len(psi); self.nc=int(np.log2(self.D))
        self.basis=list(basis); self.K=len(basis); self.names=names or [f"c{k}" for k in range(self.K)]

def example1(): return Setup(np.array([1,0,0,1])/np.sqrt(2),[3,0],["Blue","Red"])
def example2(): return Setup(np.array([1,1,1,0])/np.sqrt(3),[0,1,2])
def general(K):
    if K==2: return example1()
    nc=int(np.ceil(np.log2(K))); D=2**nc
    psi=np.zeros(D); psi[:K]=1; return Setup(psi/np.sqrt(K),list(range(K)))

def chi(S,k):
    z=S.psi.copy(); z[S.basis[k]]*=-1
    v=np.concatenate([z,S.psi])/np.sqrt(2)      # control (qubit 0) then candidate
    return v

def op_on(S,idx,U):
    """single-qubit U on qubit idx of (control, cand qubits...), qubit 0 = control"""
    n=1+S.nc; mats=[I2]*n; mats[idx]=U; return kron(*mats)

def apply_kraus(rho,Ks): return sum(K@rho@K.conj().T for K in Ks)

def depol_kraus(p): # PennyLane DepolarizingChannel convention
    return [np.sqrt(1-p)*I2]+[np.sqrt(p/3)*P for P in (X,Y,Zm)]
def dephase_kraus(q): return [np.sqrt(1-q)*I2,np.sqrt(q)*Zm]
def ampdamp_kraus(g): return [np.array([[1,0],[0,np.sqrt(1-g)]],complex),np.array([[0,np.sqrt(g)],[0,0]],complex)]
def chan_on(S,idx,Ks): return [op_on(S,idx,K) for K in Ks]

def tally_distribution(S,counts,p_depol=0.0,q_dephase=0.0,g_damp=0.0,p_hgate=0.0,p_ro=0.0,mem_steps=0,cand_only=True):
    """Return dict: accept prob, per-candidate prob among accepted, spoiled prob (all among accepted).
    p_depol: depolarizing on each candidate qubit after voting (paper's model).
    q_dephase,g_damp: per-voting-step memory errors; composed over mem_steps=N steps on candidate qubits and control.
    p_hgate: depolarizing after final Hadamard on control.  p_ro: readout bit-flip on each measured bit."""
    N=sum(counts); n=1+S.nc
    Hc=op_on(S,0,H)
    probs=np.zeros(2**n)
    for k,nk in enumerate(counts):
        if nk==0: continue
        v=chi(S,k); rho=np.outer(v,v.conj())
        if mem_steps:
            qd=0.5*(1-(1-2*q_dephase)**mem_steps); gd=1-(1-g_damp)**mem_steps
            for idx in range(0 if not cand_only else 1,n):
                if q_dephase: rho=apply_kraus(rho,chan_on(S,idx,dephase_kraus(qd)))
                if g_damp: rho=apply_kraus(rho,chan_on(S,idx,ampdamp_kraus(gd)))
        if p_depol:
            for idx in range(1,n): rho=apply_kraus(rho,chan_on(S,idx,depol_kraus(p_depol)))
        rho=Hc@rho@Hc.conj().T
        if p_hgate: rho=apply_kraus(rho,chan_on(S,0,depol_kraus(p_hgate)))
        probs+= nk/N*np.real(np.diag(rho))
    if p_ro:
        # classical readout flips on each of n measured bits
        M=np.array([[1-p_ro,p_ro],[p_ro,1-p_ro]]); Mfull=kron(*([M]*n)); probs=Mfull@probs
    acc=probs[2**S.nc:].sum()      # control=1
    post=probs[2**S.nc:]
    valid=np.array([post[b] for b in S.basis])
    spoiled=post.sum()-valid.sum()
    return dict(accept=acc,valid=valid/acc,spoiled=spoiled/acc)
if __name__=="__main__":
    S=example1()
    for p in (0,.05,.1,.2):
        r=tally_distribution(S,[2,2],p_depol=p); print("Ex1 p",p,np.round(r['valid'],4),round(r['spoiled'],4),round(r['accept'],4))
    S=example2()
    for p in (0,.05,.1,.2):
        r=tally_distribution(S,[3,3,2],p_depol=p); print("Ex2 p",p,np.round(r['valid'],4),round(r['spoiled'],4),round(r['accept'],4))
