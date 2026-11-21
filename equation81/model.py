"""Physical operator references and exact infinite-oscillator matrix elements.

K is the MAXIMUM INPUT occupation of EACH mode. It is not the Fourier cutoff.
Eq81 conserves total occupation. Building its generator on the complete sector
r+s<=2*K gives its exact action on inputs 0<=r,s<=K (up to matrix arithmetic).
No diagonalizing mixer, period-32 identity or scalar symbol is used in the fit.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import expm
from scipy.special import eval_genlaguerre, gammaln, logsumexp
from scipy.optimize import minimize_scalar


def sector_states(total):
    return [(r, t-r) for t in range(total+1) for r in range(t+1)]


def reference_target(K, gamma=np.pi/32, unitary_path=None):
    states=sector_states(2*K)
    index={s:j for j,s in enumerate(states)}
    dim=len(states)
    F=np.zeros((dim,dim),complex)
    for j,(r,s) in enumerate(states):
        F[j,j]=2*(r+s)
        if s: F[index[(r+1,s-1)],j]=-1j*np.sqrt((r+1)*s)
        if r: F[index[(r-1,s+1)],j]=1j*np.sqrt(r*(s+1))
    if unitary_path is None:
        U=expm(1j*gamma*(F@F))
        name='exp(i*gamma*F^2), Eq81 when gamma=pi/32'
    else:
        U=np.asarray(np.load(unitary_path,allow_pickle=False),dtype=complex)
        name='user supplied unitary on the declared invariant total-number sector'
        if U.shape!=(dim,dim):
            raise ValueError(f'Custom unitary must have shape {(dim,dim)}; use sector_states(2*K) order')
    residual=float(np.linalg.norm(U.conj().T@U-np.eye(dim),2))
    if residual>1e-9: raise ValueError('Reference target is not unitary on the declared sector')
    # Axis order: output mode 1, input mode 1, output mode 2, input mode 2.
    d=K+1; extent=2*K+1
    tensor=np.zeros((extent,d,extent,d),complex)
    for r in range(d):
        for s in range(d):
            for m,n in states: tensor[m,r,n,s]=U[index[(m,n)],index[(r,s)]]
    return tensor,dict(name=name,gamma=float(gamma),input_cutoff_per_mode=K,
        input_dimension=d*d,reference_invariant_sector_total_cutoff=2*K,
        reference_sector_dimension=dim,unitarity_residual=residual,
        scope='All input states in span{|r,s>:r,s<=K}; full output, not just compression into this box')


def displacement_columns(alpha,M,K):
    """<m|D(alpha)|n>, all m<M and n<=K; not exp of truncated q or p."""
    x=abs(alpha)**2
    out=np.zeros((M,K+1),complex)
    if x==0:
        out[np.arange(min(M,K+1)),np.arange(min(M,K+1))]=1
        return out
    for n in range(K+1):
        m=np.arange(n,M)
        scale=-.5*x+.5*(gammaln(n+1)-gammaln(m+1))+(m-n)*np.log(abs(alpha))
        out[m,n]=np.exp(scale+1j*(m-n)*np.angle(alpha))*eval_genlaguerre(n,m-n,x)
        low=np.arange(min(n,M))
        out[low,n]=(np.exp(-.5*x+.5*(gammaln(low+1)-gammaln(n+1)))
            *(-alpha.conjugate())**(n-low)*eval_genlaguerre(low,n-low,x))
    return out


def dictionary_columns(N,kappa,M,K):
    """B_uv=exp(i*kappa*u*q) exp(i*kappa*v*p), increasing u then v.
    BCH gives B_uv=exp(-i*kappa^2*u*v/2)D(kappa*(-v+i*u)/sqrt(2)).
    """
    ns=np.arange(-N,N+1)
    return np.array([np.exp(-.5j*kappa*kappa*u*v)*displacement_columns(
        kappa*(-v+1j*u)/np.sqrt(2),M,K) for u in ns for v in ns])


def target_columns(tensor,M):
    extent,d,_,_=tensor.shape
    if M<extent: raise ValueError('Output cutoff must include the reference support')
    out=np.zeros((M,d,M,d),complex)
    out[:extent,:,:extent,:]=tensor
    return out.transpose(0,2,1,3).reshape(M*M,d*d)


def realized_columns(c,columns):
    count,M,d=columns.shape
    pair_coefficients=c.reshape(count,count)
    out=np.empty((M*M,d*d),complex)
    for r in range(d):
        for s in range(d):
            out[:,r*d+s]=(columns[:,:,r].T@pair_coefficients@columns[:,:,s]).reshape(-1)
    return out


def tail_bound(N,kappa,K,M,coefficient_l1):
    """Analytic omitted-output bound, evaluated in ordinary floating point.

    For each displaced |n>, n<=K and x<=xmax, the positive generating function
    G_n(e^t)=exp(x*(e^t-1))*sum_j binom(n,j)e^{t(n-j)}[x*(e^t-1)^2]^j/j!
    gives Pr(nout>=M)<=exp(-t*M)G_K(e^t;xmax). Union bound for two
    output modes, Frobenius bound for (K+1)^2 input columns and l1 triangle
    bound for c yield ||omitted output|| <= ||c||_1 sqrt(2*(K+1)^2*beta).
    This is NOT an interval certificate of floating-point roundoff.
    """
    x=(kappa*N)**2
    if x==0: return dict(bound=0.,log_probability_bound=-float('inf'))
    def objective(t):
        terms=[gammaln(K+1)-2*gammaln(j+1)-gammaln(K-j+1)
               +(K-j)*t+j*(np.log(x)+2*np.log(np.expm1(t))) for j in range(K+1)]
        return float(x*np.expm1(t)-M*t+logsumexp(terms))
    result=minimize_scalar(objective,bounds=(1e-5,5.),method='bounded')
    log_beta=min(0.,objective(float(result.x)))
    bound=coefficient_l1*np.exp(.5*(np.log(2*(K+1)**2)+log_beta))
    return dict(bound=float(bound),log_probability_bound=log_beta,chernoff_t=float(result.x),
        maximum_displacement_squared=float(x),numeric_scope='analytic inequality, ordinary floating-point evaluation')


def evaluate_error(c,N,kappa,K,M,tensor,normalization=None):
    columns=dictionary_columns(N,kappa,M,K)
    actual=realized_columns(c,columns)
    target=target_columns(tensor,M)
    delta=actual-target
    err=float(np.linalg.norm(delta,2))
    tail=tail_bound(N,kappa,K,M,float(np.abs(c).sum()))
    report=dict(operator_error_retained_output=err,
        operator_error_plus_tail_estimate=float(np.hypot(err,tail['bound'])),
        rms_column_error=float(np.linalg.norm(delta)/(K+1)),
        output_levels_per_mode=M,omitted_output=tail)
    if normalization is not None:
        norm=float(normalization)
        gram=actual.conj().T@actual/(norm*norm)
        probs=np.linalg.eigvalsh((gram+gram.conj().T)/2)
        report.update(normalization_product=norm,
            raw_success_probability_min=float(max(0.,probs[0])),
            raw_success_probability_max=float(max(0.,probs[-1])),
            normalized_success_block_error=err/norm,
            normalized_success_probability_tail_uncertainty=(tail['bound']/norm)**2)
        # Norm of W|0> - |0>U, using unitarity of W and the success overlap.
        overlap=target.conj().T@actual/norm
        gram_delta=2*np.eye((K+1)**2)-overlap-overlap.conj().T
        report['raw_full_input_isometry_error']=float(np.sqrt(max(0.,np.linalg.eigvalsh(gram_delta)[-1])))
    return report


def fit_coefficients(tensor,N,kappa,K,rcond=1e-10):
    d=K+1; extent=tensor.shape[0]; x=(kappa*N)**2
    M=max(64,extent,int(np.ceil(x+12*np.sqrt((2*K+1)*(x+1))+8*K+32)))
    attempts=[]
    for attempt in range(4):
        columns=dictionary_columns(N,kappa,M,K)
        # One-mode frame: rows=(output level,input level), columns=(nq,np).
        frame=columns.reshape(columns.shape[0],M*d).T
        u,s,vh=np.linalg.svd(frame,full_matrices=False)
        keep=s>rcond*s[0]
        inverse=(vh[keep].conj().T/s[keep])@u[:,keep].conj().T
        # Y has compact exact support; no need to materialize its many zeros.
        inverse_small=inverse[:,:extent*d]
        pair=inverse_small@tensor.reshape(extent*d,extent*d)@inverse_small.T
        c=pair.reshape((2*N+1,)*4)
        tail=tail_bound(N,kappa,K,M,float(np.abs(c).sum()))
        attempts.append(dict(output_levels=M,coefficient_l1=float(np.abs(c).sum()),tail=tail))
        if tail['bound']<1e-12: break
        M=int(1.5*M)+32
    else: raise RuntimeError('Could not control the output tail during coefficient fit')
    return c,dict(method='separable_SVD_regularized_operator_least_squares',
        equation='Y approximately frame @ C_pair @ frame.T (transpose, not adjoint)',
        objective='Hilbert-Schmidt norm of (A_N-U) on the complete input box; not scalar Fourier L2',
        fit_output_levels=M,frame_shape=list(frame.shape),frame_retained_rank=int(keep.sum()),
        singular_value_cutoff_relative=rcond,frame_retained_condition=float(s[0]/s[keep][-1]),
        coefficient_l1=float(np.abs(c).sum()),coefficient_frobenius=float(np.linalg.norm(c)),
        attempts=attempts,qualification='Small singular directions are dropped separately in each one-mode frame; operator norm is checked afterwards')


def check_stages(tensor, N, kappa, input_cutoff, levels, fitted, tt, restored, normalization):
    """Use progressively larger output boxes; keep the omitted-output bound separate."""
    base = evaluate_error(fitted, N, kappa, input_cutoff, levels, tensor)
    after_tt = evaluate_error(tt, N, kappa, input_cutoff, levels+48, tensor)
    final = evaluate_error(restored, N, kappa, input_cutoff, levels+48, tensor, normalization)
    larger = evaluate_error(restored, N, kappa, input_cutoff, levels+80, tensor)
    key = 'operator_error_retained_output'
    shift = abs(larger[key] - final[key])
    tail = final['omitted_output']['bound']
    coeff_diff = float(np.linalg.norm(restored - tt))
    if shift > max(1e-9, 5*tail):
        raise ArithmeticError('Output-cutoff refinement did not stabilize.')
    if np.abs(restored - tt).sum() > max(1e-7, 1e-10*np.abs(fitted).sum()):
        raise ArithmeticError('Saved-word reconstruction failed its coefficient check.')
    return dict(N=N, kappa=kappa, fit_error=base[key], tt_error=after_tt[key],
                circuit_error=final[key], output_tail_bound=tail, cutoff_shift=shift,
                synthesis_coefficient_residual=coeff_diff, normalization=normalization,
                success_min=final['raw_success_probability_min'], queries=8*N)
