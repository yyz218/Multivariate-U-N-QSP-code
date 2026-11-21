"""Coherent three-photon Hamiltonian, arXiv:2510.05405 Eq. (2), Kerr=0.

Dimensionless target: U(theta)=exp[-i theta (aaa+a^dagger a^dagger a^dagger)].
Every mode has Fock levels 0,...,d-1. All claims using these matrices are
finite-dimensional; increasing d is a separate model-convergence check.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.linalg import expm, eigh
from scipy import sparse
from scipy.sparse.linalg import expm_multiply


def local_operators(d):
    if d < 2:
        raise ValueError('fock_dim must be at least 2')
    a = np.diag(np.sqrt(np.arange(1, d)), 1).astype(complex)
    q = (a + a.conj().T) / np.sqrt(2)
    p = (a - a.conj().T) / (1j * np.sqrt(2))
    return a, q, p


def kron3(a, b, c):
    return np.kron(np.kron(a, b), c)


def hamiltonian(d):
    a, _, _ = local_operators(d)
    A = kron3(a, a, a)
    return A + A.conj().T


def sparse_hamiltonian(d):
    a = sparse.diags(np.sqrt(np.arange(1, d)), 1, shape=(d, d), format='csr')
    A = sparse.kron(sparse.kron(a, a, format='csr'), a, format='csr')
    return A + A.T


def taylor_evolution(h, theta, degree):
    """Compiler input from H powers only; never reads the reference expm.

    For Hermitian H and real theta, integral Taylor remainder gives
    ||exp(-i theta H)-T_K||_2 <= (|theta| ||H||_2)^(K+1)/(K+1)!.
    This is an exact-arithmetic bound for the chosen truncated H, not a
    floating-point enclosure or a bound on the infinite oscillator model.
    """
    if degree < 0:
        raise ValueError('Taylor degree must be nonnegative')
    term = np.eye(h.shape[0], dtype=complex)
    total = term.copy()
    for k in range(1, degree + 1):
        term = (-1j * theta / k) * (h @ term)
        total += term
    x = abs(theta) * float(np.linalg.norm(h, 2))
    bound = 0.0 if x == 0 else math.exp((degree + 1) * math.log(x) - math.lgamma(degree + 2))
    if not np.all(np.isfinite(total)):
        raise RuntimeError('Taylor expansion overflowed; reduce theta')
    return total, bound


def independent_reference(h, theta):
    """Two different numerical reference constructions; neither fits coefficients."""
    reference = expm(-1j * theta * h)
    vals, vecs = eigh(h)
    spectral = (vecs * np.exp(-1j * theta * vals)) @ vecs.conj().T
    diagnostics = {
        'expm_vs_eigendecomposition_operator_norm': float(np.linalg.norm(reference - spectral, 2)),
        'reference_unitarity_operator_norm': float(np.linalg.norm(reference.conj().T @ reference - np.eye(len(h)), 2)),
    }
    return reference, diagnostics


def low_input_indices(d, low):
    return np.array([i*d*d+j*d+k for i in range(low) for j in range(low)
                     for k in range(low)], dtype=int)


def embed_output(x, old_d, new_d):
    out = np.zeros((new_d**3, x.shape[1]), complex)
    idx = low_input_indices(new_d, old_d)
    out[idx] = x
    return out


def cutoff_reference(d, theta, low, ref_dims, reference):
    """Compare embedded output vectors, including amplitude outside the small box.

    These are empirical cutoff differences, not a certified infinite-space
    error. The high-dimensional reference is used only in validation.
    """
    base = reference[:, low_input_indices(d, low)]
    history = []
    previous, previous_d = None, None
    largest = None
    for rd in ref_dims:
        initial = np.zeros((rd**3, low**3), complex)
        initial[low_input_indices(rd, low), np.arange(low**3)] = 1
        high = expm_multiply(-1j * theta * sparse_hamiltonian(rd), initial, traceA=0.0)
        diff = embed_output(base, d, rd) - high
        outside = high.copy()
        outside[low_input_indices(rd, d)] = 0
        rec = {
            'reference_fock_dim': rd,
            'low_input_operator_norm_difference': float(np.linalg.norm(diff, 2)),
            'vacuum_vector_difference': float(np.linalg.norm(diff[:, 0])),
            'vacuum_probability_outside_small_box': float(np.vdot(outside[:, 0], outside[:, 0]).real),
            'high_reference_norm_residual': float(np.linalg.norm(high.conj().T @ high - np.eye(low**3), 2)),
        }
        if previous is not None:
            refinement = embed_output(previous, previous_d, rd) - high
            rec['reference_refinement_low_input_operator_norm'] = float(np.linalg.norm(refinement, 2))
            rec['reference_refinement_vacuum_vector_norm'] = float(np.linalg.norm(refinement[:, 0]))
        history.append(rec)
        previous, previous_d, largest = high, rd, high
    return history, largest
