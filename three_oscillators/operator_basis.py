"""Ordered displacement coefficients and a real-operator QSP gate evaluator.

This is NOT Fourier integration of the commuting scalar function exp(-i h(q,p)).
It projects the operator Taylor polynomial onto products of finite-dimension
unitaries, preserving q1,p1,q2,p2,q3,p3 order throughout.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm, svd
from .model import local_operators


def local_basis(d, N, phase_scale, rcond=1e-12):
    _, q, p = local_operators(d)
    frequencies = np.arange(-N, N+1)
    uq = np.array([expm(1j * phase_scale * n * q) for n in frequencies])
    up = np.array([expm(1j * phase_scale * n * p) for n in frequencies])
    basis = np.array([x @ y for x in uq for y in up])
    matrix = basis.reshape(len(basis), d*d).T
    u, s, vh = svd(matrix, full_matrices=False)
    keep = s > rcond*s[0]
    inverse = (vh[keep].conj().T / s[keep]) @ u[:, keep].conj().T
    diagnostics = {
        'local_basis_rank': int(np.count_nonzero(keep)),
        'local_operator_space_dimension': d*d,
        'local_basis_complete': bool(np.count_nonzero(keep) == d*d),
        'local_basis_retained_condition_number': float(s[0]/s[keep][-1]),
        'local_basis_singular_values': s.tolist(),
        'local_basis_pseudoinverse_rcond': rcond,
        'phase_scale': phase_scale,
    }
    return matrix, inverse, diagnostics


def tensor_map(tensor, matrix, axis):
    out = np.tensordot(matrix, tensor, axes=(1, axis))
    return np.moveaxis(out, 0, axis)


def to_coefficients(operator, d, N, inverse):
    # U[i1,i2,i3,j1,j2,j3] -> U[(i1,j1),(i2,j2),(i3,j3)].
    t = operator.reshape((d,)*6).transpose(0,3,1,4,2,5).reshape((d*d,)*3)
    for axis in range(3):
        t = tensor_map(t, inverse, axis)
    return t.reshape((2*N+1,)*6)


def from_coefficients(coefficients, d, matrix):
    L = coefficients.shape[0]
    t = coefficients.reshape((L*L,)*3)
    for axis in range(3):
        t = tensor_map(t, matrix, axis)
    return t.reshape((d,)*6).transpose(0,2,4,1,3,5).reshape(d**3,d**3)


def apply_local(unitary, vectors, mode, d):
    # vectors[channel, system_output, input_column], system basis |n1,n2,n3>.
    t = vectors.reshape(vectors.shape[0], d, d, d, vectors.shape[-1])
    t = np.tensordot(unitary, t, axes=(1, mode+1))
    t = np.moveaxis(t, 0, mode+1)
    return t.reshape(vectors.shape)


def evaluate_saved_circuit(saved, d, phase_scale, inputs=None):
    """Propagate from saved R/Pi/signs only, using noncommuting physical signals.

    Each success ancilla is touched only in its own core. Contracting its 0->0
    block immediately equals final projection of all six ancillas. The bond
    is NOT projected between cores. Every padded bond path is retained.
    No original TT coefficients enter this implementation.
    """
    B = int(saved['bond_dimension'])
    M = int(saved['num_variables'])
    if M != 6:
        raise ValueError('This target requires six ordered signal cores')
    if inputs is None:
        inputs = np.eye(d**3, dtype=complex)
    a, q, p = local_operators(d)
    signals = [expm(1j*phase_scale*q), expm(1j*phase_scale*p)]
    signs = saved['signal_signs']
    bond = np.zeros((B, d**3, inputs.shape[1]), complex)
    bond[0] = inputs
    for m in range(M, 0, -1):
        R, Pi = saved[f'm{m}_R'], saved[f'm{m}_Pi']
        channel = np.concatenate((bond, np.zeros_like(bond)), axis=0)
        channel = (R[-1] @ channel.reshape(2*B, -1)).reshape(channel.shape)
        U = signals[(m-1)%2]
        for j in range(len(signs)-1, -1, -1):
            projected = (Pi[j] @ channel.reshape(2*B, -1)).reshape(channel.shape)
            if signs[j] == 1:
                channel = apply_local(U, projected, (m-1)//2, d) + channel-projected
            else:
                channel = projected + apply_local(U.conj().T, channel-projected, (m-1)//2, d)
            channel = (R[j] @ channel.reshape(2*B, -1)).reshape(channel.shape)
        bond = channel[:B].copy()
    raw_success = bond[0]
    return float(saved['normalization_product'])*raw_success, raw_success
