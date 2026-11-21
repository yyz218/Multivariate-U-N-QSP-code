#!/usr/bin/env python3
"""Ordered Fourier TT cores to numerical U(channel_dimension)-QSP parameters.

Spectral completion and degree reduction for the single-core compiler.

Convention (written matrix order, RIGHTMOST operation acts first):
    W_m(z) = R[0] G[0](z) R[1] ... G[d-1](z) R[d], d = 2*N.
    G[j](z) = z*Pi[j] + I-Pi[j]              if signs[j] = +1,
            = Pi[j] + z**(-1)*(I-Pi[j])      if signs[j] = -1.
There are N positive and N negative slots. Thus W_m is z**(-N)
times the ordinary-polynomial construction, with NO additional U**(-N)
query and NO assumption of access to U**(1/2).

The success block is the FULL leading B x B block, with B common to all
cores and a separate success qubit for each W_m. Its target is the
zero-padded D_m(z)/alpha_m, not just its one leading entry.

References: Lu, Liu & Lin, arXiv:2408.01439v4, Sec. 2.2, Theorem 4,
Lemma 5 and the selected-column construction after Eq. (19).
The spectral factor routine is a right-factor Wilson/Newton iteration,
with finite-polynomial coefficient residual checks; it is not a verbatim
implementation of the spectral-factor algorithm cited as Ref. [45] there.
"""
from __future__ import annotations


import numpy as np
from scipy.linalg import qr, svd


def adjoint(a):
    return a.conj().swapaxes(-1, -2)


def polynomial_values(coefficients, z):
    """Ordinary coefficients have layout (degree, output_row, input_column)."""
    return np.einsum('tl,lij->tij', z[:, None]**np.arange(len(coefficients)),
                     coefficients, optimize=True)


def positive_gram_coefficients(a):
    """Coefficients k>=0 of A(z)^dagger A(z) on |z|=1.

    The negative coefficients are the adjoints of the positive ones.
    This is finite convolution, independent of a Fourier sampling grid.
    """
    d = len(a)-1
    return np.array([sum(adjoint(a[j]) @ a[j+k]
                         for j in range(d-k+1)) for k in range(d+1)])


def spectral_factor_residual(p, q):
    residual = positive_gram_coefficients(p) + positive_gram_coefficients(q)
    residual[0] -= np.eye(p.shape[-1])
    coefficient_l2 = np.sqrt(np.linalg.norm(residual[0])**2
                              + 2*np.linalg.norm(residual[1:])**2)
    # By triangle inequality this bounds the matrix operator residual everywhere
    # on the unit circle in exact arithmetic. Its computed value is not an
    # interval-certified enclosure of floating-point roundoff.
    uniform_bound = np.linalg.norm(residual[0], ord=2)
    uniform_bound += 2*sum(np.linalg.norm(a, ord=2) for a in residual[1:])
    return float(coefficient_l2), float(uniform_bound)


def right_spectral_factor(p, *, tolerance=1e-12, initial_grid=64,
                          max_grid=4096, max_iterations=40):
    """Find polynomial Q, deg(Q)<=deg(P), with Q^dagger Q=I-P^dagger P.

    P may be rectangular. Only the active input-column dimension is used.
    For E=Q^{-dagger} S Q^{-1}-I, choose the analytic part H with half its
    constant coefficient. Then H+H^dagger=E and Q_new=(I+H)Q is a Newton
    correction. Frequencies are exp(+i*k*theta); fft(samples)/M extracts
    coefficients in this convention. At even-grid Nyquist take half.

    The iteration is evaluated on a grid, but acceptance uses the finite
    degree-d factor's convolution residual over all frequencies. A failed
    residual check increases the grid, rather than declaring grid values a
    polynomial factorization certificate. Failure at max_grid raises an error.
    """
    d, r = len(p)-1, p.shape[-1]
    grid = 1 << int(np.ceil(np.log2(max(initial_grid, 16*(d+1)))))
    if grid > max_grid:
        raise ValueError(f'max_grid={max_grid} is smaller than the required starting grid {grid}')
    attempts = []
    I = np.eye(r, dtype=complex)
    while grid <= max_grid:
        z = np.exp(2j*np.pi*np.arange(grid)/grid)
        pv = polynomial_values(p, z)
        spectrum = I-adjoint(pv) @ pv
        spectrum = (spectrum+adjoint(spectrum))/2
        q0 = adjoint(np.linalg.cholesky(np.mean(spectrum, axis=0)))
        q_samples = np.broadcast_to(q0, (grid, r, r)).copy()
        iteration_history = []
        stop_tolerance = max(50*np.finfo(float).eps*np.sqrt(r), tolerance*0.01)
        for iteration in range(1, max_iterations+1):
            inverse = np.linalg.inv(q_samples)
            error = adjoint(inverse) @ spectrum @ inverse-I
            error = (error+adjoint(error))/2
            grid_error = float(np.max(np.linalg.norm(error, axis=(1, 2))))
            iteration_history.append(grid_error)
            if not np.isfinite(grid_error):
                raise RuntimeError('Non-finite spectral factor iteration')
            if grid_error <= stop_tolerance:
                break
            error_coefficients = np.fft.fft(error, axis=0)/grid
            analytic = np.zeros_like(error_coefficients)
            analytic[0] = error_coefficients[0]/2
            analytic[1:grid//2] = error_coefficients[1:grid//2]
            analytic[grid//2] = error_coefficients[grid//2]/2
            correction = np.fft.ifft(analytic, axis=0)*grid
            q_samples = (I+correction) @ q_samples
        q_fft = np.fft.fft(q_samples, axis=0)/grid
        q = q_fft[:d+1].copy()
        l2, uniform_bound = spectral_factor_residual(p, q)
        attempts.append({
            'grid_size': grid, 'iterations': iteration,
            'iteration_grid_residuals': iteration_history,
            'discarded_fft_coefficient_norm': float(np.linalg.norm(q_fft[d+1:])),
            'finite_polynomial_gram_coefficient_l2_residual': l2,
            'finite_polynomial_gram_uniform_bound_estimate': uniform_bound,
        })
        if uniform_bound <= tolerance:
            return q, {'method': 'right_factor_Wilson_Newton',
                       'degree_cap': d, 'active_column_dimension': r,
                       'accepted_grid': grid, 'attempts': attempts}
        grid *= 2
    raise RuntimeError(f'Polynomial spectral factor failed: {attempts[-1]}')


def peel_isometry(f, *, rank_atol=1e-12, rank_rtol=1e-12):
    """Recursively factor selected columns, without constructing other columns.

    Use Pi=projector(range(F_0)^perp). Exact isometry gives F_0^dagger F_k=0,
    hence Pi F_k=F_k and Pi F_0=0. The strictly normalized outer-factor
    construction gives full-column-rank F_0. Its minimum singular value
    cannot decrease under the exact update, since
        Fnew_0^dagger Fnew_0 = F_0^dagger F_0 + F_1^dagger Pi F_1.
    This avoids dividing by tiny highest-degree singular values. It is an
    admissible (possibly larger-rank) projector in the same degree recursion.
    The degree-reduced coefficients are
        Fnew_l = (I-Pi) F_l + Pi F_{l+1},  l=0,...,k-1.
    Keep diagnostics for both discarded end terms. At the last step polar
    orthogonalization gives an exact numerical isometry; its adjustment is
    measured. Complete only this constant isometry to a full unitary.
    """
    working = f.copy()
    degree, channel, B = len(f)-1, f.shape[1], f.shape[2]
    bases, ranks, diagnostics = [], [], []
    for k in range(degree, 0, -1):
        constant_basis, singular_values, _ = svd(working[0], full_matrices=True)
        threshold = max(rank_atol, rank_rtol*float(singular_values[0]))
        constant_rank = int(np.count_nonzero(singular_values > threshold))
        if constant_rank != B or singular_values[-1] < 1e-10*singular_values[0]:
            raise RuntimeError('Constant coefficient is ill-conditioned; outer-factor assumption failed')
        rank = channel-B
        basis = np.column_stack([constant_basis[:, B:], constant_basis[:, :B]])
        projector = basis[:, :rank] @ adjoint(basis[:, :rank])
        lower_leak = np.linalg.norm(projector @ working[0])
        upper_leak = np.linalg.norm(working[-1]-projector @ working[-1])
        diagnostics.append({
            'degree_before_step': k, 'projector_rank': rank,
            'rank_threshold': threshold,
            'projector_choice': 'orthogonal_complement_of_constant_coefficient_range',
            'constant_coefficient_singular_values': singular_values.tolist(),
            'negative_degree_discarded_frobenius': float(lower_leak),
            'highest_degree_discarded_frobenius': float(upper_leak),
            'projector_hermiticity_residual': float(np.linalg.norm(projector-adjoint(projector))),
            'projector_idempotence_residual': float(np.linalg.norm(projector@projector-projector)),
        })
        bases.append(basis)
        ranks.append(rank)
        working = np.array([working[l]-projector@working[l]+projector@working[l+1]
                            for l in range(k)])
    constant = working[0]
    u, s, vh = svd(constant, full_matrices=False)
    constant_isometry = u @ vh
    basis_full, _ = qr(constant_isometry, mode='full')
    constant_unitary = np.column_stack([constant_isometry, basis_full[:, B:]])
    # bases are in written order (degree d, d-1, ..., 1).
    if degree:
        rotations = [bases[0]]
        rotations.extend(adjoint(bases[j-1])@bases[j] for j in range(1, degree))
        rotations.append(adjoint(bases[-1])@constant_unitary)
    else:
        rotations = [constant_unitary]
    rotations = np.array(rotations)
    projectors = np.zeros((degree, channel, channel), dtype=complex)
    for j, rank in enumerate(ranks):
        projectors[j, :rank, :rank] = np.eye(rank)
    return rotations, projectors, np.array(ranks, dtype=int), {
        'projector_strategy': 'stable_constant_range_complement',
        'steps_in_written_left_to_right_order': diagnostics,
        'constant_isometry_residual_before_polar': float(np.linalg.norm(adjoint(constant)@constant-np.eye(B))),
        'constant_polar_adjustment_frobenius': float(np.linalg.norm(constant_isometry-constant)),
        'constant_singular_values': s.tolist(),
        'constant_completion_unitarity_residual': float(np.linalg.norm(adjoint(constant_unitary)@constant_unitary-np.eye(channel))),
    }
