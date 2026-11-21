"""Compile coefficients in an ordered signal basis into a sequence of matrix QSP rotations."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.linalg import svd

from .synthesis import peel_isometry, right_spectral_factor
from .verification import restore_coefficients


@dataclass
class Compilation:
    """TT approximation and scale-restored reconstruction from the saved circuit."""

    cores: list[np.ndarray]
    tt_coefficients: np.ndarray
    recovered_coefficients: np.ndarray
    normalization: float
    parameter_file: Path
    queries: int


def compile_coefficients(coefficients, signals, output_dir):
    """Factor C, synthesize each core, then reload and contract the saved blocks.

    Axis j labels signal j in the written product. Each axis has powers -N,...,N.
    The case directory receives qsp_parameters.npz and schedule.json.
    """
    coefficients = np.asarray(coefficients, dtype=complex)
    signals = list(signals)
    if coefficients.ndim < 1 or coefficients.ndim != len(signals):
        raise ValueError("Supply one signal name per coefficient axis.")
    width = coefficients.shape[0]
    if width % 2 != 1 or any(length != width for length in coefficients.shape):
        raise ValueError("All coefficient axes must have the same odd length 2*N+1.")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("Coefficients must be finite.")
    if not all(isinstance(name, str) and name.strip() for name in signals):
        raise ValueError("Each signal needs a nonempty string name.")

    cores, _, _ = tt_svd(coefficients)
    tt_coefficients = reconstruct_coefficients(cores)
    parameter_file = synthesize(cores, signals, output_dir)
    recovered, normalization = restore_coefficients(parameter_file)
    return Compilation(cores, tt_coefficients, recovered, normalization,
                       parameter_file, len(signals) * (width - 1))


def tt_svd(coefficients, relative_tolerance=1e-12):
    """Sequential complex SVD; tail energy, not a per-singular-value cutoff.

    Each of d-1 local Frobenius residual budgets is
    relative_tolerance * ||C||_F / sqrt(d-1).
    The theoretical bound holds in exact arithmetic; measured residuals
    below also account for floating-point effects.
    """
    shape = coefficients.shape
    d = len(shape)
    norm = float(np.linalg.norm(coefficients))
    if d == 1:
        return [coefficients.reshape(1, shape[0], 1).copy()], [], []
    local_budget = relative_tolerance * norm / np.sqrt(d-1)
    cores, discarded_norms, singular_values = [], [], []
    remainder = coefficients.copy()
    left_rank = 1
    for k in range(d-1):
        matrix = remainder.reshape(left_rank * shape[k], -1)
        u, s, vh = svd(matrix, full_matrices=False)
        tails = np.r_[np.cumsum((s[::-1]**2))[::-1], 0.0]
        admissible = np.flatnonzero(tails[1:] <= local_budget**2)
        rank = int(admissible[0] + 1) if admissible.size else len(s)
        cores.append(u[:, :rank].reshape(left_rank, shape[k], rank))
        discarded_norms.append(float(np.sqrt(tails[rank])))
        singular_values.append(s)
        remainder = s[:rank, None] * vh[:rank, :]
        left_rank = rank
    cores.append(remainder.reshape(left_rank, shape[-1], 1))
    return cores, discarded_norms, singular_values


def reconstruct_coefficients(cores):
    value = cores[0][0]
    for core in cores[1:]:
        value = np.tensordot(value, core, axes=(-1, 0))
    return value[..., 0]


def synthesize(cores, signals, folder):
    """Export one independently flagged word per core. Axes follow written order."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if not cores or len(cores) != len(signals):
        raise ValueError("Supply one signal name per TT core.")
    if any(c.ndim != 3 or min(c.shape) < 1 or not np.all(np.isfinite(c)) for c in cores):
        raise ValueError("Each core needs finite (left, frequency, right) slices.")
    width = cores[0].shape[1]
    if width % 2 != 1 or any(c.shape[1] != width for c in cores):
        raise ValueError("Each frequency axis must cover the same range -N through N.")
    if (cores[0].shape[0] != 1 or cores[-1].shape[2] != 1
            or any(a.shape[2] != b.shape[0] for a, b in zip(cores, cores[1:]))):
        raise ValueError("Neighboring bonds must match and boundary bonds must be scalar.")
    N = width // 2
    largest = max(max(c.shape[0], c.shape[2]) for c in cores)
    B = 1 << (largest - 1).bit_length()
    signs = np.tile([1, -1], N)
    stored = dict(N=N, num_variables=len(cores), bond_dimension=B, signal_signs=signs)
    alphas = []
    for m, core in enumerate(cores, 1):
        left, _, right = core.shape
        alpha = 1.05 * sum(np.linalg.norm(core[:, k, :], 2) for k in range(width))
        if alpha == 0:
            alpha = 1.0
        alphas.append(alpha)
        target = core.transpose(1, 0, 2) / alpha
        complement, _ = right_spectral_factor(target, tolerance=1e-12, max_grid=8192)
        columns = np.zeros((width, 2 * B, B), dtype=complex)
        columns[:, :left, :right] = target
        columns[:, B:B+right, :right] = complement
        columns[0, B+right:, right:] = np.eye(B - right)
        rotations, projectors, _, _ = peel_isometry(columns)
        stored[f'm{m}_R'] = rotations
        stored[f'm{m}_Pi'] = projectors
    stored.update(alphas=np.array(alphas), normalization_product=np.prod(alphas))
    path = folder / 'qsp_parameters.npz'
    np.savez_compressed(path, **stored)
    steps = []
    for m in range(len(cores), 0, -1):
        steps.append(dict(core=m, rotation=2*N))
        for j in range(2*N-1, -1, -1):
            steps.append(dict(core=m, signal=signals[m-1], sign=int(signs[j]), projector_index=j))
            steps.append(dict(core=m, rotation=j))
    schedule = dict(time_order=steps, success_qubits=len(cores), bond_dimension=B,
                    queries=len(cores)*2*N, signal_axes=signals,
                    initial_and_final="All success flags zero, logical bond endpoint zero",
                    positive_query="Apply S on Pi, identity on its complement",
                    negative_query="Identity on Pi, apply S dagger on its complement")
    (folder / 'schedule.json').write_text(json.dumps(schedule, indent=2), encoding='utf-8')
    return path
