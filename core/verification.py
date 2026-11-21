"""Expand saved matrix-QSP words and contract all bond paths."""
import numpy as np


def reconstruct_laurent_coefficients(rotations, projectors, signs):
    """Exact algebraic expansion in floating point; no spectral FFT involved.

    Expand right to left using explicit matrix projectors, not their ranks.
    Return integer powers and full channel-matrix coefficients.
    """
    dimension = rotations.shape[1]
    coefficients = {0: rotations[-1].copy()}
    I = np.eye(dimension)
    for j in range(len(signs)-1, -1, -1):
        Pi = projectors[j]
        if signs[j] == 1:
            terms = [(1, Pi), (0, I-Pi)]
        elif signs[j] == -1:
            terms = [(0, Pi), (-1, I-Pi)]
        else:
            raise ValueError('Signals must be +1 or -1')
        updated = {}
        for old_power, coefficient in coefficients.items():
            for shift, multiplier in terms:
                power = old_power+shift
                contribution = rotations[j] @ (multiplier @ coefficient)
                if power not in updated:
                    updated[power] = contribution
                else:
                    updated[power] += contribution
        coefficients = updated
    powers = np.array(sorted(coefficients), dtype=int)
    return powers, np.array([coefficients[n] for n in powers])


def evaluate_parameter_product(rotations, projectors, signs, z):
    """Independent, LEFT-to-RIGHT product at a single unit-modulus input."""
    I = np.eye(rotations.shape[1])
    result = rotations[0].copy()
    for j, sign in enumerate(signs):
        Pi = projectors[j]
        gate = z*Pi+I-Pi if sign == 1 else Pi+(I-Pi)/z
        result = result @ gate @ rotations[j+1]
    return result


def compose_coefficient_chain(coefficient_blocks):
    """Contract all bond indices, keeping all frequency axes and padded paths."""
    tensor = coefficient_blocks[0][:, 0, :]
    for coefficient in coefficient_blocks[1:]:
        tensor = np.tensordot(tensor, coefficient, axes=(-1, 1))
    return tensor[..., 0]


def restore_coefficients(path):
    """Read the emitted words and contract their scale-restored, padded success blocks."""
    blocks = []
    with np.load(path, allow_pickle=False) as saved:
        B = int(saved['bond_dimension'])
        for m, alpha in enumerate(saved['alphas'], 1):
            powers, word = reconstruct_laurent_coefficients(
                saved[f'm{m}_R'], saved[f'm{m}_Pi'], saved['signal_signs'])
            N = int(saved['N'])
            if not np.array_equal(powers, np.arange(-N, N+1)):
                raise ValueError("Unexpected Laurent power order in the saved word.")
            blocks.append(alpha * word[:, :B, :B])
        normalization = float(saved['normalization_product'])
    return compose_coefficient_chain(blocks), normalization


def shared_register_residual(parameter_file, cores, phases):
    """Compare the cyclic shared-register product and the normalized TT at one point."""
    count = len(cores)
    if len(phases) != count:
        raise ValueError('Supply one phase per core.')
    with np.load(parameter_file, allow_pickle=False) as saved:
        bond = int(saved['bond_dimension'])
        cutoff = int(saved['N'])
        permutation = [0, count, *range(1, count)]
        shift = np.kron(np.eye(count + 1)[:, permutation], np.eye(bond))
        product = np.eye((count + 1) * bond, dtype=complex)
        logical = np.ones((1, 1), dtype=complex)
        for m, (core, z) in enumerate(zip(cores, phases), 1):
            word = evaluate_parameter_product(saved[f'm{m}_R'], saved[f'm{m}_Pi'],
                                              saved['signal_signs'], z)
            extended = np.eye((count + 1) * bond, dtype=complex)
            extended[:2 * bond, :2 * bond] = word
            product = product @ (shift @ extended)
            powers = z ** np.arange(-cutoff, cutoff + 1)
            core_value = np.einsum('lnr,n->lr', core, powers)
            logical = logical @ (core_value / saved['alphas'][m - 1])
    return float(abs(product[0, 0] - logical.item()))
