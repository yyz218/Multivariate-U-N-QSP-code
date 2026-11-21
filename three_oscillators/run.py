"""Compile three-photon evolution and compare it at two Fock cutoffs."""
import argparse
import sys
from pathlib import Path

from matplotlib.figure import Figure
import numpy as np
from threadpoolctl import threadpool_limits

if __package__ in (None, ''):
    # Make the shared package available when launching this file directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import compile_coefficients
from core.results import run_directory, write_table
from three_oscillators import model, operator_basis

SIGNALS = ('q1', 'p1', 'q2', 'p2', 'q3', 'p3')


def run_sweep(folder, orders, cutoff, d, theta, kappa, low, reference_dims):
    hamiltonian = model.hamiltonian(d)
    reference, diagnostics = model.independent_reference(hamiltonian, theta)
    checks, high_reference = model.cutoff_reference(d, theta, low, reference_dims, reference)
    frame, inverse, info = operator_basis.local_basis(d, cutoff, kappa)
    low_inputs = model.low_input_indices(d, low)
    high_d = reference_dims[-1]

    print(f"Reference dimension {d**3}, matrix exponential vs. spectral reference "
          f"{diagnostics['expm_vs_eigendecomposition_operator_norm']:.3e}")
    print(f"Local dictionary rank {info['local_basis_rank']} of {d**2}")
    if len(checks) > 1:
        change = checks[-1]['reference_refinement_low_input_operator_norm']
        print(f'Low-input reference difference at d={reference_dims[-2]} and {high_d}: {change:.3e}')
    print(' K     finite error     larger-cutoff error     replay residual   vacuum success    queries')
    rows = []
    for order in orders:
        polynomial, bound = model.taylor_evolution(hamiltonian, theta, order)
        coefficients = operator_basis.to_coefficients(polynomial, d, cutoff, inverse)
        circuit = compile_coefficients(coefficients, SIGNALS, folder / f'N{cutoff}_K{order}')

        from_coefficients = operator_basis.from_coefficients(circuit.recovered_coefficients, d, frame)
        with np.load(circuit.parameter_file, allow_pickle=False) as saved:
            restored, raw = operator_basis.evaluate_saved_circuit(saved, d, kappa)
        replay_error = float(np.linalg.norm(restored - from_coefficients, 2))
        if replay_error > 1e-8:
            raise ArithmeticError('Gate-by-gate replay and coefficient reconstruction disagree.')
        low_output = model.embed_output(restored[:, low_inputs], d, high_d)
        row = dict(K=order, N=cutoff, d=d, theta=theta, kappa=kappa, low_levels=low,
                   reference_d=high_d, taylor_error=float(np.linalg.norm(polynomial - reference, 2)),
                   taylor_bound=bound, finite_gate_error=float(np.linalg.norm(restored - reference, 2)),
                   low_input_large_cutoff_error=float(np.linalg.norm(low_output - high_reference, 2)),
                   replay_residual=replay_error, vacuum_success=float(np.linalg.norm(raw[:, 0])**2),
                   normalization=circuit.normalization, queries=circuit.queries)
        rows.append(row)
        print(f"{order:2d}   {row['finite_gate_error']:12.5e}      "
              f"{row['low_input_large_cutoff_error']:12.5e}        {replay_error:12.5e}    "
              f"{row['vacuum_success']:12.5e}    {circuit.queries:3d}")
    return rows


def plot_results(rows, path):
    figure = Figure(figsize=(7, 3.8), layout='constrained')
    ax = figure.subplots()
    orders = [row['K'] for row in rows]
    ax.semilogy(orders, [row['finite_gate_error'] for row in rows], 'o-', color='#14645b',
                label=f"All inputs, d = {rows[0]['d']}")
    ax.semilogy(orders, [row['low_input_large_cutoff_error'] for row in rows], 's--',
                color='#ac511b', label=f"Low inputs vs. d = {rows[0]['reference_d']}")
    ax.set(xlabel='Classical Taylor order K', ylabel='Restored operator error',
           title='Three-oscillator evolution', xticks=orders)
    ax.grid(True, alpha=0.25)
    ax.legend()
    figure.savefig(path, dpi=150, facecolor='white')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orders', type=int, nargs='+', default=[2, 6, 12], metavar='K',
                        help='classical Taylor orders (default 2, 6, 12)')
    parser.add_argument('--cutoff', type=int, default=2, metavar='N', help='signal cutoff (default 2)')
    parser.add_argument('--fock-dim', type=int, default=4, metavar='D', help='levels per mode (default 4)')
    parser.add_argument('--theta', type=float, default=0.1, help='evolution parameter (default 0.1)')
    parser.add_argument('--kappa', type=float, default=1.2, help='signal spacing (default 1.2)')
    parser.add_argument('--low', type=int, default=2, metavar='L',
                        help='number of low input levels per mode (default 2)')
    parser.add_argument('--reference-dims', type=int, nargs='+', default=[8, 12], metavar='D',
                        help='increasing reference dimensions above the model cutoff (default 8, 12)')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('results'),
                        help='parent directory for the new run')
    args = parser.parse_args()
    if args.cutoff < 1 or min(args.orders) < 0:
        parser.error('The signal cutoff must be positive and Taylor orders nonnegative.')
    if args.fock_dim < 2 or not 1 <= args.low <= args.fock_dim:
        parser.error('Choose at least two Fock levels and low input levels between 1 and the Fock dimension.')
    if args.reference_dims != sorted(set(args.reference_dims)) or args.reference_dims[0] <= args.fock_dim:
        parser.error('Reference dimensions must increase and exceed the model Fock dimension.')
    if not np.isfinite(args.theta) or not np.isfinite(args.kappa) or args.kappa <= 0:
        parser.error('Theta must be finite and signal spacing finite and positive.')
    with run_directory(args.output) as folder, threadpool_limits(limits=2):
        rows = run_sweep(folder, sorted(set(args.orders)), args.cutoff, args.fock_dim,
                         args.theta, args.kappa, args.low, args.reference_dims)
        write_table(folder / 'errors.csv', rows)
        plot_results(rows, folder / 'cutoff_comparison.png')


if __name__ == '__main__':
    main()
