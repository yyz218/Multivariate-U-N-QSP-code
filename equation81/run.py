"""Reproduce the two-oscillator Eq. (81) benchmark with four ordered signals."""
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
from core.verification import shared_register_residual
from equation81 import model

SIGNALS = ('q1', 'p1', 'q2', 'p2')


def run_sweep(folder, cutoffs, input_cutoff, kappa):
    reference, info = model.reference_target(input_cutoff)
    print(f"Input dimension {info['input_dimension']}, reference sector dimension "
          f"{info['reference_sector_dimension']}")
    print(' N     fitted error       TT error      QSP error    min success    queries')
    rows = []
    for n in cutoffs:
        coefficients, fit = model.fit_coefficients(reference, n, kappa, input_cutoff)
        circuit = compile_coefficients(coefficients, SIGNALS, folder / f'N{n}')
        row = model.check_stages(reference, n, kappa, input_cutoff, fit['fit_output_levels'],
                                 coefficients, circuit.tt_coefficients,
                                 circuit.recovered_coefficients, circuit.normalization)
        row.update(input_cutoff=input_cutoff, queries=circuit.queries)
        rows.append(row)
        print(f"{n:2d}   {row['fit_error']:12.5e}   {row['tt_error']:12.5e}   "
              f"{row['circuit_error']:12.5e}   {row['success_min']:12.5e}   {circuit.queries:3d}")
        if n == cutoffs[0]:
            phases = np.exp(1j * np.array([0.21, 0.37, 0.58, 0.91]))
            residual = shared_register_residual(circuit.parameter_file, circuit.cores, phases)
            if residual >= 1e-9:
                raise ArithmeticError('Shared-register product failed its scalar-phase check.')
            print(f'     Shared-register check at one scalar phase per core: {residual:.3e}')
    return rows


def plot_results(rows, path):
    figure = Figure(figsize=(9, 3.6), layout='constrained')
    accuracy, success = figure.subplots(1, 2)
    n = [row['N'] for row in rows]
    accuracy.semilogy(n, [row['fit_error'] for row in rows], 'o-', color='#14645b',
                      markerfacecolor='none', markersize=8, label='Coefficient fit')
    accuracy.semilogy(n, [row['circuit_error'] for row in rows], 'x--', color='#ac511b',
                      label='Restored QSP block')
    accuracy.set(xlabel='Signal cutoff N', ylabel='Operator error', title='Accuracy on the input space')
    accuracy.legend()
    success.semilogy(n, [row['success_min'] for row in rows], 'o-', color='#344b80')
    success.set(xlabel='Signal cutoff N', ylabel='Worst-input probability', title='Success per attempt')
    for ax in (accuracy, success):
        ax.set_xticks(n)
        ax.grid(True, alpha=0.25)
    figure.savefig(path, dpi=150, facecolor='white')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cutoffs', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6],
                        metavar='N', help='signal cutoffs (default 1 through 6)')
    parser.add_argument('--input-cutoff', type=int, default=2, metavar='K',
                        help='largest input occupation per mode (default 2)')
    parser.add_argument('--kappa', type=float, default=1.0, help='signal spacing (default 1)')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('results'),
                        help='parent directory for the new run')
    args = parser.parse_args()
    if min(args.cutoffs) < 1 or args.input_cutoff < 0:
        parser.error('Signal cutoffs must be positive and the maximum occupation nonnegative.')
    if not np.isfinite(args.kappa) or args.kappa <= 0:
        parser.error('Signal spacing must be finite and positive.')
    with run_directory(args.output) as folder, threadpool_limits(limits=2):
        rows = run_sweep(folder, sorted(set(args.cutoffs)), args.input_cutoff, args.kappa)
        write_table(folder / 'errors.csv', rows)
        plot_results(rows, folder / 'accuracy_and_success.png')


if __name__ == '__main__':
    main()
