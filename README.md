# Multivariate U(N)-QSP

Two oscillator experiments share one compiler. Given coefficients in an ordered signal basis, it computes a tensor-train (TT) decomposition, synthesizes a matrix QSP sequence for each core, then reloads the saved rotations and projectors to reconstruct the success block.

## Reproduce the examples

From the repository root, use Python 3.10 or later.

```bash
python -m pip install -r requirements.txt
python equation81/run.py
python three_oscillators/run.py
```

| Example | Default sweep | Results |
|---|---|---|
| Equation 81 | Four signals, $q_1,p_1,q_2,p_2$. Vary the signal cutoff $N=1,\ldots,6$. | [CSV](equation81/results/errors.csv) · [Accuracy and success plot](equation81/results/accuracy_and_success.png) |
| Three oscillators | Six signals, $q_1,p_1,q_2,p_2,q_3,p_3$. Fix $N=2$ and vary Taylor order $K=2,6,12$. | [CSV](three_oscillators/results/errors.csv) · [Fock-cutoff comparison](three_oscillators/results/cutoff_comparison.png) |

Each command prints its new `<example>/results/run_<id>/` directory. It contains `errors.csv`, the comparison plot, and a folder per case with `qsp_parameters.npz` and the time-ordered `schedule.json`. Use `--help` to see the settings for either experiment.

## Reading the comparisons

**Equation 81.** The code fits $U=\exp(i\pi F^2/32)$ on nine inputs with 0, 1 or 2 photons per mode, where $F=2(n_1+n_2)-i a_1^\dagger a_2+i a_2^\dagger a_1$. It compares the fit, the TT approximation and the scale-restored QSP block against the same matrix-exponential reference. Output outside the input space counts toward the operator error. At $N=6$, `circuit_error` is $1.27\times10^{-9}$ and the minimum success probability per attempt is $2.89\times10^{-7}$.

**Three oscillators.** A Taylor expansion of $\exp(-i\theta H)$ supplies the coefficients, with $H=a_1a_2a_3+a_1^\dagger a_2^\dagger a_3^\dagger$ and $\theta=0.1$. At Taylor order 12, gate-by-gate replay gives an operator error of $1.82\times10^{-13}$ against the four-level-per-mode reference. On inputs with 0 or 1 photon per mode, the error against a 12-level reference is $1.86\times10^{-2}$. The two curves show the effect of the Fock cutoff on the comparison.

The physical runs use one success flag per core and a shared TT bond. The $M$-core sequence uses $2MN$ signal queries per attempt. The Equation 81 script also checks shared-register composition at one set of scalar phases.

## Add an example

Place the target model and a `run.py` in the directory. For a coefficient tensor saved beside the script, call the shared compiler as follows.

```python
from pathlib import Path
import numpy as np
from core import compile_coefficients

here = Path(__file__).resolve().parent
C = np.load(here / "coefficients.npy", allow_pickle=False)
signals = [f"S{j + 1}" for j in range(C.ndim)]
circuit = compile_coefficients(C, signals, here / "results")
print(np.linalg.norm(circuit.recovered_coefficients - circuit.tt_coefficients))
```

Run `python -m my_example.run` from the repository root, using the new directory's name. Each axis of `C` lists powers $-N,\ldots,N$. Axis order fixes the written signal product, whose rightmost factor acts first. The snippet measures coefficient reconstruction error. To measure gate error, evaluate the reconstructed block on the example's physical inputs and compare its output with the target.