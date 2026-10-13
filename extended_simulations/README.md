# Extended simulations (revision, Section 9.5)

NumPy/SciPy simulator for the phase-flip counting protocol.

- `qv.py`: exact simulator (identity/control/candidate registers; depolarizing, dephasing, amplitude-damping, Hadamard-gate and readout noise; branch-class method for noise after voting).
- `fullstate.py`: full-state Pauli-trajectory simulation of gate errors during voting (N up to 128).
- `exp.py`, `exp2.py`: experiments (sampling bounds, adaptive stopping, noise margins). Results: `results.json`, `results2.json`.
- `figs.py`: produces `noise_margin.png` and `sampling_check.png`.

Run with `python3 exp.py`, `python3 exp2.py`, `python3 figs.py` (requires numpy, scipy, matplotlib). Seeds are fixed in the scripts.

`regenerate_example_figures.py` re-plots the eight example figures of Section 9 (single-column size, same style as the new figures) from the numbers printed in the notebook outputs of the repository, so the plotted results are unchanged. (The notebook itself needs PennyLane.)
