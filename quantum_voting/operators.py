"""Phase-flip gate operators for the Quantum Voting Protocol.

Implements the controlled phase-flip encoding described in Section II-C of:
    Ali Emre, A., & Ammar, D. (2025). arXiv:2510.15243

The key idea: to flip the phase of a specific computational basis state |b_0 b_1⟩,
we apply X gates to map that state to |11⟩, apply CZ (which gives -1 to |11⟩),
then undo the X gates.
"""

import pennylane as qml
import numpy as np


def phase_flip(wires, target_bits):
    """Apply a phase-flip (-1) to a specific computational basis state.

    Implements: X-mask → CZ → X-unmask

    Args:
        wires (list[int]): Wire indices of the candidate register (exactly 2).
        target_bits (list[int]): Binary label of the target state,
            e.g. [0, 0] for |00⟩, [1, 0] for |10⟩.
    """
    # Step 1: X gates on qubits where target bit is 0
    for i, bit in enumerate(target_bits):
        if bit == 0:
            qml.PauliX(wires=wires[i])

    # Step 2: CZ flips the phase of |11⟩
    qml.CZ(wires=wires)

    # Step 3: undo X gates
    for i, bit in enumerate(target_bits):
        if bit == 0:
            qml.PauliX(wires=wires[i])


def malicious_gate(wires):
    """Dishonest voter gate — arbitrary rotations instead of a valid phase-flip.

    Used to simulate a tampering / manipulation attempt.
    """
    qml.RY(np.pi / 3, wires=wires[0])
    qml.RY(np.pi / 4, wires=wires[1])
