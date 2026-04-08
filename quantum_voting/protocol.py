"""Core Quantum Voting Protocol implementation.

Implements the centralized voting model (Sections II–IV) of:
    Ali Emre, A., & Ammar, D. (2025). arXiv:2510.15243

Stages:
    1. Initialization — voter-ID superposition, candidate entangled state, control |+⟩
    2. Voting — controlled phase-flips conditioned on (voter ID, control=0)
    3. Interference — Hadamard on control qubit
    4. Tallying — post-select control=|1⟩, measure candidate register

Extended with optional noise injection and dishonest voter simulation.
"""

import pennylane as qml
import numpy as np

from .operators import phase_flip, malicious_gate


class QuantumVotingProtocol:
    """Parametric quantum voting circuit builder and runner.

    Args:
        n_voters: Number of voters N.
        n_candidates: Number of candidates K.
        candidate_names: Human-readable names, length K.
        candidate_basis_map: {candidate_idx: [b0, b1]} — basis state flipped.
        candidate_state: Amplitude vector for the candidate register.
        vote_distribution: List of length N; vote_distribution[i] = candidate idx.
        noise_level: Depolarizing channel probability on candidate qubits (0 = off).
        dishonest_voters: Set of voter indices that apply malicious operations.
    """

    def __init__(self, n_voters, n_candidates, candidate_names,
                 candidate_basis_map, candidate_state, vote_distribution,
                 noise_level=0.0, dishonest_voters=None):
        self.n_voters = n_voters
        self.n_candidates = n_candidates
        self.candidate_names = list(candidate_names)
        self.candidate_basis_map = candidate_basis_map
        self.candidate_state = np.array(candidate_state, dtype=complex)
        self.vote_distribution = list(vote_distribution)
        self.noise_level = noise_level
        self.dishonest_voters = set(dishonest_voters or [])

        # Qubit counts
        self.n_id_qubits = int(np.ceil(np.log2(n_voters)))
        self.n_cand_qubits = int(np.log2(len(candidate_state)))

        # Wire layout: [ID … | Control | Candidate …]
        self.id_wires = list(range(self.n_id_qubits))
        self.control_wire = self.n_id_qubits
        self.cand_wires = list(range(self.n_id_qubits + 1,
                                     self.n_id_qubits + 1 + self.n_cand_qubits))
        self.total_qubits = self.n_id_qubits + 1 + self.n_cand_qubits

        # Inverse map: basis tuple → candidate index
        self.basis_to_candidate = {
            tuple(bits): idx for idx, bits in candidate_basis_map.items()
        }

        # Vote counts per candidate (honest voters only for theoretical)
        self._vote_counts = {}
        for i, v in enumerate(vote_distribution):
            if i not in self.dishonest_voters:
                self._vote_counts[v] = self._vote_counts.get(v, 0) + 1

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    @property
    def theoretical_probabilities(self):
        """Exact probabilities: votes_k / N for each candidate (ideal case)."""
        return {
            self.candidate_names[k]: self._vote_counts.get(k, 0) / self.n_voters
            for k in range(self.n_candidates)
        }

    @property
    def scenario_label(self):
        """Short label describing the current scenario configuration."""
        parts = []
        if self.noise_level > 0:
            parts.append(f"noise={self.noise_level:.0%}")
        if self.dishonest_voters:
            parts.append(f"dishonest={sorted(self.dishonest_voters)}")
        return ", ".join(parts) if parts else "ideal"

    def info(self):
        """Return a human-readable summary of the protocol configuration."""
        parts = [
            f"Voters       : {self.n_voters} ({self.n_id_qubits} ID qubits)",
            f"Candidates   : {self.n_candidates} ({self.n_cand_qubits} cand. qubits)",
            f"Total qubits : {self.total_qubits}",
            f"Wires        : ID={self.id_wires}, Ctrl={self.control_wire}, "
            f"Cand={self.cand_wires}",
            f"Votes        : {self._format_votes()}",
            f"Scenario     : {self.scenario_label}",
        ]
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Circuit
    # ------------------------------------------------------------------
    def _build_circuit(self):
        """Construct the full voting circuit (to be called inside a QNode)."""
        # 1. Candidate register
        qml.StatePrep(self.candidate_state, wires=self.cand_wires)

        # 2. Voter-ID superposition
        for w in self.id_wires:
            qml.Hadamard(wires=w)

        # 3. Control qubit in |+⟩
        qml.Hadamard(wires=self.control_wire)

        # 4. Voting: controlled phase-flips
        n_id = self.n_id_qubits
        ctrl_wires = self.id_wires + [self.control_wire]

        for voter_idx in range(2 ** n_id):
            if voter_idx >= self.n_voters:
                continue  # unused ID slot
            voter_id_bits = [(voter_idx >> (n_id - 1 - k)) & 1
                             for k in range(n_id)]
            ctrl_vals = voter_id_bits + [0]

            if voter_idx in self.dishonest_voters:
                # Dishonest voter applies arbitrary malicious gate
                qml.ctrl(
                    malicious_gate,
                    control=ctrl_wires,
                    control_values=ctrl_vals,
                )(wires=self.cand_wires)
            else:
                # Honest voter applies correct phase-flip
                cand_idx = self.vote_distribution[voter_idx]
                target_bits = self.candidate_basis_map[cand_idx]
                qml.ctrl(
                    phase_flip,
                    control=ctrl_wires,
                    control_values=ctrl_vals,
                )(wires=self.cand_wires, target_bits=target_bits)

        # 5. Noise injection (depolarizing channel on candidate qubits)
        if self.noise_level > 0:
            for w in self.cand_wires:
                qml.DepolarizingChannel(self.noise_level, wires=w)

        # 6. Interference
        qml.Hadamard(wires=self.control_wire)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run_statevector(self):
        """Exact simulation (statevector). Only works when noise_level == 0.

        Returns:
            probs (dict): {candidate_name: probability}
            post_prob (float): Total probability mass in control=|1⟩.
        """
        if self.noise_level > 0:
            raise ValueError(
                "Statevector simulation is not available with noise. "
                "Use run_shots() with default.mixed backend instead."
            )
        dev = qml.device("default.qubit", wires=self.total_qubits)

        @qml.qnode(dev)
        def circuit():
            self._build_circuit()
            return qml.state()

        sv = circuit()
        return self._post_select_statevector(sv)

    def run_shots(self, shots=10000):
        """Shot-based simulation.

        Uses default.mixed when noise is present, default.qubit otherwise.

        Returns:
            probs (dict): {candidate_name: probability}
            valid_shots (int): Shots where control=|1⟩.
            total_shots (int): Total shots requested.
        """
        backend = "default.mixed" if self.noise_level > 0 else "default.qubit"
        dev = qml.device(backend, wires=self.total_qubits, shots=shots)

        @qml.qnode(dev)
        def circuit():
            self._build_circuit()
            return qml.counts(wires=[self.control_wire] + self.cand_wires)

        results = circuit()
        return self._post_select_shots(results, shots)

    # ------------------------------------------------------------------
    # Post-selection helpers
    # ------------------------------------------------------------------
    def _post_select_statevector(self, statevector):
        n = self.total_qubits
        candidate_probs = {}

        for i in range(2 ** n):
            control_val = (i >> (n - 1 - self.control_wire)) & 1
            if control_val != 1:
                continue
            cand_bits = tuple(
                (i >> (n - 1 - w)) & 1 for w in self.cand_wires
            )
            prob = np.abs(statevector[i]) ** 2
            candidate_probs[cand_bits] = candidate_probs.get(cand_bits, 0.0) + prob

        total_prob = sum(candidate_probs.values())
        result = {}
        for cand_idx in range(self.n_candidates):
            bits = tuple(self.candidate_basis_map[cand_idx])
            p = candidate_probs.get(bits, 0.0)
            result[self.candidate_names[cand_idx]] = p / total_prob if total_prob > 0 else 0.0

        # Check for spoiled (non-candidate) basis states
        valid_bits = {tuple(self.candidate_basis_map[k]) for k in range(self.n_candidates)}
        spoiled_prob = sum(p for bits, p in candidate_probs.items() if bits not in valid_bits)
        if spoiled_prob > 0 and total_prob > 0:
            result["Spoiled"] = spoiled_prob / total_prob

        return result, total_prob

    def _post_select_shots(self, counts, total_shots):
        n_measured = 1 + self.n_cand_qubits
        candidate_counts = {}
        valid_shots = 0
        spoiled = 0

        for bitstring, count in counts.items():
            if isinstance(bitstring, (int, np.integer)):
                bits = [(int(bitstring) >> (n_measured - 1 - k)) & 1
                        for k in range(n_measured)]
            elif isinstance(bitstring, str):
                bits = [int(b) for b in bitstring]
            else:
                bits = list(bitstring)

            control_val = bits[0]
            cand_bits = tuple(bits[1:])

            if control_val == 1:
                valid_shots += count
                if cand_bits in self.basis_to_candidate:
                    cand_idx = self.basis_to_candidate[cand_bits]
                    name = self.candidate_names[cand_idx]
                    candidate_counts[name] = candidate_counts.get(name, 0) + count
                else:
                    spoiled += count

        result = {}
        for name in self.candidate_names:
            c = candidate_counts.get(name, 0)
            result[name] = c / valid_shots if valid_shots > 0 else 0.0
        if spoiled > 0:
            result["Spoiled"] = spoiled / valid_shots

        return result, valid_shots, total_shots

    # ------------------------------------------------------------------
    def _format_votes(self):
        parts = []
        for i, v in enumerate(self.vote_distribution):
            tag = " ✗" if i in self.dishonest_voters else ""
            parts.append(f"V{i}→{self.candidate_names[v]}{tag}")
        return ", ".join(parts)
