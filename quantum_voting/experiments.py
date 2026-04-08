"""Pre-defined experiment configurations matching the paper's examples.

Example 1 — Section V-A: 4 Voters, 2 Candidates (Blue vs Red)
Example 2 — Section V-B: 8 Voters, 3 Candidates
"""

import numpy as np

from .protocol import QuantumVotingProtocol


def example1_4voters_2candidates():
    """Paper Example 1: 4 Voters, 2 Candidates (Blue vs Red).

    Candidate state: Bell state (|00⟩ + |11⟩) / √2
    Vote distribution: V0→Blue, V1→Red, V2→Blue, V3→Red
    Expected: P(Blue) = 50%, P(Red) = 50%

    Blue vote → flips |11⟩ component  →  after tallying P(|11⟩) = Blue_votes/N
    Red  vote → flips |00⟩ component  →  after tallying P(|00⟩) = Red_votes/N
    """
    return QuantumVotingProtocol(
        n_voters=4,
        n_candidates=2,
        candidate_names=["Blue", "Red"],
        candidate_basis_map={
            0: [1, 1],  # Blue → phase-flip |11⟩
            1: [0, 0],  # Red  → phase-flip |00⟩
        },
        candidate_state=np.array([1, 0, 0, 1]) / np.sqrt(2),  # Bell state
        vote_distribution=[0, 1, 0, 1],  # Blue, Red, Blue, Red
    )


def example2_8voters_3candidates():
    """Paper Example 2: 8 Voters, 3 Candidates.

    Candidate state: W-like state (|00⟩ + |01⟩ + |10⟩) / √3
    Voters 0,3,5 → Candidate 0  (flip |00⟩)
    Voters 1,4,6 → Candidate 1  (flip |01⟩)
    Voters 2,7   → Candidate 2  (flip |10⟩)
    Expected: P(C0) = 3/8 = 37.5%, P(C1) = 3/8 = 37.5%, P(C2) = 2/8 = 25.0%
    """
    return QuantumVotingProtocol(
        n_voters=8,
        n_candidates=3,
        candidate_names=["Candidate 0", "Candidate 1", "Candidate 2"],
        candidate_basis_map={
            0: [0, 0],  # Candidate 0 → phase-flip |00⟩
            1: [0, 1],  # Candidate 1 → phase-flip |01⟩
            2: [1, 0],  # Candidate 2 → phase-flip |10⟩
        },
        candidate_state=np.array([1, 1, 1, 0]) / np.sqrt(3),  # W-like state
        vote_distribution=[0, 1, 2, 0, 1, 0, 1, 2],
    )
