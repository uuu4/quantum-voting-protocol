"""Quantum Voting Protocol — Phase-Flip Counting on Entangled Candidate States.

Reference:
    Ali Emre, A., & Ammar, D. (2025). Quantum Voting Protocol for Centralized
    and Distributed Voting Based on Phase-Flip Counting. arXiv:2510.15243
"""

from .protocol import QuantumVotingProtocol
from .experiments import example1_4voters_2candidates, example2_8voters_3candidates

__version__ = "1.0.0"
__all__ = [
    "QuantumVotingProtocol",
    "example1_4voters_2candidates",
    "example2_8voters_3candidates",
]
