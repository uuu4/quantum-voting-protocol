# Quantum Voting Protocol

**Paper:** Ali Emre, A., & Ammar, D. (2025). *Quantum Voting Protocol for Centralized and Distributed Voting Based on Phase-Flip Counting.*

- **arXiv:** [2510.15243](https://arxiv.org/abs/2510.15243)
- **Zenodo:** [10.5281/zenodo.17373814](https://doi.org/10.5281/zenodo.17373814)

## Overview

This repository provides the numerical experiment code for the quantum voting protocol described in the paper. The protocol encodes votes as **phase-flips** on entangled candidate states and uses a **control qubit** with Hadamard interference to extract vote tallies via post-selection.

### Key Results

| Example | Voters | Candidates | Expected Probabilities |
|---------|--------|------------|----------------------|
| 1 (§V-A) | 4 | 2 (Blue/Red) | P(Blue) = P(Red) = 50.0% |
| 2 (§V-B) | 8 | 3 | P(C0) = P(C1) = 37.5%, P(C2) = 25.0% |

