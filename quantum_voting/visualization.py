"""Publication-quality visualisation for Quantum Voting Protocol results.

Produces matplotlib figures suitable for inclusion in academic papers:
- Grouped bar charts (theoretical vs simulation)
- Multi-scenario bar charts
- Convergence plots (probability vs shot count)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Academic style defaults ─────────────────────────────────────────
COLORS = {
    "theoretical": "#2C3E50",
    "simulation":  "#E74C3C",
    "candidates":  ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"],
    "theory_line": ["#C0392B", "#2980B9", "#27AE60", "#E67E22", "#8E44AD"],
    "scenarios":   ["#2C3E50", "#E74C3C", "#F39C12", "#9B59B6", "#1ABC9C"],
}

def _apply_style():
    """Set matplotlib rcParams for publication figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_comparison(theoretical, simulation, candidate_names, *,
                    title="", save_path=None, shots_label="10 000"):
    """Grouped bar chart: theoretical (statevector) vs simulation (shots).

    Args:
        theoretical (dict): {name: prob}
        simulation  (dict): {name: prob}
        candidate_names (list[str]): Ordered list of candidate names.
        title (str): Figure title.
        save_path (str | None): If given, save figure to this path.
        shots_label (str): Label for the shot count shown in the legend.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    x = np.arange(len(candidate_names))
    width = 0.32

    theo_vals = [theoretical.get(n, 0) * 100 for n in candidate_names]
    sim_vals  = [simulation.get(n, 0) * 100  for n in candidate_names]

    bars1 = ax.bar(x - width / 2, theo_vals, width,
                   color=COLORS["theoretical"], label="Statevector (exact)",
                   edgecolor="white", linewidth=0.8, zorder=3)
    bars2 = ax.bar(x + width / 2, sim_vals, width,
                   color=COLORS["simulation"],
                   label=f"Sampling ({shots_label} shots)",
                   edgecolor="white", linewidth=0.8, zorder=3)

    # Value labels above bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                f"{h:.2f}%", ha="center", va="bottom", fontsize=9,
                color=COLORS["theoretical"], fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                f"{h:.2f}%", ha="center", va="bottom", fontsize=9,
                color=COLORS["simulation"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(candidate_names)
    ax.set_ylabel("Probability (%)")
    ax.set_title(title, pad=12)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(0, max(max(theo_vals), max(sim_vals)) * 1.25)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"  → Figure saved to {save_path}")
    return fig


def plot_scenarios(scenario_results, candidate_names, *,
                   title="", save_path=None):
    """Multi-scenario grouped bar chart.

    Args:
        scenario_results (list[tuple]): Each element is
            (label, probs_dict) where probs_dict = {name: prob}.
        candidate_names (list[str]): Ordered candidate names.
        title (str): Figure title.
        save_path (str | None): If given, save figure to this path.
    """
    _apply_style()
    n_scenarios = len(scenario_results)
    n_cands = len(candidate_names)

    fig, ax = plt.subplots(figsize=(max(8, n_scenarios * 2), 5))
    x = np.arange(n_cands)
    total_width = 0.75
    bar_width = total_width / n_scenarios

    for s_idx, (label, probs) in enumerate(scenario_results):
        vals = [probs.get(n, 0) * 100 for n in candidate_names]
        # Also add spoiled if present
        if "Spoiled" in probs:
            label += f" (spoiled: {probs['Spoiled']*100:.1f}%)"
        offset = (s_idx - (n_scenarios - 1) / 2) * bar_width
        color = COLORS["scenarios"][s_idx % len(COLORS["scenarios"])]
        bars = ax.bar(x + offset, vals, bar_width * 0.9,
                      color=color, label=label,
                      edgecolor="white", linewidth=0.6, zorder=3)
        for bar in bars:
            h = bar.get_height()
            if h > 0.5:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                        f"{h:.1f}%", ha="center", va="bottom",
                        fontsize=7, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(candidate_names)
    ax.set_ylabel("Probability (%)")
    ax.set_title(title, pad=12)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"  → Figure saved to {save_path}")
    return fig


def plot_convergence(shot_counts, histories, theoretical,
                     candidate_names, *, title="", save_path=None):
    """Line plot: probability convergence as shot count increases.

    Args:
        shot_counts (list[int]): x-axis values.
        histories (dict): {name: list[float]} — measured probs at each shot count.
        theoretical (dict): {name: float} — exact probabilities.
        candidate_names (list[str]): Ordered candidate names.
        title (str): Figure title.
        save_path (str | None): If given, save figure to this path.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, name in enumerate(candidate_names):
        c = COLORS["candidates"][idx % len(COLORS["candidates"])]
        ct = COLORS["theory_line"][idx % len(COLORS["theory_line"])]
        theo_pct = theoretical[name] * 100

        # Theoretical dashed line
        ax.axhline(y=theo_pct, color=ct, linestyle="--", alpha=0.5,
                   label=f"{name} theory ({theo_pct:.1f}%)")
        # Measured points
        vals = [h * 100 for h in histories[name]]
        ax.plot(shot_counts, vals, marker="o", color=c, linewidth=1.8,
                markersize=6, label=f"{name} measured", zorder=4)

    ax.set_xscale("log")
    ax.set_xlabel("Number of Shots (log scale)")
    ax.set_ylabel("Probability (%)")
    ax.set_title(title, pad=12)
    ax.legend(loc="best", framealpha=0.9, ncol=2)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"  → Figure saved to {save_path}")
    return fig
