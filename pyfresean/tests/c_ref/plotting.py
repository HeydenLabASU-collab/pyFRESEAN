"""Optional plots for pyfresean vs FRESEAN COARSE (C) c_ref comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from pyfresean.analysis.fresean import FRESEAN
from pyfresean.postprocess.plotting import plot_spectra
from pyfresean.tests.c_ref.io import load_c_eigenvalues_dat


def _c_eigenvalues_vdos_normalized(
    c_eval: np.ndarray,
    n_corr: int,
    win_time_0: float,
    n_dof: int,
) -> np.ndarray:
    avg_temp = (
        np.sum(c_eval[0])
        + 2 * np.sum(c_eval[1:])
    ) / (2 * n_corr - 1) / win_time_0 / (8.3145 * 0.1) / n_dof
    vdos_norm = n_corr * win_time_0 * (8.3145 * 0.1 * avg_temp)
    if vdos_norm <= 0:
        return c_eval
    return c_eval / vdos_norm


def load_c_eigenvalues_vdos_normalized(analysis: FRESEAN, eval_dat: Path | str) -> np.ndarray:
    """C eigenvalues with the same VDOS normalization as pyfresean results."""
    return _c_eigenvalues_vdos_normalized(
        load_c_eigenvalues_dat(eval_dat),
        analysis.n_corr,
        analysis.results.win_time[0],
        analysis._n_dof,
    )


def plot_eigenvalues_vs_c(
    analysis: FRESEAN,
    eval_dat: Path | str,
    output_path: Path | str,
    freq_indices: Sequence[int] = (0, 1),
    n_modes: int = 12,
    dpi: int = 150,
) -> Path:
    """
    Save a figure comparing leading eigenvalues from pyfresean and C at selected frequencies.

    One subplot per frequency: mode index vs eigenvalue, both series overlaid.
    Format is inferred from ``output_path`` suffix (default PNG if unknown).
    """
    import matplotlib.pyplot as plt

    eval_path = Path(eval_dat)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    c_eval = load_c_eigenvalues_vdos_normalized(analysis, eval_path)
    py_eval = analysis.results.eigenvalues
    freqs = analysis.results.freqs

    n_freq = len(freq_indices)
    fig, axs = plt.subplots(
        n_freq,
        1,
        figsize=(6, 1.4 * n_freq),
        sharex=True,
        squeeze=False,
    )

    for row, freq_index in enumerate(freq_indices):
        ax = axs[row, 0]
        n_plot = min(n_modes, py_eval.shape[1], c_eval.shape[1])
        mode_numbers = np.arange(1, n_plot + 1)
        ax.plot(
            mode_numbers,
            py_eval[freq_index, :n_plot],
            "o-",
            color="C0",
            label="pyfresean",
            markersize=4,
        )
        ax.plot(
            mode_numbers,
            c_eval[freq_index, :n_plot],
            "s--",
            color="C1",
            label="FRESEAN COARSE (C)",
            markersize=4,
        )
        freq_cm1 = freqs[freq_index]
        ax.set_ylabel(f"λ @ {freq_cm1:.2f} cm$^{-1}$")
        ax.legend(loc="best", fontsize=8)

    axs[-1, 0].set_xlabel("eigenvalue number (mode index)")
    fig.tight_layout()
    fmt = output_file.suffix.lstrip(".").lower() or "png"
    fig.savefig(output_file, format=fmt, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_file


def plot_eigenvalues_vs_c_separate_images(
    analysis: FRESEAN,
    eval_dat: Path | str,
    output_py_path: Path | str,
    output_c_path: Path | str,
    freq_indices: Sequence[int] = (0, 1),
    n_modes: int = 12,
    dpi: int = 150,
) -> tuple[Path, Path]:
    """Save two figures: pyfresean eigenvalues and C eigenvalues (same layout each)."""
    import matplotlib.pyplot as plt

    eval_path = Path(eval_dat)
    py_path = Path(output_py_path)
    c_path = Path(output_c_path)
    py_path.parent.mkdir(parents=True, exist_ok=True)
    c_path.parent.mkdir(parents=True, exist_ok=True)

    c_eval = load_c_eigenvalues_vdos_normalized(analysis, eval_path)
    py_eval = analysis.results.eigenvalues
    freqs = analysis.results.freqs

    def _save_single(values: np.ndarray, label: str, path: Path) -> None:
        n_freq = len(freq_indices)
        fig, axs = plt.subplots(
            n_freq,
            1,
            figsize=(6, 1.4 * n_freq),
            sharex=True,
            squeeze=False,
        )
        for row, freq_index in enumerate(freq_indices):
            ax = axs[row, 0]
            n_plot = min(n_modes, values.shape[1])
            mode_numbers = np.arange(1, n_plot + 1)
            ax.plot(mode_numbers, values[freq_index, :n_plot], "o-", markersize=4)
            ax.set_ylabel(f"λ @ {freqs[freq_index]:.2f} cm$^{-1}$")
            ax.set_title(label, fontsize=9)
        axs[-1, 0].set_xlabel("eigenvalue number (mode index)")
        fig.tight_layout()
        fmt = path.suffix.lstrip(".").lower() or "png"
        fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    _save_single(py_eval, "pyfresean", py_path)
    _save_single(c_eval, "FRESEAN COARSE (C)", c_path)
    return py_path, c_path


def plot_vdos_total_vs_c(
    analysis: FRESEAN,
    eval_dat: Path | str,
    output_path: Path | str,
    xlim: tuple[float, float] | None = None,
    dpi: int = 150,
) -> Path:
    """
    Save total VDoS overlay: pyfresean vs C (full frequency grid).

    Format is inferred from ``output_path`` suffix (default PNG if unknown).
    """
    import matplotlib.pyplot as plt

    eval_path = Path(eval_dat)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    c_eval = load_c_eigenvalues_vdos_normalized(analysis, eval_path)
    c_vdos = np.sum(c_eval, axis=1)
    freqs = analysis.results.freqs
    py_vdos = analysis.results.vdos_total

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_spectra(
        freqs,
        [py_vdos, c_vdos],
        ax=ax,
        labels=["pyfresean", "FRESEAN COARSE (C)"],
        colors=["C0", "C1"],
        linestyles=["-", "--"],
        title="Total VDoS (pyfresean vs C)",
        legend=True,
    )
    if xlim is not None:
        ax.set_xlim(*xlim)
    fig.tight_layout()
    fmt = output_file.suffix.lstrip(".").lower() or "png"
    fig.savefig(output_file, format=fmt, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_file
