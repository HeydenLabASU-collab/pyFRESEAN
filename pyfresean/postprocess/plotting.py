from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline


def interpolate_spectrum(
    frequencies: np.ndarray,
    intensities: np.ndarray,
    interpol_step: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    frequencies = np.asarray(frequencies, dtype=float)
    intensities = np.asarray(intensities, dtype=float)
    x_hi = np.linspace(
        frequencies[0], frequencies[-1], interpol_step * len(frequencies)
    )
    y_hi = CubicSpline(frequencies, intensities)(x_hi)
    return x_hi, y_hi


def plot_spectra(
    frequencies: np.ndarray,
    intensities: np.ndarray | Sequence[np.ndarray],
    ax: plt.Axes | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    labels: str | Sequence[str] | None = None,
    colors: str | Sequence[str] = "black",
    interpol_step: int = 5,
    show_markers: bool = True,
    vlines: float | Sequence[float] | None = None,
    vline_colors: str | Sequence[str] | None = None,
    title: str | None = None,
    xlabel: str = "frequency (cm$^{-1}$)",
    ylabel: str = "VDoS",
    legend: bool = True,
    **line_kwargs,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    if np.ndim(intensities) == 1:
        series = [np.asarray(intensities, dtype=float)]
    else:
        series = [np.asarray(y, dtype=float) for y in intensities]

    if isinstance(colors, str):
        color_list = [colors] * len(series)
    else:
        color_list = list(colors)

    linestyles = line_kwargs.pop("linestyles", None)
    if linestyles is None:
        style_list = ["-"] * len(series)
    elif isinstance(linestyles, str):
        style_list = [linestyles] * len(series)
    else:
        style_list = list(linestyles)

    if labels is None:
        label_list: list[str | None] = [None] * len(series)
    elif isinstance(labels, str):
        label_list = [labels]
    else:
        label_list = list(labels)

    for idx, y in enumerate(series):
        x_hi, y_hi = interpolate_spectrum(
            frequencies, y, interpol_step=interpol_step
        )
        label = label_list[idx] if idx < len(label_list) else None
        color = color_list[idx] if idx < len(color_list) else "black"
        linestyle = style_list[idx] if idx < len(style_list) else "-"
        ax.plot(
            x_hi,
            y_hi,
            color=color,
            linestyle=linestyle,
            label=label,
            **line_kwargs,
        )
        if show_markers:
            ax.plot(frequencies, y, "o", color=color, markersize=2)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    if vlines is not None:
        if isinstance(vlines, (float, int)):
            vlines = [float(vlines)]
        if vline_colors is None:
            vline_colors = ["gray"] * len(vlines)
        elif isinstance(vline_colors, str):
            vline_colors = [vline_colors] * len(vlines)
        for vline, vcolor in zip(vlines, vline_colors):
            ax.axvline(vline, color=vcolor, linestyle=":", alpha=0.7)

    if legend and any(label_list):
        ax.legend()
    return ax
