import numpy as np

from pyfresean.postprocess import interpolate_spectrum, plot_spectra


def test_interpolate_spectrum_length():
    freqs = np.linspace(0, 100, 10)
    vdos = np.sin(freqs / 10)
    x_hi, y_hi = interpolate_spectrum(freqs, vdos, interpol_step=5)
    assert len(x_hi) == 50
    assert len(y_hi) == 50


def test_plot_spectra_returns_axes():
    freqs = np.linspace(0, 100, 10)
    vdos = np.sin(freqs / 10)
    ax = plot_spectra(freqs, vdos, legend=False)
    assert len(ax.lines) == 2  # spline + markers
