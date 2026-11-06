import sys

import numpy as np
import pytest

from pyfresean.postprocess.visualization import (
    oscillation_frames,
    scale_to_max_length,
    visualize_mode_displacement,
)
from pyfresean.tests.utils import make_Universe


def test_scale_to_max_length():
    vectors = np.array([[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    scaled = scale_to_max_length(vectors, max_length=1.5)
    np.testing.assert_allclose(scaled[0], [1.5, 0.0, 0.0])
    np.testing.assert_allclose(scaled[1], 0.0)


def test_oscillation_frames_hits_both_extremes():
    positions = np.zeros((2, 3))
    displacements = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    frames = oscillation_frames(positions, displacements, n_frames=4)
    assert frames.shape == (4, 2, 3)
    np.testing.assert_allclose(frames[0], positions)
    np.testing.assert_allclose(frames[1], displacements)
    np.testing.assert_allclose(frames[3], -displacements)


def test_oscillation_frames_rejects_bad_input():
    with pytest.raises(ValueError, match="n_frames"):
        oscillation_frames(np.zeros((1, 3)), np.ones((1, 3)), n_frames=1)
    with pytest.raises(ValueError, match="shape"):
        oscillation_frames(np.zeros((2, 3)), np.ones((1, 3)))


def test_visualize_mode_displacement_adds_frames_and_both_arrow_directions(
    monkeypatch,
):
    universe = make_Universe(size=(3, 1, 1), n_frames=1)
    universe.atoms.positions = np.eye(3)
    saved = universe.atoms.positions.copy()
    displacements = np.array(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]
    )
    calls = {}

    class _View:
        def addModelsAsFrames(self, data, fmt):
            calls["frames"] = data
            calls["fmt"] = fmt

        def setStyle(self, spec):
            calls["style"] = spec

        def addArrow(self, spec):
            calls.setdefault("arrows", []).append(spec)

        def setBackgroundColor(self, color):
            calls["background"] = color

        def zoomTo(self):
            calls["zoomed"] = True

        def animate(self, options):
            calls["animate"] = options

    class _Py3Dmol:
        @staticmethod
        def view(width, height):
            calls["size"] = (width, height)
            return _View()

    monkeypatch.setitem(sys.modules, "py3Dmol", _Py3Dmol)
    visualize_mode_displacement(
        universe.atoms,
        displacements,
        n_frames=4,
        max_length=2.0,
        style="stick",
    )

    assert calls["fmt"] == "pdb"
    assert calls["frames"].count("MODEL") == 4
    assert "ATOM" in calls["frames"]
    assert calls["animate"]["loop"] == "forward"
    assert calls["style"] == {"stick": {"radius": 0.15}}
    assert len(calls["arrows"]) == 4
    assert calls["arrows"][0]["color"] == "orange"
    assert calls["arrows"][1]["color"] == "gold"
    assert calls["arrows"][0]["radius"] == 0.24
    assert calls["arrows"][2]["radius"] == calls["arrows"][0]["radius"]
    np.testing.assert_allclose(universe.atoms.positions, saved)


def test_visualize_mode_displacement_uses_separate_arrow_positions(monkeypatch):
    universe = make_Universe(size=(2, 1, 1), n_frames=1)
    universe.atoms.positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    calls = {}

    class _View:
        def addModelsAsFrames(
            self, data, fmt
        ):  # pylint: disable=unused-argument
            calls["frames"] = data

        def setStyle(self, spec):
            pass

        def addArrow(self, spec):
            calls.setdefault("arrows", []).append(spec)

        def setBackgroundColor(self, color):
            pass

        def zoomTo(self):
            pass

        def animate(self, options):
            pass

    class _Py3Dmol:
        @staticmethod
        def view(width, height):  # pylint: disable=unused-argument
            return _View()

    monkeypatch.setitem(sys.modules, "py3Dmol", _Py3Dmol)
    visualize_mode_displacement(
        universe.atoms,
        np.ones((2, 3)),
        arrow_positions=np.array([[5.0, 0.0, 0.0]]),
        arrow_displacements=np.array([[0.0, 4.0, 0.0]]),
        max_length=2.0,
        animate=True,
        n_frames=4,
        style="cartoon",
    )
    assert calls["arrows"][0]["start"]["x"] == pytest.approx(5.0)
    assert len(calls["arrows"]) == 2


def test_visualize_mode_displacement_cartoon_plus_stick_sets_both_styles(
    monkeypatch,
):
    universe = make_Universe(size=(2, 1, 1), n_frames=1)
    calls = {}

    class _View:
        def addModelsAsFrames(
            self, data, fmt
        ):  # pylint: disable=unused-argument
            pass

        def setStyle(self, spec):
            calls["style"] = spec

        def addArrow(self, spec):  # pylint: disable=unused-argument
            pass

        def setBackgroundColor(self, color):  # pylint: disable=unused-argument
            pass

        def zoomTo(self):
            pass

        def animate(self, options):  # pylint: disable=unused-argument
            pass

    class _Py3Dmol:
        @staticmethod
        def view(width, height):  # pylint: disable=unused-argument
            return _View()

    monkeypatch.setitem(sys.modules, "py3Dmol", _Py3Dmol)
    visualize_mode_displacement(
        universe.atoms,
        np.ones((2, 3)),
        style="cartoon+stick",
        animate=False,
        arrows=False,
    )
    assert calls["style"] == {
        "cartoon": {"color": "spectrum"},
        "stick": {"radius": 0.12},
    }
