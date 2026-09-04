"""
Global pytest fixtures
"""

# Use this file if you need to share any fixtures
# across multiple modules
# More information at
# https://docs.pytest.org/en/stable/how-to/fixtures.html#scope-sharing-fixtures-across-classes-modules-packages-or-session

import pytest

from pyfresean.data.files import MDANALYSIS_LOGO


def pytest_addoption(parser):
    parser.addoption(
        "--c-ref-plot",
        action="store_true",
        default=False,
        help="Write c_ref comparison plots (PNG; requires matplotlib).",
    )
    parser.addoption(
        "--c-ref-plot-dir",
        action="store",
        default=None,
        metavar="DIR",
        help=(
            "Output directory for --c-ref-plot "
            "(default: tests/data/fresean_c_ref/ala_dipeptide_gas_300K/plots/)."
        ),
    )


@pytest.fixture
def mdanalysis_logo_text() -> str:
    """Example fixture demonstrating how data files can be accessed"""
    with open(MDANALYSIS_LOGO, "r", encoding="utf8") as f:
        logo_text = f.read()
    return logo_text
