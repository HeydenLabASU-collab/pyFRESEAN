"""
Unit and regression test for the pyfresean package.
"""

# Import package, test suite, and other packages as needed
import sys

import pyfresean
from pyfresean import FRESEAN, Align, CoarseGrain, CoarseGrainMap, Unwrap


def test_pyfresean_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "pyfresean" in sys.modules


def test_mdanalysis_logo_length(mdanalysis_logo_text):
    """Example test using a fixture defined in conftest.py"""
    logo_lines = mdanalysis_logo_text.split("\n")
    assert len(logo_lines) == 46, "Logo file does not have 46 lines!"


def test_public_exports():
    assert FRESEAN is not None
    assert Align is not None
    assert Unwrap is not None
    assert CoarseGrain is not None
    assert CoarseGrainMap is not None
    assert pyfresean.__version__
