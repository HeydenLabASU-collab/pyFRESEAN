"""
PyFRESEAN
A Python package for FRESEAN based analysis
"""

# Add imports here
from importlib.metadata import version

from pyfresean.analysis import FRESEAN
from pyfresean.coarsegrain import CoarseGrain, CoarseGrainMap, SUPPORTED_CG_METHODS
from pyfresean.transformations import Align, Unwrap

__all__ = [
    "FRESEAN",
    "Align",
    "Unwrap",
    "CoarseGrain",
    "CoarseGrainMap",
    "SUPPORTED_CG_METHODS",
    "__version__",
]
__version__ = version("pyfresean")
