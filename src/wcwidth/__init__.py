"""Minimal wcwidth 0.8.2 exports required by bundled ftfy 6.0.3.

The upstream package imports a broad terminal-formatting API from its package
initializer. Version 3 only needs ``wcwidth`` and ``wcswidth`` because ftfy's
import graph loads its formatting helper. All retained implementation modules
are authenticated upstream 0.8.2 source; this initializer narrows the exports.
"""

from ._wcswidth import wcswidth
from ._wcwidth import wcwidth


__version__ = "0.8.2"
__all__ = ("wcwidth", "wcswidth")
