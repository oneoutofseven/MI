"""Compatibility with mi_nmr.engine from the first release."""

import sys

from mi_nmr.shared import engine

sys.modules[__name__] = engine
