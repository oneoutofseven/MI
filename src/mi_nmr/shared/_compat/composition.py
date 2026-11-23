"""Compatibility with mi_nmr.composition from the first release."""

import sys

from mi_nmr.shared import composition

sys.modules[__name__] = composition
