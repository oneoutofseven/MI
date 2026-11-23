"""Compatibility with mi_nmr.data from the first release."""

import sys

from mi_nmr.shared import data

sys.modules[__name__] = data
