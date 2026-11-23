"""Compatibility with mi_nmr.masks from the first release."""

import sys

from mi_nmr.pretraining import masks

sys.modules[__name__] = masks
