"""Compatibility with mi_nmr.readout from the first release."""

import sys

from mi_nmr.finetuning import readout

sys.modules[__name__] = readout
