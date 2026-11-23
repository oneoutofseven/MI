"""Compatibility with mi_nmr.checkpoints from the first release."""

import sys

from mi_nmr.shared import checkpoints

sys.modules[__name__] = checkpoints
