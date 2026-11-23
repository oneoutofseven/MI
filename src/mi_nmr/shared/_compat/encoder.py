"""Compatibility with mi_nmr.encoder from the first release."""

import sys

from mi_nmr.shared import encoder

sys.modules[__name__] = encoder
