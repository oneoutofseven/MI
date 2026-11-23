"""Compatibility with mi_nmr.objectives from the first release."""

import sys

from mi_nmr.pretraining import objectives

sys.modules[__name__] = objectives
