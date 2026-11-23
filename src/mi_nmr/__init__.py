"""MI: measurement-aware learning for plasma NMR."""

from pathlib import Path as _Path

# Let Python resolve first-release module names only when a caller imports one.
__path__.append(str(_Path(__file__).parent / "shared" / "_compat"))
