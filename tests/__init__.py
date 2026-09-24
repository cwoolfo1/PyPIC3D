"""Shared unittest package for PyPIC3D tests."""
import sys
from pathlib import Path

# The BZ demo is run as a script from its own directory, so its modules import
# each other by bare name; make those names resolvable when tests import it.
sys.path.append(str(Path(__file__).resolve().parents[1] / "demos/static_metric_relativity/bz_monopole"))
