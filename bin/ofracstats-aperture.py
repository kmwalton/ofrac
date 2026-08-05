#!/usr/bin/env python
"""Deprecated alias for ofracstats_aperture.py; runs it and warns.

Kept because this name is old enough to be baked into other people's
scripts. Scheduled for removal after 2027-02-05.
"""

import os
import runpy
import sys

_NEW = "ofracstats_aperture.py"
_TARGET = os.path.join(os.path.dirname(os.path.abspath(__file__)), _NEW)

print(
    f"warning: {os.path.basename(__file__)} has been renamed to {_NEW}. "
    f"This alias still works but will be removed after 2027-02-05; "
    f"please update whatever invoked it.",
    file=sys.stderr,
)

# argv[0] becomes the new name so that argparse's usage/--help text teaches
# it rather than repeating the name we are trying to retire. run_name lets
# the target's __main__ block run, and its SystemExit sets our exit status.
sys.argv[0] = _TARGET
runpy.run_path(_TARGET, run_name="__main__")
