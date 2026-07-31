#!/usr/bin/env python
"""Run RFGen on one or more .rfp inputs and pickle the resulting OFracGrid.

RFGen writes its fracture network to a statically named file,
'Report-Rfgen.txt', in its working directory.  This wrapper therefore runs
each RFGen job inside its own temporary directory, parses that output with
ofracs.parse(), and writes the OFracGrid as a pickle in the *calling*
directory, named after the input file with a '.pkl' extension:

    ofrac-rfgenwrapper.py fracs_r00.rfp     ->  ./fracs_r00.pkl

The temporary directory is discarded afterwards; pass --keep-rfd to also
retain RFGen's raw text output as <name>.rfd next to the pickle.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from multiprocessing import Pool

from ofrac import ofracs


DEFAULT_RFGEN_EXE_NAME = 'rfgen.1436.exe'
"""RFGen executable used when neither --exe nor $RFGEN_EXE_NAME is set."""

RFGEN_OUT_FN = 'Report-Rfgen.txt'
"""The statically named DFN output that RFGen writes to its working dir."""


def find_rfgen_exe(exe=None):
    """Resolve the RFGen executable.

    Precedence: the --exe value, then $RFGEN_EXE_NAME, then the default.  In
    all cases the name is passed through shutil.which(), so a bare executable
    name found on PATH works as well as an explicit path.
    """
    name = exe or os.environ.get('RFGEN_EXE_NAME', DEFAULT_RFGEN_EXE_NAME)
    return shutil.which(name), name


def run_one(rfp_fn, rfgen_exe, force=False, keep_rfd=False):
    """Run RFGen on `rfp_fn` and pickle the resulting OFracGrid.

    Arguments:
        rfp_fn : str
            Path to the RFGen input (.rfp) file.
        rfgen_exe : str
            Path to the RFGen executable.
        force : bool
            Overwrite an existing .pkl instead of skipping the run.
        keep_rfd : bool
            Also write RFGen's raw text output as <name>.rfd.

    Returns a tuple of `(return code, status string, message)`
    where
        return code is the RFGen return code, or 0 (success) if skipped
        status string is 'skipped', 'complete' or 'error'
    """

    out_dir = os.getcwd()
    stem = os.path.splitext(os.path.basename(rfp_fn))[0]
    pkl_path = os.path.join(out_dir, stem + '.pkl')
    rfd_path = os.path.join(out_dir, stem + '.rfd')

    if not os.path.isfile(rfp_fn):
        return (1, 'error', f'No such input file: {rfp_fn}')

    if os.path.isfile(pkl_path) and not force:
        return (0, 'skipped',
            f'Pickle exists (use --force): {os.path.basename(pkl_path)}')

    rfp_path = os.path.abspath(rfp_fn)
    orig_wd = os.getcwd()

    # RFGen's output file name is fixed, so each run needs its own directory.
    # tempdir must stay in local scope: it is removed when it goes out of it.
    _fnpfx = os.path.basename(__file__) + '_'
    tempdir = tempfile.TemporaryDirectory(prefix=_fnpfx, dir=out_dir)

    try:
        os.chdir(tempdir.name)

        # RFGen resolves relative paths in the .rfp against its working
        # directory, so give it a local copy under its original name.
        local_rfp_fn = os.path.basename(rfp_path)
        shutil.copy(rfp_path, local_rfp_fn)

        cp = subprocess.run(
                [rfgen_exe, local_rfp_fn, ],
                capture_output=True,
                text=True,
            )

        if cp.returncode != 0:
            return (cp.returncode, 'error',
                f'RFGen failed on {rfp_fn}\n'
                f'STDOUT:\n{cp.stdout}\n\nSTDERR:\n{cp.stderr}')

        if not os.path.isfile(RFGEN_OUT_FN):
            return (1, 'error',
                f'RFGen succeeded but produced no {RFGEN_OUT_FN} '
                f'for {rfp_fn}')

        try:
            grid = ofracs.parse(RFGEN_OUT_FN)
        except ofracs.NotValidOFracGridError as exc:
            return (1, 'error',
                f'Could not parse RFGen output of {rfp_fn}: {exc}')

        with open(pkl_path, 'wb') as fout:
            ofracs.OFracGrid.pickleTo(grid, fout)

        if keep_rfd:
            os.replace(RFGEN_OUT_FN, rfd_path)

    finally:
        os.chdir(orig_wd)

    msg = f'Created: {os.path.basename(pkl_path)} ({grid.getFxCount()} fractures)'
    if keep_rfd:
        msg += f', {os.path.basename(rfd_path)}'

    return (0, 'complete', msg)


def build_parser():
    argp = argparse.ArgumentParser(
        description=(
            'Run RFGen on each RFP_FILE in a temporary directory, parse the '
            "resulting 'Report-Rfgen.txt' as an OFracGrid, and write it as a "
            'pickle in the current directory named after the input file with '
            'a .pkl extension.'
        ),
    )

    argp.add_argument('-p', '--nproc',
        type=int,
        default=1,
        metavar='INT',
        help='Number of RFGen runs to perform in parallel (default: 1).')

    argp.add_argument('--exe',
        metavar='RFGEN_EXE',
        default=None,
        help=(
            'RFGen executable to invoke.  Defaults to $RFGEN_EXE_NAME, or '
            f'{DEFAULT_RFGEN_EXE_NAME} if that is unset.  Resolved on PATH.'))

    argp.add_argument('--keep-rfd',
        action='store_true',
        help=("Also write RFGen's raw text output as <name>.rfd beside the "
              'pickle (default: discard it with the temporary directory).'))

    argp.add_argument('-f', '--force',
        action='store_true',
        help='Overwrite an existing <name>.pkl instead of skipping the run.')

    argp.add_argument('rfp_files',
        metavar='RFP_FILE',
        nargs='+',
        help='One or more RFGen input (.rfp) files.')

    return argp


def main():
    argp = build_parser()
    args = argp.parse_args()

    if args.nproc < 1:
        argp.error('--nproc must be >= 1')

    rfgen_exe, requested = find_rfgen_exe(args.exe)
    if rfgen_exe is None:
        print(f'error: RFGen executable not found: {requested}',
            file=sys.stderr)
        sys.exit(1)

    # Guard the case where two inputs in different directories share a stem
    # and would therefore contend for one .pkl in the calling directory.
    stems = [os.path.splitext(os.path.basename(f))[0] for f in args.rfp_files]
    dupes = sorted({s for s in stems if stems.count(s) > 1})
    if dupes:
        argp.error('input files with duplicate names would write the same '
            'pickle: ' + ', '.join(dupes))

    jobs = [(f, rfgen_exe, args.force, args.keep_rfd) for f in args.rfp_files]

    if args.nproc > 1 and len(jobs) > 1:
        with Pool(min(args.nproc, len(jobs))) as pool:
            results = pool.starmap(run_one, jobs)
    else:
        results = [run_one(*j) for j in jobs]

    exitcode = 0
    for rfp_fn, (rc, status, msg) in zip(args.rfp_files, results):
        stream = sys.stderr if status == 'error' else sys.stdout
        print(f'{rfp_fn}: {status}: {msg}', file=stream)
        if rc != 0:
            exitcode = rc

    sys.exit(exitcode)


if __name__ == '__main__':
    main()
