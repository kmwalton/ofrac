#!/usr/bin/env python
"""Run RFGen on one or more .rfp inputs and pickle the resulting OFracGrid.

RFGen writes its fracture network to a statically named file,
'Report-Rfgen.txt', in its working directory.  This wrapper therefore runs
each RFGen job inside its own temporary directory, parses that output with
ofracs.parse(), and writes the OFracGrid as a pickle in the *calling*
directory, named after the input file with a '.pkl' extension:

    ofrac-rfgenwrapper.py fracs_r00.rfp     ->  ./fracs_r00.pkl

Given seeds, it substitutes each one into the .rfp's //random_number_seed
line and generates the cross product of seeds and inputs, naming each
pickle after the seed that produced it:

    ofrac-rfgenwrapper.py --nseeds 20 fracs.rfp
                                            ->  ./fracs_r01.pkl .. _r20.pkl
    ofrac-rfgenwrapper.py --seed 3 7 42 -- fracs.rfp
                                            ->  ./fracs_r03.pkl, _r07, _r42

Note the '--': --seed takes a list, so it must be separated from the
trailing RFP_FILE arguments.

The temporary directory is discarded afterwards; pass --keep-rfd to also
retain RFGen's raw text output as <name>.rfd next to the pickle, and
--keep-rfp to retain the seeded .rfp that produced it.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from multiprocessing import Pool

from ofrac import ofracs


FALLBACK_RFGEN_EXE_NAME = 'rfgen.1436.exe'
"""Used only when the PATH scan finds no RFGen at all."""

RFGEN_OUT_FN = 'Report-Rfgen.txt'
"""The statically named DFN output that RFGen writes to its working dir."""

SEED_RE = re.compile(
    r'^[ \t]*[+-]?(?:\d+\.?\d*|\.\d+)[ \t]*//random_number_seed[ \t]*$',
    re.MULTILINE)
"""The seed line in an .rfp, e.g. '1                \t//random_number_seed'."""


def substitute_seed(text, seed):
    """Return the .rfp text with its //random_number_seed set to `seed`.

    Raises rather than silently generating a duplicate network if the field
    is absent or ambiguous -- an unsubstituted .rfp yields N identical
    "realizations", which is indistinguishable from a working pool
    downstream.
    """
    new, n = SEED_RE.subn(f'{seed} //random_number_seed', text)
    if n != 1:
        raise ValueError(f'expected exactly one //random_number_seed line in '
                         f'the .rfp, found {n}')
    return new


def parse_seed_args(values):
    """Flatten --seed values, which may be spaced and/or comma separated."""
    seeds = []
    for v in values:
        for tok in re.split(r'[,\s]+', v):
            if not tok:
                continue
            if not re.fullmatch(r'[+-]?\d+', tok):
                raise ValueError(f'not an integer seed: {tok!r}')
            seeds.append(int(tok))
    return seeds


def read_seeds_file(path):
    """Read integer seeds from a file.

    Seeds may be separated by whitespace or commas, one per line or many.
    Blank lines are ignored, as is anything after a '#', '!' or '//'
    comment marker, following the .rfp files' own conventions.
    """
    with open(path, 'r') as fin:
        text = fin.read()

    text = re.sub(r'(#|!|//).*', '', text)
    return parse_seed_args(text.split())


def seed_stem(stem, seed, width):
    """Realization pickle stem: '<stem>_r<seed>', zero padded to `width`."""
    return f'{stem}_r{seed:0>{width}}'

RFGEN_EXE_RE = re.compile(
    r'^rfgen'
    r'(?:[._-](?P<version>\d+(?:[._]\d+)*[a-z]*))?'
    r'(?P<ext>\.[^.]+)?$',
    re.IGNORECASE)
"""Matches 'rfgen.exe', 'rfgen.1436.exe', 'rfgen-1.2.3b', ... but not
'RFGen_help.1432b.txt', whose extension is rejected separately."""


def _version_key(version):
    """Sort key ordering RFGen version codes, lowest first.

    '1436' > '1432b' > '1432' > '1388', and an unversioned name sorts below
    every versioned one, since it carries no claim to being newer.
    """
    if not version:
        return ((-1,), '')
    m = re.match(r'^(?P<nums>\d+(?:[._]\d+)*)(?P<suffix>[a-z]*)$',
        version, re.IGNORECASE)
    nums = tuple(int(n) for n in re.split(r'[._]', m.group('nums')))
    return (nums, m.group('suffix').lower())


def _is_executable(path):
    """True if `path` looks runnable: a PATHEXT extension, or the x bit."""
    exts = [e.lower() for e in
        os.environ.get('PATHEXT', '').split(os.pathsep) if e]
    ext = os.path.splitext(path)[1].lower()
    if exts:
        return ext in exts
    return os.access(path, os.X_OK)


def scan_path_for_rfgen():
    """Find every RFGen-looking executable on $PATH.

    Returns a list of `(directory, candidates)` pairs in PATH order, where
    `candidates` is that directory's matches sorted newest version first as
    `(version string or None, file name)`.  Directories with no match are
    omitted, as are duplicate directories (PATH may repeat them).
    """
    found = []
    seen_dirs = set()

    for d in os.environ.get('PATH', '').split(os.pathsep):
        if not d:
            continue
        key = os.path.normcase(os.path.abspath(d))
        if key in seen_dirs or not os.path.isdir(d):
            continue
        seen_dirs.add(key)

        try:
            names = os.listdir(d)
        except OSError:
            # unreadable PATH entries are the OS's problem, not ours
            continue

        cands = []
        for n in names:
            m = RFGEN_EXE_RE.match(n)
            if m and _is_executable(os.path.join(d, n)):
                cands.append((m.group('version'), n))

        if cands:
            cands.sort(key=lambda c: _version_key(c[0]), reverse=True)
            found.append((d, cands))

    return found


def find_rfgen_exe(exe=None):
    """Resolve the RFGen executable to run.

    Precedence: the --exe value, then $RFGEN_EXE_NAME, then the newest
    version found in the first $PATH component holding any RFGen.  The first
    two are passed through shutil.which(), so a bare name on PATH works as
    well as an explicit path.

    Returns `(resolved path or None, requested name, report lines)`, where
    the report describes what the PATH scan saw.  It is empty when an
    explicit --exe was given, since nothing was inferred in that case.
    """
    if exe:
        return shutil.which(exe), exe, []

    found = scan_path_for_rfgen()
    newest = os.path.join(found[0][0], found[0][1][0][1]) if found else None

    env_name = os.environ.get('RFGEN_EXE_NAME')
    if env_name:
        name, origin = env_name, ' from $RFGEN_EXE_NAME'
    elif newest:
        name, origin = newest, ''
    else:
        name, origin = FALLBACK_RFGEN_EXE_NAME, ' (no rfgen found on $PATH)'

    resolved = shutil.which(name)
    _norm = lambda p: os.path.normcase(os.path.abspath(p))
    chosen = _norm(resolved) if resolved else None

    report = [f'RFGen: using {resolved or name}{origin}']

    if resolved and newest and chosen != _norm(newest):
        report.append(f'  note: a newer build is on $PATH: {newest}')

    # Everything else the scan turned up, so a stale pick is visible.
    for i, (d, cands) in enumerate(found):
        others = [n for _, n in cands if _norm(os.path.join(d, n)) != chosen]
        if others:
            label = 'also in' if i == 0 else 'shadowed in'
            report.append(f'  {label} {d}: ' + ', '.join(others))

    return resolved, name, report


def run_one(rfp_fn, rfgen_exe, out_stem=None, seed=None,
        force=False, keep_rfd=False, keep_rfp=False):
    """Run RFGen on `rfp_fn` and pickle the resulting OFracGrid.

    Arguments:
        rfp_fn : str
            Path to the RFGen input (.rfp) file.
        rfgen_exe : str
            Path to the RFGen executable.
        out_stem : str or None
            Base name of the outputs.  Defaults to the input's stem.
        seed : int or None
            Substituted into the .rfp's //random_number_seed line.  When
            None the input is used as it stands.
        force : bool
            Overwrite an existing .pkl instead of skipping the run.
        keep_rfd : bool
            Also write RFGen's raw text output as <out_stem>.rfd.
        keep_rfp : bool
            Also write the seeded input as <out_stem>.rfp.

    Returns a tuple of `(return code, status string, message)`
    where
        return code is the RFGen return code, or 0 (success) if skipped
        status string is 'skipped', 'complete' or 'error'
    """

    out_dir = os.getcwd()
    if out_stem is None:
        out_stem = os.path.splitext(os.path.basename(rfp_fn))[0]
    pkl_path = os.path.join(out_dir, out_stem + '.pkl')
    rfd_path = os.path.join(out_dir, out_stem + '.rfd')
    rfp_out_path = os.path.join(out_dir, out_stem + '.rfp')

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
        if seed is None:
            shutil.copy(rfp_path, local_rfp_fn)
        else:
            try:
                with open(rfp_path, 'r') as fin:
                    text = substitute_seed(fin.read(), seed)
            except (OSError, ValueError) as exc:
                return (1, 'error',
                    f'Could not seed {os.path.basename(rfp_fn)}: {exc}')
            with open(local_rfp_fn, 'w') as fout:
                fout.write(text)
            if keep_rfp:
                shutil.copy(local_rfp_fn, rfp_out_path)

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
    for kept, path in ((keep_rfd, rfd_path),
                       (keep_rfp and seed is not None, rfp_out_path)):
        if kept:
            msg += f', {os.path.basename(path)}'

    return (0, 'complete', msg)


def build_parser():
    argp = argparse.ArgumentParser(
        description=(
            'Run RFGen on each RFP_FILE in a temporary directory, parse the '
            "resulting 'Report-Rfgen.txt' as an OFracGrid, and write it as a "
            'pickle in the current directory named after the input file with '
            'a .pkl extension.  Given seeds, every seed is run against every '
            'input and the pickles are named <stem>_r<seed>.pkl.'
        ),
    )

    seedgrp = argp.add_mutually_exclusive_group()

    seedgrp.add_argument('--seed',
        metavar='SEED',
        nargs='+',
        default=None,
        help=('Explicit list of integer seeds, space and/or comma separated, '
              'e.g. --seed 3 7 42 or --seed 3,7,42.  Being greedy, it must '
              "be separated from the RFP_FILE arguments by '--'."))

    seedgrp.add_argument('--nseeds',
        metavar='N',
        type=int,
        default=None,
        help='Use the seeds 1..N.')

    seedgrp.add_argument('--fseeds',
        metavar='SEEDS_FILE',
        default=None,
        help=('Read integer seeds from SEEDS_FILE, separated by whitespace '
              "or commas.  Blank lines are ignored, as is anything after a "
              "'#', '!' or '//' comment marker."))

    argp.add_argument('-p', '--nproc',
        type=int,
        default=1,
        metavar='INT',
        help='Number of RFGen runs to perform in parallel (default: 1).')

    argp.add_argument('--exe',
        metavar='RFGEN_EXE',
        default=None,
        help=(
            'RFGen executable to invoke.  Defaults to $RFGEN_EXE_NAME, or, '
            'if that is unset, the highest-versioned rfgen*.exe in the first '
            '$PATH component containing one.  Unless --exe is given, the '
            'choice and the alternatives passed over are reported.'))

    argp.add_argument('--keep-rfd',
        action='store_true',
        help=("Also write RFGen's raw text output as <name>.rfd beside the "
              'pickle (default: discard it with the temporary directory).'))

    argp.add_argument('--keep-rfp',
        action='store_true',
        help=('Also write the seeded input as <name>.rfp beside the pickle, '
              'so a realization can be re-run by hand (seeded runs only).'))

    argp.add_argument('-f', '--force',
        action='store_true',
        help='Overwrite an existing <name>.pkl instead of skipping the run.')

    # nargs='*', though at least one is required: a missing RFP_FILE is
    # usually a greedy --seed having eaten it, and argparse's own "required"
    # complaint would pre-empt the seed scan that can say so.
    argp.add_argument('rfp_files',
        metavar='RFP_FILE',
        nargs='*',
        help='One or more RFGen input (.rfp) files.')

    return argp


def resolve_seeds(args, argp):
    """Turn the mutually exclusive seed options into a list of seeds.

    Returns an empty list when none were given, meaning the inputs are run
    exactly as they stand.
    """
    if args.nseeds is not None:
        if args.nseeds < 1:
            argp.error('--nseeds must be >= 1')
        return list(range(1, args.nseeds + 1))

    if args.seed is not None:
        try:
            seeds = parse_seed_args(args.seed)
        except ValueError as exc:
            # Overwhelmingly this is the first RFP_FILE, swallowed by the
            # greedy --seed, so name the remedy rather than just the rule.
            argp.error(f'--seed: {exc}: every --seed value must be an '
                "integer.  Separate the seeds from the RFP_FILE arguments "
                "with '--', e.g. --seed 3 7 42 -- fracs.rfp")
        if not seeds:
            argp.error('--seed: no seeds given')
        return seeds

    if args.fseeds is not None:
        try:
            seeds = read_seeds_file(args.fseeds)
        except OSError as exc:
            argp.error(f'--fseeds: {exc}')
        except ValueError as exc:
            argp.error(f'--fseeds: {args.fseeds}: {exc}')
        if not seeds:
            argp.error(f'--fseeds: no seeds found in {args.fseeds}')
        return seeds

    return []


def build_jobs(rfp_files, seeds, rfgen_exe, force, keep_rfd, keep_rfp):
    """Expand inputs x seeds into run_one() argument tuples, with labels.

    Ordered seed-major, so the inputs of a realization advance together: if
    a run is cut short, what is on disk is a set of complete realizations
    rather than every seed of the first input and none of the last.
    """
    stems = [os.path.splitext(os.path.basename(f))[0] for f in rfp_files]

    jobs = []
    labels = []

    if not seeds:
        for f, stem in zip(rfp_files, stems):
            jobs.append((f, rfgen_exe, stem, None, force, keep_rfd, keep_rfp))
            labels.append(f)
        return jobs, labels

    width = max(len(str(s)) for s in seeds)
    for seed in seeds:
        for f, stem in zip(rfp_files, stems):
            jobs.append((f, rfgen_exe, seed_stem(stem, seed, width), seed,
                force, keep_rfd, keep_rfp))
            labels.append(f'{f} r{seed}')

    return jobs, labels


def main():
    argp = build_parser()
    args = argp.parse_args()

    if args.nproc < 1:
        argp.error('--nproc must be >= 1')

    # before the RFP_FILE check: a greedy --seed is the likelier culprit
    seeds = resolve_seeds(args, argp)

    if not args.rfp_files:
        argp.error('at least one RFP_FILE is required')

    rfgen_exe, requested, report = find_rfgen_exe(args.exe)
    for line in report:
        print(line, file=sys.stderr)

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

    jobs, labels = build_jobs(args.rfp_files, seeds, rfgen_exe,
        args.force, args.keep_rfd, args.keep_rfp)

    if seeds:
        print(f'RFGen: {len(seeds)} seed(s) x {len(args.rfp_files)} input(s) '
            f'= {len(jobs)} realization(s)', file=sys.stderr)

    if args.nproc > 1 and len(jobs) > 1:
        with Pool(min(args.nproc, len(jobs))) as pool:
            results = pool.starmap(run_one, jobs)
    else:
        results = [run_one(*j) for j in jobs]

    exitcode = 0
    for label, (rc, status, msg) in zip(labels, results):
        stream = sys.stderr if status == 'error' else sys.stdout
        print(f'{label}: {status}: {msg}', file=stream)
        if rc != 0:
            exitcode = rc

    sys.exit(exitcode)


if __name__ == '__main__':
    main()
