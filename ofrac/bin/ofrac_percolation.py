#!/usr/bin/env python
"""Tests whether an orthogonal fracture network percolates across its boundaries.

Reads >=1 file in the format produced by RFGen or Fractran, given on the
command line, and tests --- for the whole domain, or for each of one or more
sub-zones --- whether the fracture network's intersection graph connects all
of the requested boundary faces. See `ofrac.ofracs.OFracGrid.test_percolation`
for what "connects" means.

MASK

The boundary faces to test are given as a mask, in one of three forms:

- a 6-character string of '0'/'1' or 'T'/'F', one per face, in the fixed
  order xmin, xmax, ymin, ymax, zmin, zmax -- e.g. "101010" or "TFTFTF" flags
  xmin, ymin and zmin;
- a comma- or space-separated list of face names -- the full form (xmin,
  xmax, ymin, ymax, zmin, zmax) or the short form (x-, x+, y-, y+, z-, z+) --
  or the keyword "all" for every face -- e.g. "xmin,zmax" or "x- z+";
- a bare integer 0-63 (decimal, or 0x../0b.. literals), read as a bitmask
  where bit 0 is xmin and bit 5 is zmax -- e.g. 33 (0b100001) flags xmin and
  zmax.

SAMPLE ZONES AND DOMAIN

See `-s`/`--sample-zones` and `-d`/`--domain` below; both are parsed the same
way as in `ofracstats_pcalc.py` (see `ofrac.spatialzone.SpatialZone`). Each
sample zone is tested on its own: the network is restricted to the fractures
that reach into the zone, and the requested faces are measured against the
zone's own box rather than the whole network's domain. The (comparatively
expensive) fracture-intersection sweep is computed once per input file and
reused across every zone.

EXIT STATUS

0 if every zone tested percolates across its requested faces; 1 if at least
one does not.
"""

import argparse
import re
import sys
from itertools import chain

from ofrac.ofracs import parse as parse_dfn
from ofrac.spatialzone import SpatialZone

FACE_NAMES = ('xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax')
"""The 6 boundary faces, in the fixed order every mask form agrees on."""

FACE_ALIASES = {
    'x-': 'xmin', 'x+': 'xmax',
    'y-': 'ymin', 'y+': 'ymax',
    'z-': 'zmin', 'z+': 'zmax',
    **{name: name for name in FACE_NAMES},
}


def parse_bnd_mask(s):
    """Parse a mask string into a 6-entry list of bool, in `FACE_NAMES` order

    Accepts the three forms documented in this module's docstring: a 6-char
    '0'/'1' or 'T'/'F' string, symbolic face names, or an integer bitmask.
    """
    s = s.strip()

    if re.fullmatch(r'[01]{6}', s):
        return [c == '1' for c in s]

    if re.fullmatch(r'[TFtf]{6}', s):
        return [c in 'Tt' for c in s]

    try:
        value = int(s, 0)
    except ValueError:
        value = None

    if value is not None:
        if not (0 <= value < 2**len(FACE_NAMES)):
            raise argparse.ArgumentTypeError(
                f"numeric mask must be between 0 and {2**len(FACE_NAMES)-1} "
                f"(got {value})")
        return [bool(value & (1 << i)) for i in range(len(FACE_NAMES))]

    tokens = [t for t in re.split(r'[,\s]+', s) if t]
    if not tokens:
        raise argparse.ArgumentTypeError('mask must not be empty')

    if len(tokens) == 1 and tokens[0].lower() == 'all':
        return [True] * len(FACE_NAMES)

    bnd = [False] * len(FACE_NAMES)
    for token in tokens:
        key = token.lower()
        if key not in FACE_ALIASES:
            raise argparse.ArgumentTypeError(
                f"unrecognized face '{token}'; use one of "
                f"{', '.join(FACE_NAMES)} (or x-,x+,y-,y+,z-,z+), the "
                "keyword 'all', a 6-character string of 0/1 or T/F, or a "
                "numeric bitmask 0-63")
        bnd[FACE_NAMES.index(FACE_ALIASES[key])] = True

    return bnd


def _mask_description(bnd):
    """Return the flagged face names, comma-separated, for display"""
    flagged = [name for name, flag in zip(FACE_NAMES, bnd) if flag]
    return ','.join(flagged) if flagged else '(none)'


def _resolve_domain(fxNet, args_domain):
    if args_domain:
        return SpatialZone(asString=args_domain)

    mima = list(map(float, chain(*fxNet.getBounds())))
    return SpatialZone(start=(mima[0], mima[2], mima[4]),
        end=(mima[1], mima[3], mima[5]))


def _resolve_sample_zones(args_sample_zones, dom):
    if args_sample_zones:
        return [SpatialZone(asString=s)
            for s in args_sample_zones.split(';')]
    return [dom]


def test_zones(fxNet, bnd, dom, sampleZones):
    """Return a list of (zone, percolates) for `sampleZones` in `fxNet`

    Computes the fracture-intersection sweep once and reuses it for every
    zone, per `OFracGrid.test_percolation`'s `pairs` parameter.
    """
    pairs = fxNet.getFxIntersections()

    results = []
    for zone in sampleZones:
        mask = fxNet.getFxMaskIn(zone.start(), zone.end())
        percolates = fxNet.test_percolation(bnd, mask=mask, pairs=pairs,
            start=zone.start(), end=zone.end())
        results.append((zone, percolates))

    return results


def run(args):
    fxNets = [parse_dfn(fnin) for fnin in args.FILES]
    fxNet = fxNets[0] if len(fxNets) == 1 else fxNets[0].merge(*fxNets[1:])

    dom = _resolve_domain(fxNet, args.domain)
    sampleZones = _resolve_sample_zones(args.sample_zones, dom)

    results = test_zones(fxNet, args.mask, dom, sampleZones)
    allPercolate = all(percolates for (_, percolates) in results)

    if not args.quiet:
        if args.json:
            import json
            json.dump({
                'mask': _mask_description(args.mask),
                'domain': str(dom),
                'zones': [
                    {'zone': str(zone), 'percolates': percolates}
                    for (zone, percolates) in results
                ],
                'all_percolate': allPercolate,
            }, sys.stdout, indent=2)
            print()
        else:
            print(f'Mask: {_mask_description(args.mask)}')
            print(f'Domain: {dom}')
            for (i, (zone, percolates)) in enumerate(results):
                status = 'PERCOLATES' if percolates else 'does not percolate'
                print(f'Zone {i} [{zone}]: {status}')

    return 0 if allPercolate else 1


def main(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Tests whether an orthogonal fracture network percolates '
            'across its boundaries.',
        epilog=__doc__)

    parser.add_argument('mask', type=parse_bnd_mask,
        help="The boundary faces to test for connectivity. See MASK in this "
             "tool's --help epilog for the accepted forms.")

    parser.add_argument('FILES', nargs='+',
        help='List of RFGen-style input files, or Fractran problem prefix.')

    parser.add_argument('-s', '--sample-zones', default=None,
        help="""Regions to test individually (must be rectangle- or
        box-shaped). Separate subzones with ';'. Parsed the same way as
        `ofracstats_pcalc.py`'s option of the same name, e.g. "(5,5,5)",
        "start(0,0,3) end(5,5,4)", "(0,0,3)(5,5,4)", or
        "(5,5,5);(0,0,2.5)si(5,5,2.5)". If omitted, one zone covering the
        whole domain is used.""")

    parser.add_argument('-d', '--domain', default=None,
        help="""The whole domain, parsed the same way as `--sample-zones`.
        If omitted, a box bounding all fractures is used.""")

    parser.add_argument('--json', action='store_true',
        help='Print results as JSON instead of text.')

    parser.add_argument('-q', '--quiet', action='store_true',
        help="""Print nothing to the console. The exit status still reports
        the result: 0 if every zone percolates, 1 if at least one does
        not.""")

    return run(parser.parse_args(argv))


if __name__ == '__main__':
    sys.exit(main())
