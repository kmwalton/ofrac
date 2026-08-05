"""Tests for the ofracstats-percolation CLI tool.

Covers the mask-string parsing (the three accepted forms), the per-zone test
loop that reuses one precomputed intersection sweep across sample zones, and
the `main()` entry point's text/JSON/quiet output and exit-status contract.
"""

import argparse
import contextlib
import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent                      # .../libdev/ofrac
if str(_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT.parent))

from ofrac.ofracs import OFracGrid  # noqa: E402


def _load_module(name, filename):
    """Import a hyphenated script as a module (its CLI is __main__-guarded)."""
    spec = importlib.util.spec_from_file_location(name, _ROOT / 'bin' / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


percolation = _load_module('ofracstats_percolation', 'ofracstats-percolation.py')


class TestParseBndMask(unittest.TestCase):

    def test_binary_string(self):
        self.assertEqual(percolation.parse_bnd_mask('100001'),
            [True, False, False, False, False, True])

    def test_tf_string(self):
        self.assertEqual(percolation.parse_bnd_mask('TfFfFt'),
            [True, False, False, False, False, True])

    def test_symbolic_full_names(self):
        self.assertEqual(percolation.parse_bnd_mask('xmin,zmax'),
            [True, False, False, False, False, True])

    def test_symbolic_short_names_and_whitespace(self):
        self.assertEqual(percolation.parse_bnd_mask('x-  z+'),
            [True, False, False, False, False, True])

    def test_all_keyword(self):
        self.assertEqual(percolation.parse_bnd_mask('all'), [True] * 6)

    def test_numeric_bitmask(self):
        # bit0 = xmin, bit5 = zmax: 1 + 32 = 33
        self.assertEqual(percolation.parse_bnd_mask('33'),
            [True, False, False, False, False, True])

    def test_numeric_bitmask_hex(self):
        self.assertEqual(percolation.parse_bnd_mask('0x21'),
            [True, False, False, False, False, True])

    def test_numeric_bitmask_out_of_range_rejected(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            percolation.parse_bnd_mask('64')

    def test_unrecognized_face_name_rejected(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            percolation.parse_bnd_mask('bogus')

    def test_empty_mask_rejected(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            percolation.parse_bnd_mask('   ')


class TestMaskDescription(unittest.TestCase):

    def test_lists_flagged_faces_in_order(self):
        self.assertEqual(
            percolation._mask_description(
                [True, False, True, False, False, True]),
            'xmin,ymin,zmax')

    def test_no_faces_flagged(self):
        self.assertEqual(percolation._mask_description([False] * 6), '(none)')


class TestZones(unittest.TestCase):

    def _chain_net(self):
        """A-B-C chain, as in TestConnectivity in test_OFracGrid.py: 0 meets
        1, 1 meets 2, but 0 and 2 never meet directly. Fracture 0 alone
        reaches xmin; fracture 2 alone reaches zmax."""
        return OFracGrid(domainSize=(10., 10., 10.), fx=[
            (0., 3., 0., 3., 2., 2., 1e-4),   # 0: perp z at z=2
            (3., 3., 0., 3., 0., 6., 1e-4),   # 1: perp x at x=3, crosses 0
            (2., 8., 3., 3., 3., 10., 1e-4),  # 2: perp y at y=3, crosses 1
        ])

    def test_whole_domain_percolates_via_the_chain(self):
        g = self._chain_net()
        dom = percolation.SpatialZone(start=(0., 0., 0.), end=(10., 10., 10.))
        bnd = percolation.parse_bnd_mask('xmin,zmax')

        [(zone, percolates)] = percolation.test_zones(g, bnd, dom, [dom])
        self.assertIs(zone, dom)
        self.assertTrue(percolates)

    def test_subzone_around_the_bridge_only_does_not_percolate(self):
        g = self._chain_net()
        dom = percolation.SpatialZone(start=(0., 0., 0.), end=(10., 10., 10.))
        bnd = percolation.parse_bnd_mask('xmin,zmax')

        # this subzone only reaches fracture 1 (the bridge), which touches
        # neither xmin nor zmax on its own
        subzone = percolation.SpatialZone(
            start=(3., 0., 3.), end=(3., 1., 10.))

        [(_, percolates)] = percolation.test_zones(g, bnd, dom, [subzone])
        self.assertFalse(percolates)

    def test_multiple_zones_are_each_tested_independently(self):
        g = self._chain_net()
        dom = percolation.SpatialZone(start=(0., 0., 0.), end=(10., 10., 10.))
        bnd = percolation.parse_bnd_mask('xmin,zmax')
        subzone = percolation.SpatialZone(
            start=(3., 0., 3.), end=(3., 1., 10.))

        results = percolation.test_zones(g, bnd, dom, [dom, subzone])
        self.assertEqual([p for (_, p) in results], [True, False])


class TestMain(unittest.TestCase):

    def _args(self, **overrides):
        defaults = dict(mask=percolation.parse_bnd_mask('xmin,zmax'),
            FILES=[], sample_zones=None, domain='(10,10,10)',
            json=False, quiet=False)
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def setUp(self):
        # main() parses files via ofrac.ofracs.parse, which this test does
        # not want to exercise; patch it to hand back a fixed in-memory net
        self._orig_parse_dfn = percolation.parse_dfn
        self.net = TestZones()._chain_net()
        percolation.parse_dfn = lambda fnin: self.net

    def tearDown(self):
        percolation.parse_dfn = self._orig_parse_dfn

    def test_text_output_and_exit_status(self):
        args = self._args(FILES=['dummy.rfd'])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = percolation.main(args)

        self.assertEqual(status, 0)
        self.assertIn('PERCOLATES', out.getvalue())

    def test_json_output_is_well_formed(self):
        args = self._args(FILES=['dummy.rfd'], json=True)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = percolation.main(args)

        self.assertEqual(status, 0)
        data = json.loads(out.getvalue())
        self.assertEqual(data['mask'], 'xmin,zmax')
        self.assertTrue(data['all_percolate'])
        self.assertEqual(len(data['zones']), 1)
        self.assertTrue(data['zones'][0]['percolates'])

    def test_quiet_suppresses_output_but_keeps_exit_status(self):
        # a subzone that only reaches the bridge fracture: does not percolate
        args = self._args(FILES=['dummy.rfd'],
            sample_zones='start(3,0,3)end(3,1,10)', quiet=True)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = percolation.main(args)

        self.assertEqual(status, 1)
        self.assertEqual(out.getvalue(), '')

    def test_exit_status_is_1_when_any_zone_fails(self):
        args = self._args(FILES=['dummy.rfd'],
            sample_zones='(10,10,10);start(3,0,3)end(3,1,10)')
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = percolation.main(args)

        self.assertEqual(status, 1)


if __name__ == '__main__':
    unittest.main()
