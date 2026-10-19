"""Tests of matrix block sampling in MatrixBlockOFracGrid"""

import argparse
import unittest

import numpy as np

from ofrac.ofracs import OFracGrid
from ofrac.matrixblock import MatrixBlockOFracGrid
from ofrac.bin.ofracstats_block import BlockHow, BlockHowLHS


def _make_4x4x2_grid():
    """Persistent fractures at x=1,3; y=2; z=1, making 3x2x2 = 12 blocks"""
    ap = 1e-4
    return MatrixBlockOFracGrid(OFracGrid(domainSize=(4., 4., 2.), fx=[
        (1., 1., 0., 4., 0., 2., ap),
        (3., 3., 0., 4., 0., 2., ap),
        (0., 4., 2., 2., 0., 2., ap),
        (0., 4., 0., 4., 1., 1., ap),
    ],))


class TestLHSPoints(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)

    def _strata(self, pts, origin, size, nstrata):
        """Stratum index of each point on each axis"""
        o, s, k = map(np.asarray, (origin, size, nstrata))
        return np.floor((pts - o)/s*k).astype(int)

    def test_scalar_is_classic_lhs(self):
        o, s = (10., 20., 30.), (4., 4., 2.)
        pts = MatrixBlockOFracGrid._lhs_points(o, s, 5)
        self.assertEqual(pts.shape, (5, 3))
        istrat = self._strata(pts, o, s, 3*[5,])
        for iax in range(3):
            with self.subTest(axis='xyz'[iax]):
                self.assertEqual(sorted(istrat[:,iax]), list(range(5)))

    def test_triple_covers_every_stratum(self):
        o, s, ns = (0., 0., 0.), (4., 4., 2.), (2, 7, 3)
        pts = MatrixBlockOFracGrid._lhs_points(o, s, ns)
        self.assertEqual(pts.shape, (7, 3))
        istrat = self._strata(pts, o, s, ns)
        for iax, k in enumerate(ns):
            with self.subTest(axis='xyz'[iax]):
                counts = np.bincount(istrat[:,iax], minlength=k)
                self.assertEqual(len(counts), k)
                self.assertTrue(np.all(counts >= 7//k))
                self.assertTrue(np.all(counts <= -(-7//k)))

    def test_bad_strata(self):
        for bad in [(2, 3), 0, (1, -1, 2)]:
            with self.subTest(nstrata=bad):
                with self.assertRaises(ValueError):
                    MatrixBlockOFracGrid._lhs_points((0,0,0), (1,1,1), bad)


class TestFindBlocksLHS(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.g = _make_4x4x2_grid()

    def test_blocks_are_unique(self):
        blocks = self.g.find_blocks_lhs(40)
        boxes = [tuple(bl.bb) for bl in blocks]
        self.assertEqual(len(boxes), len(set(boxes)))
        self.assertLessEqual(len(blocks), 12)

    def test_blocks_sharing_fractures_kept_apart(self):
        # all twelve blocks touch the x=1, y=2, z=1 or x=3 fractures and differ
        # only in which domain boundaries bound them
        blocks = self.g.find_blocks_lhs(200)
        self.assertEqual(len(blocks), 12)
        self.assertAlmostEqual(sum(bl.V for bl in blocks), 32.)

    def test_one_stratum_per_axis(self):
        blocks = self.g.find_blocks_lhs(1)
        self.assertEqual(len(blocks), 1)

    def test_filter(self):
        # only blocks between x=1 and x=3 have four fracture faces
        blocks = self.g.find_blocks_lhs(200, filter_keywords=['4sides'])
        self.assertEqual(len(blocks), 4)
        self.assertTrue(all(tuple(bl.bb[:2]) == (1., 3.) for bl in blocks))
        self.assertEqual(
            self.g.find_blocks_lhs(200, filter_keywords=['5sides']), [])


class TestBlockHowLHS(unittest.TestCase):

    def _parser(self):
        p = argparse.ArgumentParser(exit_on_error=False)
        how = p.add_mutually_exclusive_group()
        how.add_argument('-n', nargs='?', dest='make_blocks', action=BlockHow)
        BlockHowLHS.add_to(how)
        p.add_argument('FILE', nargs='?')
        p.error = lambda msg: (_ for _ in ()).throw(ValueError(msg))
        return p

    def _nstrata(self, argv):
        """Parse argv and return the nstrata given to find_blocks_lhs"""
        seen = []
        class _Grid:
            def find_blocks_lhs(self, nstrata):
                seen.append(nstrata)
        self._parser().parse_args(argv).make_blocks(_Grid())
        return seen[0]

    def test_one_count(self):
        self.assertEqual(self._nstrata(['-lhs', '4']), 4)
        self.assertEqual(self._nstrata(['--lhs', '4']), 4)

    def test_three_counts(self):
        self.assertEqual(self._nstrata(['f.pkl', '-lhs', '2', '3', '4']),
                (2, 3, 4))

    def test_bad_counts(self):
        for argv in (['-lhs', '2', '3'], ['-lhs', '0'], ['-lhs', '1','2','3','4']):
            with self.subTest(argv=argv):
                with self.assertRaises(ValueError):
                    self._parser().parse_args(argv)

    def test_exclusive_with_n(self):
        with self.assertRaises((ValueError, argparse.ArgumentError)):
            self._parser().parse_args(['-n', '5', '-lhs', '3'])


if __name__ == '__main__':
    unittest.main()
