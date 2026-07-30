"""Tests related to OFracGrid objects and their operation

WARNING. VERY INCOMPLETE IN TERMS OF COVERAGE OF TESTS.
"""

import unittest
import unittest.mock
import itertools
import pickle
import warnings
from decimal import Decimal

import numpy as np

from ofrac import ofracs
from ofrac.ofracs import (OFrac, OFracArray, OFracGrid, as_nudge_triple)
from ofrac.ofracs import (CO_SCALE, AP_SCALE, STORE_DTYPE,
        _co2i, _i2co, _ap2i, _i2ap)


class TestOFracGrid(unittest.TestCase):

    def assertArrayEqual(self, a, b, msg=None):
        if not msg:
            msg = f'Failed. Arrays not equal:\n{a}\n{b}'
        #try:
        np.testing.assert_array_equal(a, b)
        #except AssertionError as e:
        #    raise self.failureException(msg)

    def _make_1x1_domain(self):
        return OFracGrid( domainSize=(1.,1.,1.), fx=[
            (0., 1., 0., 1., 0.5, 0.5, 0.001),
        ],)

    def test_choose_nodes(self):
        g = self._make_1x1_domain()

        with self.subTest('x=0 face'):
            ipm, ifx = g.choose_nodes_block('0, 0, 0, 1, 0, 1')
            self.assertArrayEqual(ipm, [0,2,4,6,8,10])
            self.assertArrayEqual(ifx, [4,6])

        with self.subTest('x=1 face'):
            ipm, ifx = g.choose_nodes_block('1, 1, 0, 1, 0, 1')
            self.assertArrayEqual(ipm-1, [0,2,4,6,8,10])
            self.assertArrayEqual(ifx-1, [4,6])

        g.addGridline(1,0.5)
        with self.subTest('1/2 x=0 face, extra gridline'):
            ipm, ifx = g.choose_nodes_block('0, 0, 0, 0.5, 0, 1')
            self.assertArrayEqual(ipm, [0,2,6,8,12,14])
            self.assertArrayEqual(ifx, [6,8])


    def test_ni2ng(self):

        g = self._make_1x1_domain()

        ni = np.arange(12)
        expng = np.array(list([i, j, k] for k, j, i in 
                    itertools.product(range(3), range(2), range(2))))

        self.assertArrayEqual(g.ni2ng(ni), expng)


    def test_getGridLines(self):

        g = self._make_1x1_domain()

        gl = g.getGridLines()
        self.assertArrayEqual(gl[0], [0., 1.])
        self.assertArrayEqual(gl[1], [0., 1.])
        self.assertArrayEqual(gl[2], [0., 0.5, 1.])

        self.assertArrayEqual(g.getGridLines(0), [0., 1.])
        self.assertArrayEqual(g.getGridLines('x'), [0., 1.])


    def test_truncate(self):

        g = self._make_1x1_domain()
        g.addFracture(OFrac(0., 1., 0.5, 0.5, 0., 1., 0.002))
        g.addFracture(OFrac(0.5, 0.5, 0., 1., 0., 1., 0.003))

        h = g.merge() # copy
        h.setDomainSize((0,0,0),(1,1,1))
        self.assertEqual(h.getFxCount(), 3)

        h.setDomainSize((0,0,0),(1,1,0.5))
        self.assertEqual(h.getFxCount(), 3)

        h.setDomainSize((0,0,0),(0.5,0.5,0.5))
        self.assertEqual(h.getFxCount(), 3)

        h.setDomainSize((0,0,0),(0.5,0.5,0.4))
        self.assertEqual(h.getFxCount(), 2)

        h.setDomainSize((0,0,0),(0.4,0.4,0.4))
        self.assertEqual(h.getFxCount(), 0)


class TestQuantization(unittest.TestCase):
    """Fractures are stored as integer counts of N_COORD_DIG / N_APERT_DIG"""

    def test_coord_roundtrip(self):
        for v, i, s in [
                ('0',            0,          '0.000'),
                ('0.001',        1,          '0.001'),
                ('-0.001',      -1,         '-0.001'),
                ('1.234',     1234,          '1.234'),
                ('-45.6',   -45600,        '-45.600'),
                ('1000000.999', 1000000999, '1000000.999'),
                ]:
            with self.subTest(v=v):
                self.assertEqual(_co2i(v), i)
                self.assertEqual(str(_i2co(i)), s)

    def test_aperture_roundtrip(self):
        for v, i, s in [
                ('0',          0, '0.000000'),
                ('0.000001',   1, '0.000001'),
                ('0.0001234', 123, '0.000123'),
                ]:
            with self.subTest(v=v):
                self.assertEqual(_ap2i(v), i)
                self.assertEqual(str(_i2ap(i)), s)

    def test_quantizes_to_precision(self):
        """Values finer than the stored precision are rounded, not kept"""
        self.assertEqual(_co2i(1.2345), 1234)
        self.assertEqual(_ap2i(1.23456789e-4), 123)

    def test_equality_is_exact(self):
        """Coordinates that should coincide must compare equal.

        This is what integer storage buys: it is why a fracture's orientation
        can be found by testing coordinates for equality, and why fracture
        faces land on grid lines.
        """
        g = OFracGrid(domainSize=(1.,1.,1.), fx=[
            (0., 1., 0., 1., 0.3, 0.3, 0.001),
            (0., 1., 0.3, 0.3, 0., 1., 0.001),
            (0.3, 0.3, 0., 1., 0., 1., 0.001),
        ])
        self.assertEqual(g.getFxCounts(), (1,1,1))
        for a in range(3):
            self.assertIn(Decimal('0.300'), g.getGridLines(a).tolist())


class TestStorageRange(unittest.TestCase):
    """32-bit storage spans +/-2147 km; leaving it must not wrap silently"""

    def _grid(self):
        return OFracGrid(domainOrigin=(0.,0.,0.), domainSize=(10.,10.,10.),
                fx=[(0., 10., 0., 10., 5., 5., 0.0001)])

    def test_range_covers_physical_domains(self):
        self.assertGreater(np.iinfo(STORE_DTYPE).max/CO_SCALE, 2e6)

    def test_input_beyond_range_is_rejected(self):
        with self.assertRaises(OverflowError):
            OFracGrid(fx=[(0., 3e6, 0., 10., 5., 5., 0.0001)])

    def test_scale_beyond_range_raises(self):
        g = self._grid()
        with self.assertRaises(ValueError):
            g.scale((1e6, 1., 1.))
        # the offending write never landed
        self.assertEqual(g._fx[0].d[1], Decimal('10.000'))

    def test_translate_beyond_range_raises(self):
        g = self._grid()
        with self.assertRaises(ValueError):
            g.translate((3e6, 0., 0.))
        self.assertEqual(g._fx[0].d[0], Decimal('0.000'))

    def test_ordinary_moves_are_unaffected(self):
        g = self._grid()
        g.scale((1000., 1000., 1000.))
        g.translate((100000., 0., 0.))
        self.assertEqual(g._fx[0].d[0], Decimal('100000.000'))
        self.assertEqual(g._fx[0].d[1], Decimal('110000.000'))


class TestOFracArray(unittest.TestCase):
    """The numpy-backed store that replaced the list of OFrac objects"""

    FX = [
        (0., 10., 0., 10., 5.,  5.,  0.0001),
        (0., 10., 3.,  3.,  0., 10.,  0.00012),
        (7.,  7.,  0., 10., 0., 10.,  0.000345),
    ]

    def _grid(self):
        return OFracGrid(domainOrigin=(0.,0.,0.), domainSize=(10.,10.,10.),
                fx=self.FX)

    def test_grid_uses_a_store(self):
        g = self._grid()
        self.assertIsInstance(g._fx, OFracArray)
        self.assertEqual(len(g._fx), 3)
        self.assertEqual(g._fx.coords.dtype, STORE_DTYPE)
        self.assertEqual(g._fx.coords.shape, (3,6))

    def test_helper_arrays(self):
        g = self._grid()
        self.assertArrayEqual(g._fx.perp_axes, [2,1,0])
        self.assertArrayEqual(g._fx.apertures,
                [100, 120, 345])
        self.assertEqual(g.getFxCounts(), (1,1,1))

    def test_perp_vals_are_derived(self):
        """Plane coordinates are computed from the coordinates, not stored"""
        g = self._grid()
        self.assertNotIn('_perpval', OFracArray.__slots__)
        self.assertArrayEqual(g._fx.perp_vals,
                [5*CO_SCALE, 3*CO_SCALE, 7*CO_SCALE])

        # ...and stay right after the coordinates move
        g.translate((1., 2., 3.))
        self.assertArrayEqual(g._fx.perp_vals,
                [8*CO_SCALE, 5*CO_SCALE, 8*CO_SCALE])

        # ...and agree with the per-fracture method
        self.assertEqual([f.determinePerpAxisVal()[1] for f in g.iterFracs()],
                [_i2co(v) for v in g._fx.perp_vals])

    def test_perp_vals_of_empty_store(self):
        s = OFracArray()
        self.assertEqual(s.perp_vals.size, 0)
        self.assertEqual(s.perp_axes.size, 0)

    def test_indexing_returns_live_views(self):
        g = self._grid()
        f = g._fx[1]
        self.assertIsInstance(f, OFrac)
        self.assertEqual(f.d[2], Decimal('3.000'))
        self.assertIs(f.myNet, g)

        # writing through the view writes into the store...
        f.d = (0., 10., 4., 4., 0., 10.)
        self.assertEqual(g._fx.coords[1,2], 4*CO_SCALE)
        # ...and keeps the helper arrays in step
        self.assertEqual(g._fx.perp_axes[1], 1)
        self.assertEqual(g._fx.perp_vals[1], 4*CO_SCALE)

        f.ap = 0.000999
        self.assertEqual(g._fx.apertures[1], 999)

    def test_view_rejects_wrong_length(self):
        g = self._grid()
        with self.assertRaises(ValueError):
            g._fx[0].d = (1., 2., 3.)

    def test_iteration_yields_distinct_fractures(self):
        g = self._grid()
        seen = [str(f) for f in g.iterFracs()]
        self.assertEqual(len(seen), 3)
        self.assertEqual(len(set(seen)), 3)

    def test_delete(self):
        g = self._grid()
        g._fx.delete([0,2])
        self.assertEqual(len(g._fx), 1)
        self.assertEqual(g._fx[0].d[2], Decimal('3.000'))
        self.assertArrayEqual(g._fx.perp_axes, [1])

    def test_delete_is_idempotent_per_index(self):
        g = self._grid()
        g._fx.delete([1,1,1])
        self.assertEqual(len(g._fx), 2)

    def test_del_slice(self):
        g = self._grid()
        del g._fx[1:]
        self.assertEqual(len(g._fx), 1)

    def test_setitem_copies_a_row(self):
        g = self._grid()
        g._fx[0] = g._fx[2]
        self.assertEqual(str(g._fx[0]), str(g._fx[2]))
        self.assertEqual(g._fx.perp_axes[0], 0)

    def test_out_of_range(self):
        g = self._grid()
        for i in (3, -4):
            with self.subTest(i=i), self.assertRaises(IndexError):
                g._fx[i]
        self.assertEqual(str(g._fx[-1]), str(g._fx[2]))

    def test_growth_preserves_data(self):
        """Appending past the allocation must keep every earlier fracture"""
        s = OFracArray()
        expect = []
        for i in range(40):
            s.append_values(0., 10., 0., 10., float(i), float(i), 1e-4)
            expect.append(Decimal(i).quantize(Decimal('0.001')))
        self.assertEqual(len(s), 40)
        self.assertArrayEqual(s.perp_vals, [i*CO_SCALE for i in range(40)])
        self.assertEqual([f.d[4] for f in s], expect)

    def test_standalone_fracture_has_its_own_store(self):
        f = OFrac(0., 1., 0.5, 0.5, 0., 1., 0.002)
        self.assertIsNone(f.myNet)
        self.assertEqual(f.determinePerpAxisVal(), (1, Decimal('0.500')))

        g = self._grid()
        g.addFracture(f)
        self.assertEqual(g.getFxCount(), 4)
        # the grid holds a copy, not the original's row
        self.assertIsNot(g._fx[3]._store, f._store)
        self.assertEqual(str(g._fx[3]), str(f))

    def test_copy_construction(self):
        g = self._grid()
        f = OFrac(fromOFrac=g._fx[0])
        self.assertEqual(str(f), str(g._fx[0]))
        self.assertIs(f.myNet, g)
        # the copy is independent
        f.d = (0., 1., 0., 1., 9., 9.)
        self.assertEqual(g._fx[0].d[4], Decimal('5.000'))


class TestVectorizedAccessors(unittest.TestCase):

    def _grid(self):
        return OFracGrid(domainOrigin=(0.,0.,0.), domainSize=(10.,10.,10.), fx=[
            (0., 10., 0., 10., 5.,  5.,  0.0001),
            (0., 10., 3.,  3.,  0., 10.,  0.00012),
            (7.,  7.,  0., 10., 0., 10.,  0.000345),
        ])

    def assertArrayEqual(self, a, b):
        np.testing.assert_array_equal(a, b)

    def test_coordinates(self):
        g = self._grid()
        self.assertArrayEqual(g.getFxCoordinates(), [
            [0., 10., 0., 10., 5., 5.],
            [0., 10., 3., 3., 0., 10.],
            [7., 7., 0., 10., 0., 10.],
        ])

    def test_apertures(self):
        g = self._grid()
        np.testing.assert_allclose(g.getFxApertures(),
                [0.0001, 0.00012, 0.000345])

    def test_perp_axes(self):
        g = self._grid()
        self.assertArrayEqual(g.getFxPerpAxes(), [2,1,0])
        # ...and agrees with the per-fracture method
        self.assertArrayEqual(g.getFxPerpAxes(),
            [OFrac.determineFracOrientation(f) for f in g.iterFracs()])

    def test_accessors_return_copies(self):
        g = self._grid()
        c = g.getFxCoordinates()
        c[0,0] = -999.
        self.assertEqual(g._fx[0].d[0], Decimal('0.000'))


class TestFxFiltering(unittest.TestCase):
    """Masks and measures over whole fracture sets"""

    def _grid(self):
        # a 10m cube holding one fracture of each orientation
        return OFracGrid(domainOrigin=(0.,0.,0.), domainSize=(10.,10.,10.), fx=[
            (0., 10., 0., 10., 5.,  5.,  0.0001),   # xy, at z=5
            (0.,  4., 3.,  3., 0., 10.,  0.00012),  # xz, at y=3
            (7.,  7., 0., 10., 0.,  6.,  0.000345), # yz, at x=7
        ])

    def assertArrayEqual(self, a, b):
        np.testing.assert_array_equal(a, b)

    def test_mask_in(self):
        g = self._grid()

        with self.subTest('the whole domain keeps everything'):
            self.assertArrayEqual(
                g.getFxMaskIn((0.,0.,0.), (10.,10.,10.)), [True]*3)

        with self.subTest('a corner of the domain'):
            self.assertArrayEqual(
                g.getFxMaskIn((0.,0.,0.), (5.,5.,4.)), [False,True,False])

        with self.subTest('touching a face counts as intersecting'):
            # the xy fracture's plane is z=5, which is this box's z-maximum
            self.assertArrayEqual(
                g.getFxMaskIn((0.,0.,0.), (10.,10.,5.)), [True,True,True])
            # ...and a hair above that plane, only that fracture is left out
            self.assertArrayEqual(
                g.getFxMaskIn((0.,0.,5.001), (10.,10.,10.)),
                [False,True,True])

        with self.subTest('open sides'):
            inf = float('inf')
            self.assertArrayEqual(
                g.getFxMaskIn((-inf,-inf,-inf), (inf,inf,inf)), [True]*3)
            self.assertArrayEqual(
                g.getFxMaskIn((6.,-inf,-inf), (inf,inf,inf)),
                [True,False,True])

    def test_mask_perp_to(self):
        g = self._grid()
        self.assertArrayEqual(g.getFxMaskPerpTo(0), [False,False,True])
        self.assertArrayEqual(g.getFxMaskPerpTo('y'), [False,True,False])
        self.assertArrayEqual(g.getFxMaskPerpTo('Z'), [True,False,False])
        with self.assertRaises(ValueError):
            g.getFxMaskPerpTo('w')

    def test_mask_spanning(self):
        g = self._grid()

        with self.subTest('half-open: the from-face is in, the to-face is not'):
            self.assertArrayEqual(g.getFxMaskSpanning('x', 0.), [True,True,False])
            self.assertArrayEqual(g.getFxMaskSpanning('x', 4.), [True,False,False])
            self.assertArrayEqual(g.getFxMaskSpanning('x', 10.), [False]*3)

        with self.subTest('a fracture never spans its own perpendicular axis'):
            self.assertFalse(g.getFxMaskSpanning('z', 5.)[0])
            self.assertFalse(g.getFxMaskSpanning('y', 3.)[1])
            self.assertFalse(g.getFxMaskSpanning('x', 7.)[2])

    def test_queries_are_not_quantized(self):
        """A query value finer than N_COORD_DIG is not rounded onto a fracture"""
        g = self._grid()

        # the yz fracture ends at z=6; a scan plane a ten-thousandth below it
        # is still inside, and one that far above it is not
        self.assertTrue(g.getFxMaskSpanning('z', 5.9999)[2])
        self.assertFalse(g.getFxMaskSpanning('z', 6.0001)[2])

    def test_queries_land_on_stored_values(self):
        """A query written as a stored coordinate compares equal to it

        Scaling such a value by CO_SCALE in floating point lands a bit below
        or above the stored integer for a few per cent of coordinates, which
        is the difference between a face touching a query and missing it.
        """
        g = OFracGrid(domainSize=(10.,10.,10.), fx=[
            (1.001, 9.999, 1.001, 9.999, 5., 5., 0.0001),
        ])

        with self.subTest('a box ending on the fracture still touches it'):
            self.assertArrayEqual(
                g.getFxMaskIn((0.,0.,0.), (1.001,10.,10.)), [True])

        with self.subTest('a box starting on the far face still touches it'):
            self.assertArrayEqual(
                g.getFxMaskIn((9.999,0.,0.), (10.,10.,10.)), [True])

        with self.subTest('a line on the from-face crosses the fracture'):
            self.assertArrayEqual(
                g.getFxMaskAlongLine('z', 1.001, 1.001), [True])

        with self.subTest('...but one on the to-face does not'):
            self.assertArrayEqual(
                g.getFxMaskAlongLine('z', 9.999, 1.001), [False])

    def test_mask_along_line(self):
        g = self._grid()

        with self.subTest('a line along z crosses the xy fracture'):
            # at (x,y), in ascending axis order
            self.assertArrayEqual(
                g.getFxMaskAlongLine('z', 1., 1.), [True,False,False])

        with self.subTest('a line along y crosses the xz fracture'):
            # at (x,z); x=1 is within the xz fracture's 0->4 extent, x=5 is not
            self.assertArrayEqual(
                g.getFxMaskAlongLine('y', 1., 1.), [False,True,False])
            self.assertArrayEqual(
                g.getFxMaskAlongLine('y', 5., 1.), [False]*3)

        with self.subTest('a line along x crosses the yz fracture'):
            self.assertArrayEqual(
                g.getFxMaskAlongLine('x', 1., 1.), [False,False,True])
            # ...but not above its z extent
            self.assertArrayEqual(
                g.getFxMaskAlongLine('x', 1., 7.), [False]*3)

    def test_lengths(self):
        g = self._grid()

        self.assertArrayEqual(g.getFxLengths(), [
            [10., 10.,  0.],
            [ 4.,  0., 10.],
            [ 0., 10.,  6.],
        ])

    def test_lengths_clipped(self):
        g = self._grid()

        with self.subTest('clipping shortens fractures'):
            self.assertArrayEqual(
                g.getFxLengths(clip_to=((0.,0.,0.), (5.,5.,5.))), [
                    [5., 5., 0.],
                    [4., 0., 5.],
                    [0., 5., 5.],
                ])

        with self.subTest('a fracture clipped out of existence has no length'):
            # the box stops short of the yz fracture, at x=7
            self.assertArrayEqual(
                g.getFxLengths(clip_to=((0.,0.,0.), (6.,10.,10.)))[2],
                [0., 10., 6.])

    def test_plane_axes(self):
        g = self._grid()

        self.assertArrayEqual(g.getFxPlaneAxes(), [
            [ True,  True, False],   # xy
            [ True, False,  True],   # xz
            [False,  True,  True],   # yz
        ])

        with self.subTest('it selects the lengths of a fracture'):
            (ext, inPlane) = (g.getFxLengths(), g.getFxPlaneAxes())
            self.assertArrayEqual(ext[inPlane[:, 0], 0], [10., 4.])
            self.assertTrue(np.all(ext[~inPlane] == 0.))

    def test_length_bins(self):
        g = self._grid()  # extents of (10,10,0), (4,0,10) and (0,10,6)

        edges = [5., 8.]
        b = g.getFxLengthBins(edges)

        self.assertArrayEqual(b, [
            [2, 2, 0],
            [0, 0, 2],
            [0, 2, 1],
        ])

        with self.subTest('an extent equal to an edge is in the bin above'):
            self.assertArrayEqual(g.getFxLengthBins([4., 6., 10.])[:, 0],
                    [3, 1, 0])

        with self.subTest('clipping moves fractures between bins'):
            self.assertArrayEqual(
                g.getFxLengthBins(edges, clip_to=((0.,0.,0.),(5.,5.,5.)))[:, 0],
                [1, 0, 0])

    def test_areas_and_volumes(self):
        g = self._grid()

        np.testing.assert_allclose(g.getFxAreas(), [100., 40., 60.])
        np.testing.assert_allclose(g.getFxVolumes(),
                [100.*0.0001, 40.*0.00012, 60.*0.000345])
        np.testing.assert_allclose(
            g.getFxAreas(clip_to=((0.,0.,0.), (5.,5.,5.))), [25., 20., 25.])

    def test_perp_vals(self):
        g = self._grid()
        self.assertArrayEqual(g.getFxPerpVals(), [5., 3., 7.])

    def test_aperture_bins(self):
        g = self._grid()  # apertures of 100, 120 and 345 microns

        um = Decimal('1e-6')
        edges = [ Decimal(v)*um for v in ('50','200','400') ]

        self.assertArrayEqual(g.getFxApertureBins(edges), [1,1,2])
        self.assertArrayEqual(
            np.bincount(g.getFxApertureBins(edges), minlength=len(edges)+1),
            [0,2,1,0])

        with self.subTest('no edges is one bin holding everything'):
            self.assertArrayEqual(g.getFxApertureBins([]), [0,0,0])

    def test_aperture_bins_are_exact_on_an_edge(self):
        """An aperture equal to an edge belongs to the bin above it

        Comparing float metres would not reliably do this: scaling a stored
        aperture to metres lands a bit below the edge for many values.
        """
        g = self._grid()
        aps = [Decimal('0.000100'), Decimal('0.000120'), Decimal('0.000345')]

        for ap in aps:
            with self.subTest(f'an edge at {ap}'):
                # the fracture of exactly this aperture is above the edge...
                self.assertEqual(
                    list(g.getFxApertureBins([ap])).count(1),
                    sum(1 for a in aps if a >= ap))
                # ...and below the next digit up
                self.assertEqual(
                    list(g.getFxApertureBins([ap+Decimal('1e-6')])).count(1),
                    sum(1 for a in aps if a > ap))

    def test_subset(self):
        g = self._grid()
        h = g.subsetFx(g.getFxMaskPerpTo('z'))

        self.assertEqual(h.getFxCount(), 1)
        self.assertEqual(h.getFxCounts(), (0,0,1))
        self.assertEqual(str(h._fx[0]), str(g._fx[0]))

        with self.subTest('the domain is kept, but grid lines are re-made'):
            self.assertEqual(h.getDomainStart(), g.getDomainStart())
            self.assertEqual(h.getDomainEnd(), g.getDomainEnd())
            # the y=3 and x=7 fracture faces went with the fractures dropped
            self.assertArrayEqual(h.getGridLines('y'), [0., 10.])
            self.assertArrayEqual(h.getGridLines('x'), [0., 10.])
            self.assertArrayEqual(h.getGridLines('z'), [0., 5., 10.])

    def test_subset_by_index(self):
        g = self._grid()
        h = g.subsetFx([0,2])

        self.assertEqual(h.getFxCount(), 2)
        self.assertArrayEqual(h.getFxPerpAxes(), [2,0])

    def test_subset_of_nothing(self):
        g = self._grid()
        h = g.subsetFx(np.zeros(3, dtype=bool))

        self.assertEqual(h.getFxCount(), 0)
        self.assertEqual(h.getFxCounts(), (0,0,0))

    def test_subset_is_a_copy(self):
        g = self._grid()
        h = g.subsetFx(np.ones(3, dtype=bool))

        h._fx[0].d = (0., 1., 0., 1., 1., 1.)
        self.assertArrayEqual(g.getFxCoordinates()[0], [0.,10.,0.,10.,5.,5.])
        self.assertIs(h._fx[0].myNet, h)

    def test_masks_index_the_accessors(self):
        """The masks and the vectorized accessors are in the same order"""
        g = self._grid()
        m = g.getFxMaskPerpTo('x')

        self.assertArrayEqual(g.getFxCoordinates()[m],
                [[7., 7., 0., 10., 0., 6.]])
        np.testing.assert_allclose(g.getFxApertures()[m], [0.000345])


class TestStoreStaysConsistent(unittest.TestCase):
    """Operations that move fractures must keep the helper arrays correct"""

    def _grid(self):
        return OFracGrid(domainOrigin=(0.,0.,0.), domainSize=(10.,10.,10.), fx=[
            (0., 10., 0., 10., 5.,  5.,  0.0001),
            (0., 10., 3.,  3.,  0., 10.,  0.00012),
            (7.,  7.,  0., 10., 0., 10.,  0.000345),
        ])

    def _assertConsistent(self, g):
        """Helper arrays must match what the coordinates say"""
        d = g._fx.coords
        for i in range(len(g._fx)):
            a = int(g._fx.perp_axes[i])
            self.assertGreaterEqual(a, 0)
            self.assertEqual(d[i,2*a], d[i,2*a+1])
            self.assertEqual(g._fx.perp_vals[i], d[i,2*a])
        self.assertEqual(sum(g.getFxCounts()), g.getFxCount())

    def test_translate(self):
        g = self._grid()
        g.translate((1.5, 0., -2.))
        self._assertConsistent(g)
        self.assertEqual(g._fx[2].d[0], Decimal('8.500'))
        self.assertEqual(g._fx[0].determinePerpAxisVal(), (2, Decimal('3.000')))

    def test_scale(self):
        g = self._grid()
        g.scale((2., 1., 0.5))
        self._assertConsistent(g)
        self.assertEqual(g._fx[2].d[0], Decimal('14.000'))
        self.assertEqual(g._fx[0].determinePerpAxisVal(), (2, Decimal('2.500')))

    def test_setDomainSize(self):
        g = self._grid()
        g.setDomainSize((0.,0.,0.), (6.,6.,6.))
        self._assertConsistent(g)

    def test_delFracture(self):
        g = self._grid()
        g.delFracture([1])
        self._assertConsistent(g)
        self.assertEqual(g.getFxCounts(), (1,0,1))

    def test_merge(self):
        g = self._grid()
        h = g.merge(self._grid())
        self._assertConsistent(h)
        self.assertEqual(h.getFxCount(), 6)
        self.assertEqual(h.getFxCounts(), (2,2,2))

    def test_nudge_drops_keep_counts_correct(self):
        """Fractures lost to nudging must leave the orientation counts right"""
        g = OFracGrid(domainOrigin=(0.,0.,0.), domainSize=(10.,10.,10.), fx=[
            (0., 10., 0., 10., 5., 5., 0.0001),
            (0., 0.4, 3., 3., 0., 0.4, 0.00012),   # collapses when nudged
        ])
        g.collapse_policy = 'omit'
        g.nudgeAll(1.0)
        self.assertEqual(g.getFxCount(), 1)
        self._assertConsistent(g)
        self.assertEqual(g.getFxCounts(), (0,0,1))


class TestPickling(unittest.TestCase):

    def _grid(self):
        return OFracGrid(domainOrigin=(0.,0.,0.), domainSize=(10.,10.,10.), fx=[
            (0., 10., 0., 10., 5.,  5.,  0.0001),
            (0., 10., 3.,  3.,  0., 10.,  0.00012),
        ])

    def test_roundtrip(self):
        g = self._grid()
        g.metadata['k'] = 'v'
        h = pickle.loads(pickle.dumps(g, pickle.HIGHEST_PROTOCOL))

        self.assertIsInstance(h._fx, OFracArray)
        self.assertEqual(str(h), str(g))
        self.assertEqual(h.metadata, {'k':'v'})
        np.testing.assert_array_equal(h._fx.coords, g._fx.coords)
        # fractures point back at the restored grid, not the original
        self.assertIs(h._fx[0].myNet, h)

    def test_spare_capacity_is_not_pickled(self):
        g = self._grid()
        g._fx.reserve(1000)
        h = pickle.loads(pickle.dumps(g, pickle.HIGHEST_PROTOCOL))
        self.assertEqual(len(h._fx), 2)
        self.assertEqual(h._fx._d.shape, (2,6))

    def test_legacy_list_of_ofrac_is_converted(self):
        """Networks pickled before fractures were array-backed still load"""

        class _LegacyOFrac:
            """Stands in for the OFrac that carried its own Decimals"""
            def __init__(self, d, ap):
                self.d = d
                self.ap = ap

        legacy = [
            _LegacyOFrac(tuple(map(Decimal, ('0','10','0','10','5','5'))),
                Decimal('0.0001')),
            _LegacyOFrac(tuple(map(Decimal, ('0','10','3','3','0','10'))),
                Decimal('0.00012')),
        ]

        g = self._grid()
        state = dict(g.__dict__)
        state['_fx'] = legacy

        h = OFracGrid.__new__(OFracGrid)
        h.__setstate__(state)

        self.assertIsInstance(h._fx, OFracArray)
        self.assertEqual(len(h._fx), 2)
        self.assertEqual(h._fx[0].d[1], Decimal('10.000'))
        self.assertEqual(h._fx[1].ap, Decimal('0.000120'))
        self.assertArrayEqual(h._fx.perp_axes, [2,1])
        self.assertIs(h._fx[0].myNet, h)

    def test_legacy_ofrac_setstate(self):
        """An OFrac unpickled from the old slot layout gets its own store"""
        f = OFrac.__new__(OFrac)
        f.__setstate__((None, {
            'd': tuple(map(Decimal, ('0','10','0','10','5','5'))),
            'ap': Decimal('0.0001'),
            'myNet': None,
        }))
        self.assertEqual(f.d[1], Decimal('10.000'))
        self.assertEqual(f.ap, Decimal('0.000100'))
        self.assertEqual(f.determinePerpAxisVal(), (2, Decimal('5.000')))
        self.assertIsNone(f.myNet)

    def assertArrayEqual(self, a, b):
        np.testing.assert_array_equal(a, b)


# shared by the classes above that did not define their own
TestOFracArray.assertArrayEqual = TestOFracGrid.assertArrayEqual


if __name__ == '__main__':
    unittest.main()


class TestAnisotropicNudging(unittest.TestCase):
    """`nudgeAll` / `OFrac.nudge` accept a scalar or a per-axis triple.

    Anisotropic nudging exists so one axis can be conditioned without dragging
    the others finer: refining z alone leaves the x grid at its own resolution,
    which an isotropic increment cannot do.
    """

    def _net(self):
        # one x-normal and one z-normal fracture, both off any round increment
        return OFracGrid(
            domainOrigin=(0, 0, 0), domainSize=(10, 1, 10),
            fx=[OFrac(1.234, 1.234, 0.0, 1.0, 2.0, 8.0, 1e-4),
                OFrac(1.0, 9.0, 0.0, 1.0, 3.456, 3.456, 1e-4)])

    def test_scalar_is_isotropic(self):
        g = self._net()
        g.nudgeAll(0.1)
        d = [tuple(float(v) for v in f.d) for f in g.iterFracs()]
        self.assertAlmostEqual(d[0][0], 1.2)
        self.assertAlmostEqual(d[1][4], 3.5)

    def test_triple_nudges_only_the_named_axis(self):
        g = self._net()
        g.nudgeAll([0, 0, 0.1])
        d = [tuple(float(v) for v in f.d) for f in g.iterFracs()]
        self.assertAlmostEqual(d[0][0], 1.234, msg='x must be untouched')
        self.assertAlmostEqual(d[1][4], 3.5, msg='z must be nudged')

    def test_triple_leaves_untouched_axis_gridlines_alone(self):
        g = self._net()
        before = [float(v) for v in g.iterGridLines(0)]
        g.nudgeAll([0, 0, 0.1])
        self.assertEqual(before, [float(v) for v in g.iterGridLines(0)])

    def test_all_zero_is_a_no_op(self):
        g = self._net()
        before = [tuple(float(v) for v in f.d) for f in g.iterFracs()]
        g.nudgeAll([0, 0, 0])
        self.assertEqual(before, [tuple(float(v) for v in f.d) for f in g.iterFracs()])

    def test_wrong_length_is_rejected(self):
        with self.assertRaises(ValueError):
            as_nudge_triple([0.1, 0.1])

    def test_scalar_and_equal_triple_agree(self):
        a, b = self._net(), self._net()
        a.nudgeAll(0.1)
        b.nudgeAll([0.1, 0.1, 0.1])
        self.assertEqual([tuple(f.d) for f in a.iterFracs()],
                         [tuple(f.d) for f in b.iterFracs()])


class TestConnectivity(unittest.TestCase):
    """Fracture intersections, the graph they form, and its clusters.

    A fracture conducts into another only where the two meet, so the connected
    components of the intersection graph are what a filter for "fractures that
    are not part of the network" has to be built on.
    """

    def _net(self):
        """Three clusters: {0,1,2} crossing, {3,4} crossing, and a lone 5"""
        return OFracGrid(domainSize=(10., 10., 10.), fx=[
            (0., 5., 0., 5., 2., 2., 1e-4),   # 0: perp z, at z=2
            (1., 1., 0., 5., 0., 5., 1e-4),   # 1: perp x, crosses 0
            (0., 5., 3., 3., 2., 6., 1e-4),   # 2: perp y, lands on 0, crosses 1
            (8., 9., 8., 9., 7., 7., 1e-4),   # 3: perp z
            (8.5, 8.5, 8., 9., 6., 8., 1e-4), # 4: perp x, crosses 3
            (0., 1., 9., 9., 0., 1., 1e-4),   # 5: perp y, meets nothing
        ])

    def test_intersections(self):
        p = self._net().getFxIntersections()
        self.assertEqual(p.tolist(), [[0, 1], [0, 2], [1, 2], [3, 4]])

    def test_a_fracture_landing_on_another_counts(self):
        # fracture 2 terminates on fracture 0's plane rather than crossing it,
        # which is how an orthogonal network is usually generated
        self.assertIn([0, 2], self._net().getFxIntersections().tolist())

    def test_same_orientation_never_intersects(self):
        # two z-normal fractures in one plane, sharing an edge
        g = OFracGrid(domainSize=(10., 10., 10.), fx=[
            (0., 5., 0., 5., 2., 2., 1e-4),
            (5., 9., 0., 5., 2., 2., 1e-4)])

        self.assertEqual(len(g.getFxIntersections()), 0)
        self.assertEqual(g.getFxIntersections(include_coplanar=True).tolist(),
            [[0, 1]])

    def test_intersection_counts(self):
        self.assertEqual(self._net().getFxIntersectionCounts().tolist(),
            [2, 2, 2, 1, 1, 0])

    def test_cluster_labels_are_biggest_first(self):
        g = self._net()
        self.assertEqual(g.getFxClusterLabels().tolist(), [0, 0, 0, 1, 1, 2])
        self.assertEqual(g.getFxClusterSizes().tolist(), [3, 2, 1])

    def test_mask_connected_drops_the_lone_fracture(self):
        g = self._net()
        self.assertEqual(g.getFxMaskConnected().tolist(),
            [True]*5 + [False])
        self.assertEqual(g.getFxMaskConnected(min_cluster_size=3).tolist(),
            [True]*3 + [False]*3)

    def test_subsetting_by_connectedness(self):
        g = self._net()
        sub = g.subsetFx(g.getFxMaskConnected())
        self.assertEqual(sub.getFxCount(), 5)
        self.assertEqual(len(sub.getFxIntersections()), 4)

    def test_mask_connected_to_seeds(self):
        g = self._net()
        # everything reachable from the fractures in the first cluster's corner
        seed = g.getFxMaskIn((1., 0., 0.), (1., 5., 5.))
        self.assertEqual(g.getFxMaskConnectedTo(seed).tolist(),
            [True]*3 + [False]*3)

    def test_mask_connected_to_accepts_indices(self):
        g = self._net()
        self.assertEqual(g.getFxMaskConnectedTo([3]).tolist(),
            [False]*3 + [True]*2 + [False])

    def test_adjacency_matrix(self):
        a = self._net().getFxAdjacency().toarray()

        self.assertEqual(a.shape, (6, 6))
        np.testing.assert_array_equal(a, a.T)
        self.assertEqual(a.diagonal().sum(), 0, 'no fracture meets itself')
        self.assertEqual(a[0].tolist(),
            [False, True, True, False, False, False])
        self.assertEqual(a[5].sum(), 0)

    def test_blocking_does_not_change_the_answer(self):
        g = self._net()
        # a block size of one pair forces the widest possible chunking
        np.testing.assert_array_equal(
            g._fx.intersection_pairs(max_block=1), g.getFxIntersections())

    def test_percolation_across_a_spanning_cluster(self):
        g = self._net()
        # cluster {0,1,2} reaches xmin (frac 0/1), ymin (frac 0/1) and zmin
        # (frac 1), but none of the max faces
        self.assertTrue(g.test_percolation(
            [True, False, True, False, False, False]))
        self.assertTrue(g.test_percolation(
            [True, False, False, False, True, False]))
        self.assertFalse(g.test_percolation(
            [False, True, False, True, False, False]))
        self.assertFalse(g.test_percolation(
            [True, True, False, False, False, False]))

    def test_percolation_needs_two_faces(self):
        g = self._net()
        self.assertTrue(g.test_percolation([True, False, False, False, False, False]))
        self.assertTrue(g.test_percolation([False]*6))

    def test_percolation_rejects_wrong_length(self):
        with self.assertRaises(ValueError):
            self._net().test_percolation([True]*5)

    def test_percolation_of_empty_network(self):
        g = OFracGrid(domainSize=(1., 1., 1.), fx=[])
        self.assertFalse(g.test_percolation(
            [True, False, True, False, False, False]))

    def test_empty_network(self):
        g = OFracGrid(domainSize=(1., 1., 1.), fx=[])

        self.assertEqual(g.getFxIntersections().shape, (0, 2))
        self.assertEqual(g.getFxClusterLabels().tolist(), [])
        self.assertEqual(g.getFxClusterSizes().tolist(), [])
        self.assertEqual(g.getFxMaskConnected().tolist(), [])
        self.assertEqual(g.getFxMaskConnectedTo([]).tolist(), [])

    def test_agrees_with_pairwise_comparison(self):
        """The vectorized sweep must match a plain loop over every pair"""
        rng = np.random.default_rng(11)
        fx = []
        while len(fx) < 150:
            p = int(rng.integers(0, 3))
            c = np.round(rng.uniform(0., 20., 3), 3)
            d = []
            for a in range(3):
                if a == p:
                    d += [c[a], c[a]]
                else:
                    d += [c[a], min(20., c[a]+round(rng.uniform(.5, 6.), 3))]
            if all(d[2*a+1]-d[2*a] >= 0.001 for a in range(3) if a != p):
                fx.append(tuple(d)+(1e-4,))

        g = OFracGrid(domainSize=(20., 20., 20.), fx=fx)
        d = g._fx.coords
        perp = g._fx.perp_axes

        expect = [[i, j]
            for i, j in itertools.combinations(range(len(d)), 2)
            if perp[i] != perp[j]
                and all(d[i][2*a] <= d[j][2*a+1] and d[j][2*a] <= d[i][2*a+1]
                    for a in range(3))]

        self.assertTrue(expect, 'the sample network must have intersections')
        self.assertEqual(g.getFxIntersections().tolist(), expect)

    def test_adjacency_lower_triangle(self):
        g = self._net()
        (full, low) = (g.getFxAdjacency().toarray(),
            g.getFxAdjacency(form='lower').toarray())

        self.assertEqual(low.sum(), len(g.getFxIntersections()),
            'each intersection is recorded once')
        self.assertEqual(low.sum()*2, full.sum())
        np.testing.assert_array_equal(low, np.tril(full, -1))
        # the pair (0,2) is held against the later fracture
        self.assertTrue(low[2, 0])
        self.assertFalse(low[0, 2])

    def test_adjacency_rejects_an_unknown_form(self):
        with self.assertRaises(ValueError):
            self._net().getFxAdjacency(form='diagonal')

    def test_either_form_gives_the_same_components(self):
        from scipy.sparse.csgraph import connected_components
        g = self._net()
        counts = set(connected_components(
                g.getFxAdjacency(form=f), directed=False)[0]
            for f in ('full', 'lower'))
        self.assertEqual(counts, {3})


class TestLargeNetworkWarning(unittest.TestCase):
    """A quadratic search over a big network should say so."""

    def _net(self, n):
        # n parallel fractures: no intersections, but every pair is compared
        return OFracGrid(domainSize=(1., 1., float(n)), fx=[
            (0., 1., 0., 1., float(k), float(k), 1e-4) for k in range(n)])

    def test_warns_above_the_threshold(self):
        g = self._net(40)
        with unittest.mock.patch.object(ofracs, 'LARGE_NETWORK', 10):
            with self.assertWarns(ofracs.LargeNetworkWarning) as w:
                g.getFxIntersections()
        self.assertIn('Sorting', str(w.warning),
            'the warning should say what to implement')

    def test_quiet_below_the_threshold(self):
        g = self._net(40)
        with unittest.mock.patch.object(ofracs, 'LARGE_NETWORK', 100):
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                g.getFxIntersections()

    def test_none_silences_it(self):
        g = self._net(40)
        with unittest.mock.patch.object(ofracs, 'LARGE_NETWORK', None):
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                g.getFxIntersections()

    def test_warns_through_every_entry_point(self):
        g = self._net(40)
        for call in (lambda: g.getFxIntersections(),
                     lambda: g.getFxAdjacency(),
                     lambda: g.getFxMaskConnected(),
                     lambda: g._fx.intersection_pairs()):
            with self.subTest(call=call):
                with unittest.mock.patch.object(ofracs, 'LARGE_NETWORK', 10):
                    with self.assertWarns(ofracs.LargeNetworkWarning) as w:
                        call()
                # the reminder must name the caller's line, not the library's
                self.assertEqual(w.filename, __file__)
