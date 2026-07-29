"""Tests for the scan-line/scan-plane placement rules in ofracstats-pcalc.

The P-system measures are expectations over a randomly placed probe, so the
closed-form (`--sampling exact`) results are checked against networks whose
answers can be worked out by hand, and the sampled rules are checked for
agreement with them.

The fixture network, in a zone 10 x 1 x 2 m:

    A   plane normal to x at x=5, spanning y 0..1 and z 0..2  (spans the zone)
    B   plane normal to x at x=7, spanning y 0..1 and z 0..1  (half the height)
    C   plane normal to z at z=1, spanning x 0..10 and y 0..1 (spans the zone)

P10 along x: the placement plane is (y, z), of area 1 x 2 = 2.  A covers all of
it and B covers half, so a scan line crosses 1.5 fractures on average over a
scan length of 10 m -> P10-x = 0.15 /m.

P10 along z: the placement plane is (x, y), of area 10 x 1 = 10.  Only C is
normal to z and it covers the whole plane, so every scan line crosses it
exactly once over a scan length of 2 m -> P10-z = 0.5 /m.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent                      # .../libdev/ofrac
if str(_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT.parent))

from ofrac.ofracs import OFrac, OFracGrid  # noqa: E402


def _load_pcalc():
    """Import the hyphenated script as a module (its CLI is __main__-guarded)."""
    spec = importlib.util.spec_from_file_location(
        'ofracstats_pcalc', _ROOT / 'ofracstats-pcalc.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pcalc = _load_pcalc()

AP = 1.0e-4


def _fixture_zone():
    """Return (FractureZone, SpatialZone) for the network in the module docstring."""
    net = OFracGrid(domainOrigin=(0, 0, 0), domainSize=(10, 1, 2))
    for vals in (
            (5, 5, 0, 1, 0, 2, AP),      # A: normal to x, spans the zone
            (7, 7, 0, 1, 0, 1, AP),      # B: normal to x, half the height
            (0, 10, 0, 1, 1, 1, AP),     # C: normal to z, spans the zone
    ):
        net.addFracture(OFrac(*vals))

    zn = pcalc.SpatialZone(start=(0, 0, 0), end=(10, 1, 2))
    return pcalc.FractureZone(zn, net, 64), zn


class _SamplingMode:
    """Context manager setting the module-level placement rule."""

    def __init__(self, mode, seed=12345):
        self.mode, self.seed = mode, seed

    def __enter__(self):
        self._prev = (pcalc.__SAMPLING__, pcalc.__RNG__)
        pcalc.__SAMPLING__ = self.mode
        pcalc.__RNG__ = np.random.default_rng(self.seed)
        return self

    def __exit__(self, *exc):
        pcalc.__SAMPLING__, pcalc.__RNG__ = self._prev
        return False


class TestPlacementHelpers(unittest.TestCase):

    def test_overlap_clips_to_the_zone(self):
        a = np.array([0.0, -5.0, 8.0, 20.0])
        b = np.array([2.0, 5.0, 12.0, 30.0])
        got = pcalc._overlap(a, b, 0.0, 10.0)
        np.testing.assert_allclose(got, [2.0, 5.0, 2.0, 0.0])

    def test_overlap_never_negative(self):
        """A fracture wholly outside the zone contributes nothing, not a debt."""
        got = pcalc._overlap(np.array([20.0]), np.array([30.0]), 0.0, 10.0)
        self.assertEqual(float(got[0]), 0.0)

    def test_strata_places_exactly_one_point_per_stratum(self):
        with _SamplingMode('lhs'):
            n = 50
            pts = pcalc._strata(0.0, 10.0, n)
            self.assertEqual(len(pts), n)
            self.assertTrue(((pts >= 0.0) & (pts < 10.0)).all())
            # one point in each equal-width bin is the defining property
            which = np.floor(pts / (10.0 / n)).astype(int)
            self.assertCountEqual(which.tolist(), list(range(n)))

    def test_strata_handles_a_degenerate_axis(self):
        with _SamplingMode('lhs'):
            pts = pcalc._strata(3.0, 3.0, 8)
            np.testing.assert_allclose(pts, np.full(8, 3.0))

    def test_lhs_stratifies_both_axes(self):
        """Latin hypercube: every 1-D projection is perfectly stratified."""
        with _SamplingMode('lhs'):
            n = 32
            c1, c2 = pcalc._placements((0.0, 4.0), (0.0, 1.0), n)
            for pts, hi in ((c1, 4.0), (c2, 1.0)):
                which = np.floor(pts / (hi / n)).astype(int)
                self.assertCountEqual(which.tolist(), list(range(n)))

    def test_random_mode_ignores_the_collapse(self):
        """'random' stays a faithful baseline, so it samples even a dead axis."""
        with _SamplingMode('random'):
            c1, c2 = pcalc._placements((0.0, 4.0), (0.0, 1.0), 64,
                                       informative=(False, False))
            self.assertGreater(len(set(c1.tolist())), 1)
            self.assertGreater(len(set(c2.tolist())), 1)

    def test_collapsed_axis_sits_at_the_midpoint(self):
        with _SamplingMode('lhs'):
            c1, c2 = pcalc._placements((0.0, 4.0), (0.0, 1.0), 16,
                                       informative=(True, False))
            np.testing.assert_allclose(c2, np.full(16, 0.5))


class TestExactP10(unittest.TestCase):

    def test_matches_hand_computed_values(self):
        fzn, _ = _fixture_zone()
        with _SamplingMode('exact'):
            self.assertAlmostEqual(fzn.P10('x').P10, 0.15, places=9)
            self.assertAlmostEqual(fzn.P10('z').P10, 0.50, places=9)

    def test_is_deterministic(self):
        fzn, _ = _fixture_zone()
        with _SamplingMode('exact'):
            vals = {fzn.P10('x').P10 for _ in range(5)}
        self.assertEqual(len(vals), 1)

    def test_ignores_n(self):
        """The closed form is not a sampled mean, so -n cannot change it."""
        fzn, _ = _fixture_zone()
        with _SamplingMode('exact'):
            self.assertAlmostEqual(fzn.P10('x', 1).P10, fzn.P10('x', 999).P10)

    def test_equals_p32_of_the_perpendicular_set(self):
        """Stereological identity: P10 along d is P32 of the fractures normal to d.

        Here every fracture normal to x has its whole area inside the zone, so
        P10-x must equal (summed area of A and B) / zone volume.
        """
        fzn, _ = _fixture_zone()
        area = (1.0 * 2.0) + (1.0 * 1.0)      # A + B
        with _SamplingMode('exact'):
            self.assertAlmostEqual(fzn.P10('x').P10, area / (10.0 * 1.0 * 2.0),
                                   places=9)

    def test_empty_orientation_scores_zero(self):
        """No fractures normal to y, so a y scan line crosses nothing."""
        fzn, _ = _fixture_zone()
        with _SamplingMode('exact'):
            self.assertAlmostEqual(fzn.P10('y').P10, 0.0, places=12)


class TestSampledAgreesWithExact(unittest.TestCase):

    def _exact(self, d):
        fzn, _ = _fixture_zone()
        with _SamplingMode('exact'):
            return fzn.P10(d).P10

    def test_lhs_converges_to_the_closed_form(self):
        for d, tol in (('x', 0.02), ('z', 1e-9)):
            want = self._exact(d)
            fzn, _ = _fixture_zone()
            with _SamplingMode('lhs', seed=7):
                got = fzn.P10(d, 4096).P10
            self.assertAlmostEqual(got, want, delta=max(tol, tol * want),
                                   msg=f'P10-{d}: lhs {got} vs exact {want}')

    def test_random_converges_to_the_closed_form(self):
        want = self._exact('x')
        fzn, _ = _fixture_zone()
        with _SamplingMode('random', seed=7):
            got = fzn.P10('x', 20000).P10
        self.assertAlmostEqual(got, want, delta=0.01)

    def test_lhs_is_less_noisy_than_random(self):
        """The point of the change: same budget, smaller spread."""
        spread = {}
        for mode in ('random', 'lhs'):
            vals = []
            for seed in range(12):
                fzn, _ = _fixture_zone()
                with _SamplingMode(mode, seed=seed):
                    vals.append(fzn.P10('x', 64).P10)
            spread[mode] = max(vals) - min(vals)
        self.assertLess(spread['lhs'], spread['random'],
                        f'lhs {spread["lhs"]:.5f} vs random {spread["random"]:.5f}')


class TestPseudo2DDetection(unittest.TestCase):

    def test_spanning_fraction(self):
        fzn, _ = _fixture_zone()
        # every fracture spans y; only A and C span x; only A spans z
        self.assertAlmostEqual(fzn.spanning_fraction(1), 1.0)
        self.assertAlmostEqual(fzn.spanning_fraction(2), 1.0 / 3.0)

    def test_an_axis_every_fracture_spans_is_not_informative(self):
        fzn, _ = _fixture_zone()
        a = fzn.d[fzn.perp == 0, 2]        # y-from of the x-normal fractures
        b = fzn.d[fzn.perp == 0, 3]        # y-to
        self.assertFalse(fzn._informative(a, b, 1))

    def test_a_partially_spanned_axis_is_informative(self):
        """A thin axis is NOT assumed dead: B does not span z, so z still matters."""
        fzn, _ = _fixture_zone()
        a = fzn.d[fzn.perp == 0, 4]        # z-from
        b = fzn.d[fzn.perp == 0, 5]        # z-to
        self.assertTrue(fzn._informative(a, b, 2))

    def test_collapse_makes_sampling_exact(self):
        """Only C is normal to z and it spans both placement axes, so every
        placement gives the same count -- the sampled answer is the exact one."""
        want = 0.5
        vals = []
        for seed in range(6):
            fzn, _ = _fixture_zone()
            with _SamplingMode('lhs', seed=seed):
                vals.append(fzn.P10('z', 8).P10)
        self.assertEqual(len(set(vals)), 1)
        self.assertAlmostEqual(vals[0], want, places=9)


class TestExactP20P22(unittest.TestCase):

    def test_p22_matches_hand_computed_value(self):
        """Scan planes normal to z, placed along z in 0..2.

        A cuts every plane (its z-extent is the full 2 m, weight 1); B cuts half
        of them (weight 0.5); C is normal to z and cannot cut.  Each contributes
        aperture x its in-plane extent (1 m in y), over a 10 x 1 m plane.
        """
        fzn, _ = _fixture_zone()
        with _SamplingMode('exact'):
            got = fzn.P22('z').P22
        self.assertAlmostEqual(got, (1.0 * AP + 0.5 * AP) / 10.0, places=12)

    def test_p20_p22_are_deterministic(self):
        fzn, _ = _fixture_zone()
        with _SamplingMode('exact'):
            self.assertEqual(len({fzn.P22('z').P22 for _ in range(4)}), 1)
            self.assertEqual(len({fzn.P20('z').P20 for _ in range(4)}), 1)

    def test_sampled_p22_converges_to_the_closed_form(self):
        fzn, _ = _fixture_zone()
        with _SamplingMode('exact'):
            want = fzn.P22('z').P22
        fzn, _ = _fixture_zone()
        with _SamplingMode('lhs', seed=3):
            got = fzn.P22('z', 2048).P22
        self.assertAlmostEqual(got, want, delta=0.02 * want)


class TestWorkerPropagation(unittest.TestCase):
    """Pool workers re-import the module, so settings must be pushed across."""

    def test_init_worker_sets_the_mode(self):
        prev = (pcalc.__SAMPLING__, pcalc.__RNG__)
        try:
            pcalc._init_worker('exact')
            self.assertEqual(pcalc.__SAMPLING__, 'exact')
        finally:
            pcalc.__SAMPLING__, pcalc.__RNG__ = prev


class TestJobSeeding(unittest.TestCase):
    """--seed must pin a measure's placements, not a worker's stream.

    Seeding per process made the answer depend on which worker picked up which
    job, so a seeded run was reproducible only at --max-cpus 1.
    """

    def test_unseeded_runs_get_no_generator(self):
        self.assertIsNone(pcalc._job_rng(None, 0, 0))

    def test_same_job_and_seed_give_the_same_stream(self):
        a = pcalc._job_rng(11, 2, 5).random(8)
        b = pcalc._job_rng(11, 2, 5).random(8)
        np.testing.assert_array_equal(a, b)

    def test_different_jobs_get_independent_streams(self):
        base = pcalc._job_rng(11, 0, 0).random(8)
        for izn, ijob in ((0, 1), (1, 0), (1, 1)):
            other = pcalc._job_rng(11, izn, ijob).random(8)
            self.assertFalse(np.array_equal(base, other),
                             f'zone {izn} job {ijob} repeats zone 0 job 0')

    def test_a_seeded_measure_is_reproducible(self):
        """What the seed is for: same seed, same P10, however it was scheduled."""
        vals = []
        for _ in range(3):
            fzn, _ = _fixture_zone()
            with _SamplingMode('lhs', seed=None):
                vals.append(fzn.P10('x', 32, rng=pcalc._job_rng(4, 0, 0)).P10)
        self.assertEqual(len(set(vals)), 1)

    def test_an_explicit_generator_beats_the_module_one(self):
        """The job's generator must win, or the module global leaks back in."""
        fzn, _ = _fixture_zone()
        with _SamplingMode('lhs', seed=1):
            a = fzn.P10('x', 32, rng=pcalc._job_rng(4, 0, 0)).P10
        with _SamplingMode('lhs', seed=999):
            b = fzn.P10('x', 32, rng=pcalc._job_rng(4, 0, 0)).P10
        self.assertEqual(a, b)


if __name__ == '__main__':
    unittest.main()
