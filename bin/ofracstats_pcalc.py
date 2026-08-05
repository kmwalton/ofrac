#!/usr/bin/env python
"""Calculates "P-- system" values for orthogonal fracture networks.

DETAILED DESCRIPTION

Reads >=1 file in the format produced by RFGen or Fractran, given on command
line.  Computes select fracture abundance measures of sub-regions of the
fracture network.

Fracture abundance measures (density, intensity, porosity) are defined using
the "P-system" as defined by W. Dershowitz. e.g.:
[1] http://www.fracman.com/wp-content/uploads/Fracture-Intensiity-Measures-P32-with-Wang-Conversions.pdf

CURRENTLY IMPLEMENTED MEASURES:
- P10 : linear fracture density, counts per metre along scanline(s)
- P20 : areal fracture density, unbiased counts per square metre
- P30 : volumetric fracture density, counts per cubic metre
- P22 : porosity, area of fractures (aperture * length) per unit area sampled
- P32 : fracture area per unit sampled volume. [sum(length_1*length_2)]/volume_total
- P33 : porosity, volume of fractures (aperture * length * length) per unit volume sampled
 
Note: Some sampling bias may be present when using subzones. Subzones contain
and consider all fractures that intersect the subzone volume. Thus, one fracture
may be counted in two subzones. e.g. see slide 17 in [1].

P20 has been corrected to be an unbiased estimator. Fracture trace ends within
the sample zone are counted, then that count is divided by two.

INPUT FILE FORMAT
"RFGEN"-style

Orthogonal fracture input file format includes a header line and lines
describing fractures. All file lines up to the header line are ignored. Comment
lines beginning with // are ignored. And lines after the e.g.

```
// The following is the header line, where square brackets indicate optional
// columns:
//   [id] xfrom    xto    yfrom    yto    zfrom    zto aperture [orientation]
//
// <add metadata, recommended>
id  xfrom    xto    yfrom    yto    zfrom   zto  aperture  orientation
 1    0.0     1.0     0.5     0.5     0.0    1.0  0.000100   2
```

Placement of scan lines and scan planes
=======================================

The P-system measures of Dershowitz & Herda (1992) are *stereological*
estimators: P10 (fractures per metre of scan line) and P20/P22 (per unit area of
scan plane) are defined as expectations over a randomly placed probe. What the
code reports is therefore an estimate of a mean, and how the probes are placed
governs how noisy that estimate is. Selected with ``--sampling``.

``random`` -- independent uniform placements
--------------------------------------------
The historical rule: each scan line's position is drawn independently and
uniformly. This is plain Monte Carlo integration, so the standard error decays
as N^(-1/2) and, more importantly, independent points in one dimension have
exponentially distributed gaps -- the placements *clump*, leaving parts of the
domain unprobed. Retained for reproducing earlier results and for measuring how
much the alternatives buy.

``lhs`` -- Latin hypercube placement (default)
----------------------------------------------
Each placement axis is divided into N equal strata and one jittered point is
taken per stratum; the coordinate lists for the two axes are then paired at
random. This is the Latin hypercube design of McKay, Beckman & Conover (1979).
Its defining property is that *every one-dimensional projection* is perfectly
stratified, which is what is wanted here because the two placement axes seldom
matter equally -- a cross-section domain may be 50 m in x and 1 m in y.

This is stratified sampling in the sense of Cochran (1977); for the
piecewise-constant counting functions integrated here it removes the clumping
term that dominates the plain Monte Carlo error. Related, and worth reading if
this is pushed further: quasi-Monte Carlo with low-discrepancy (Sobol' or
Halton) sequences -- see Niederreiter (1992) -- and the Koksma-Hlawka
inequality, which bounds integration error by the product of the integrand's
variation and the point set's discrepancy.

An axis is skipped entirely when it provably carries no information: if every
candidate fracture spans the zone's full extent along that axis, the
intersection test cannot depend on that coordinate. All N strata are then spent
on the axis that does vary. Note this is an exact test on fracture extents, not
an assumption drawn from a dimension being thin -- a 1 m "unit" dimension whose
fractures do *not* all span it is still sampled, which is what separates a
genuinely thin 3D domain from a pseudo-2D one. Each zone reports
``spanning_frac`` and ``axis_extent`` per axis so the distinction is visible.

``exact`` -- closed-form expectation, no sampling
--------------------------------------------------
For axis-aligned (orthogonal) fracture sets these expectations can be evaluated
analytically, giving the exact answer at zero variance and less cost than N
probes.

*P10.* A scan line along axis d0 is positioned at a point (c1, c2) in the plane
of the other two axes. A fracture perpendicular to d0 occupies a rectangle
[a1,b1) x [a2,b2) in that plane, and the line crosses it exactly when (c1, c2)
falls inside the rectangle. The number of crossings is therefore the number of
rectangles covering the point, and averaging over a uniformly placed line is
integrating that coverage function over the placement plane:

    E[count] = ( 1 / A ) * SUM_i  |[a1,b1) ^ zone|_d1 * |[a2,b2) ^ zone|_d2

    E[P10]   = E[count] / m

with A the zone's area in the placement plane, m the scan-line length, and
``^`` an interval intersection. The numerator is just the total fracture area
of the set perpendicular to d0, so this reproduces the standard stereological
identity that P10 measured along d equals P32 of the fractures normal to d --
i.e. the general P32 = (2/pi) * P10 relation for isotropic fracturing collapses
to a factor of one when every fracture is normal to the scan line.

*P20 / P22.* A scan plane normal to axis d is positioned at a coordinate c along
d, so placement is one-dimensional and ``lhs`` degenerates to ordinary
stratification. A fracture cuts the plane exactly when c lies in its extent
[lo,hi) along d, so each fracture is weighted by that extent over the zone's
length on d:

    w_i      = |[lo,hi) ^ zone|_d / L_d
    E[P20]   = ( SUM_i w_i * ends_i / 2 ) / A_plane
    E[P22]   = ( SUM_i w_i * area_i )     / A_plane

This is the Cauchy/Buffon-style argument behind classical stereology: the
probability that a probe meets a body is proportional to the body's projected
extent along the probe's placement direction. See Underwood (1970) or Baddeley
& Jensen (2005) for the general treatment, and Mauldon (1994) / Zhang & Einstein
(1998) for the fracture-network case, including the sampling-bias corrections
that matter when traces are censored at the domain boundary.

**Caveat.** The closed forms above assume the fractures are a stationary set
within the zone, and they say nothing about the *variance* between individual
probes -- which is a real property of a network (fracture spacing statistics),
not only an artefact. Use ``--sampling lhs`` when that distribution is the
object of interest, and ``exact`` when the mean is.

References
----------
* Dershowitz, W.S. & Herda, H.H. (1992) Interpretation of fracture spacing and
  intensity. *Proc. 33rd U.S. Symp. Rock Mechanics*, 757-766.
* McKay, M.D., Beckman, R.J. & Conover, W.J. (1979) A comparison of three
  methods for selecting values of input variables. *Technometrics* 21(2),
  239-245.
* Cochran, W.G. (1977) *Sampling Techniques*, 3rd ed. Wiley.
* Niederreiter, H. (1992) *Random Number Generation and Quasi-Monte Carlo
  Methods*. SIAM.
* Underwood, E.E. (1970) *Quantitative Stereology*. Addison-Wesley.
* Baddeley, A. & Jensen, E.B.V. (2005) *Stereology for Statisticians*. Chapman
  & Hall/CRC.
* Mauldon, M. (1994) Intersection probabilities of impersistent joints.
  *Int. J. Rock Mech. Min. Sci.* 31(2), 107-115.
* Zhang, L. & Einstein, H.H. (1998) Estimating the mean trace length of rock
  discontinuities. *Rock Mech. Rock Engng.* 31(4), 217-235.

For documenttion of JSON output, see `ofrac.p_system`, or run

    $ pdoc ofrac.p_system

AUTHOR: Ken Walton, kmwalton@uoguelph.ca

LICENCE: GNU GPLv3

Documentation intended to work with pdoc3.

TODO:
- reimplement --batch-dir and table output

"""

import argparse
import sys
import re
import copy
import traceback
import glob
import os
import datetime
import contextlib
import multiprocessing
from random import uniform
from math import log10, floor
from itertools import chain,product
from collections import deque, defaultdict
from typing import NamedTuple

import json

import numpy as np

try:
    from ofrac.ofracs import parse as parse_dfn
    from ofrac.p_system import *
    from ofrac.p_system.constants import *
    from ofrac.spatialzone import SpatialZone
except ModuleNotFoundError:
    # accommodate "old style" PYTHONPATHing to within this module
    from ofracs import parse as parse_dfn
    from p_system import *
    from p_system.constants import *
    from spatialzone import SpatialZone

__VERBOSITY__ = 0
"""Module level verbosity"""

__SAMPLING__ = 'lhs'
"""How scan lines and scan planes are positioned.  One of:

   'random'  independent uniform draws -- simple Monte Carlo, the historical
             behaviour.  Error falls as N^-1/2 and the placements clump.
   'lhs'     Latin hypercube: one placement per equal-width stratum on every
             placement axis, paired at random.  Default.
   'exact'   no sampling at all; the expectation over all placements is
             evaluated in closed form.  Zero variance.

See "Placement of scan lines and scan planes" in the module docstring.
"""

__RNG__ = None
"""numpy Generator used when no per-job generator was supplied.  Created on
first use and seeded from the OS -- i.e. irreproducible, which is what an
unseeded run should be.  A seeded run does not go through here at all; see
`_job_rng`."""


def _rng(rng=None):
    """The generator to draw from: `rng` if given, else the module-level one."""
    global __RNG__
    if rng is not None:
        return rng
    if __RNG__ is None:
        __RNG__ = np.random.default_rng()
    return __RNG__


def _job_rng(seed, izn, ijob):
    """The generator for one measure of one zone, or None when unseeded.

    Seeding is per *job* rather than per process on purpose.  The placements a
    measure draws then depend only on ``(seed, zone, measure)`` and not on where
    the job ran, so ``--seed`` gives the same answer at any ``--max-cpus`` --
    including bit-for-bit agreement between the serial and pooled paths.

    Seeding per worker instead cannot do this: which job a worker picks up is a
    race on the pool's task queue, so the stream a given measure draws from is
    not pinned even when each worker's own stream is.

    The jobs still get mutually independent streams -- distinct spawn keys off
    one root -- so nothing is correlated across zones or measures.
    """
    if seed is None:
        return None
    return np.random.default_rng([seed, izn, ijob])


def _init_worker(sampling):
    """Carry the placement settings into a pool worker.

    Workers re-import the module, so anything set from the command line at
    module scope is back at its default in the child.  Called via
    ``Pool(initializer=...)``.

    The seed is not carried across here -- it travels with each task instead,
    see `_job_rng`.
    """
    global __SAMPLING__
    __SAMPLING__ = sampling


def _overlap(a, b, lo, hi):
    """Length of the intersection of each interval [a,b) with [lo,hi]."""
    return np.clip(np.minimum(b, hi) - np.maximum(a, lo), 0.0, None)


def _strata(lo, hi, n, rng=None):
    """`n` jittered positions, one per equal-width stratum of [lo,hi].

    One-dimensional stratified (jittered) sampling.  Unbiased, and for the
    piecewise-constant counting functions used here it removes the clumping
    that independent uniform draws produce.
    """
    if hi <= lo:
        return np.full(n, float(lo))
    return np.linspace(lo, hi, n, endpoint=False) \
        + _rng(rng).random(n) * ((hi - lo) / n)


def _uniform(lo, hi, n, rng=None):
    """`n` independent uniform positions -- the historical placement rule."""
    if hi <= lo:
        return np.full(n, float(lo))
    return _rng(rng).uniform(lo, hi, n)


def _mid(r):
    return 0.5 * (r[0] + r[1])


def _placements(r1, r2, n, informative=(True, True), rng=None):
    """Positions for `n` scan lines on the plane spanned by two axes.

    Returns ``(c1, c2)``, each of length `n`.

    Under ``'lhs'`` each axis is split into `n` equal strata with one jittered
    point per stratum, and the two coordinate lists are paired at random.  That
    is a Latin hypercube design (McKay, Beckman & Conover 1979): *every*
    one-dimensional projection is perfectly stratified, which matters here
    because the two placement axes rarely contribute equally.

    An axis flagged non-informative is not sampled at all -- every placement
    takes its mid-point, and all `n` strata are spent on the axis that does
    carry information.
    """
    i1, i2 = informative

    if __SAMPLING__ == 'random':
        # deliberately ignores `informative`: this mode exists to reproduce the
        # historical placement rule exactly, so it stays a clean baseline to
        # measure the alternatives against
        return _uniform(*r1, n, rng), _uniform(*r2, n, rng)

    if i1 and i2:
        c1 = _strata(*r1, n, rng)
        c2 = _strata(*r2, n, rng)
        _rng(rng).shuffle(c2)       # random pairing == Latin hypercube
        return c1, c2
    if i1:
        return _strata(*r1, n, rng), np.full(n, _mid(r2))
    if i2:
        return np.full(n, _mid(r1)), _strata(*r2, n, rng)
    return np.full(n, _mid(r1)), np.full(n, _mid(r2))

##############################################################################
#
#  helpers
#
def _organize_PXXResults(results_list):
    """
    Organizes a list of P-system Result objects into a flat dictionary
    with keys matching their formatted string prefixes 
    (e.g., 'P10-x', 'P20-yz', 'P30').
    """
    organized = {}
    
    for result in results_list:
        # 1. Extract the base metric name (e.g., 'P10Result' -> 'P10')
        metric_type = type(result).__name__.replace('Result', '')
        
        # 2. Build the specific key based on the object's dimensionality
        if hasattr(result, 'd_scan'):
            # 1D metrics
            key = f"{metric_type}-{result.d_scan}"
            
        elif hasattr(result, 'd_perp'):
            # 2D metrics (Assumes the PERP dictionary is available in this scope)
            direction_str = PERP[result.d_perp]
            key = f"{metric_type}-{direction_str}"
            
        else:
            # 3D metrics have no suffix
            key = metric_type
            
        # 3. Store the result
        organized[key] = result
        
    return organized


def _get_json_context(args_json):
    """Returns a context manager for a file, stdout, or a 'do-nothing' context."""
    if args_json is None:
        return contextlib.nullcontext(None)

    if args_json == '-' or args_json is sys.stdout:
        return contextlib.nullcontext(sys.stdout)

    # Standard file path
    return open(args_json, 'w')


class NotValidInputFile(Exception):
    """Custom exception for no valid parser found"""
    def __init__(self,msg):
        self.message = msg


##############################################################################
#
#  A class for fractures and fracture stats for a certain spatial regtion
#

class FractureZone:                                         #{{{
   """The fractures of a network that intersect a zone, and its P-measures.

   The fractures are kept as the `numpy` arrays that `ofrac.ofracs.OFracGrid`
   exposes --- coordinates, apertures, perpendicular axes, per-axis extents,
   areas and volumes --- rather than one object per fracture, so that every
   measure below is a whole-array operation over the zone.
   """

   def __init__(self, zn, fxNet, nScan=10):
      """Make a zone's fracture set from a network

      Arguments:
        zn(SpatialZone): the region to sample
        fxNet(ofrac.ofracs.OFracGrid): the network to take fractures from
        nScan(int): the number of scan lines/planes per measure
      """
      self.zn = zn
      self.nScan = nScan
      self.zn_vol = zn.vol()

      (start, end) = (zn.start(), zn.end())

      # the fractures of this zone: those intersecting it, in their own network
      znet = fxNet.subsetFx( fxNet.getFxMaskIn(start, end) )

      self.d = znet.getFxCoordinates()
      """(M,6) coordinates of this zone's fractures: xfrom, xto, yfrom ... zto"""

      self.ap = znet.getFxApertures()
      """(M,) apertures"""

      self.perp = znet.getFxPerpAxes()
      """(M,) index of the axis each fracture is perpendicular to"""

      self.ext = znet.getFxLengths()
      """(M,3) extent of each fracture along each axis; zero on its perp axis"""

      self.area = znet.getFxAreas()
      """(M,) area of each fracture"""

      self.vol = znet.getFxVolumes()
      """(M,) void volume (area times aperture) of each fracture"""

      self.in_plane = znet.getFxPlaneAxes()
      """(M,3) mask of the two axes lying in each fracture's plane"""

      # lengths are measured within the zone only when the user asks for that
      self.ext_zone = self.ext
      if getattr(zn, 'truncateToZone', False):
         self.ext_zone = znet.getFxLengths(clip_to=(start, end))

   def __len__(self):
      return len(self.ap)

   def __str__(self):
       return str(self.zn)

   def fracStr(self, i):
      """Return the printable form of this zone's fracture `i`"""
      d = self.d[i]
      return '({:8.3f}->{:8.3f}, {:8.3f}->{:8.3f}, {:8.3f}->{:8.3f}), ap={:.6f}'\
          .format(*d, self.ap[i])

   def setNScan( self, n ):
      """Set the number of scan lines/planes/whatever in the next PNN
      calculation(s).

      Args:
        n(int): the new number
      """
      self.nScan = n

   def _informative(self, a, b, d):
      """Does moving the placement along axis `d` change what is intersected?

      False only when *every* candidate fracture spans the zone's full extent
      on `d`; then the intersection test cannot depend on that coordinate and
      sampling it is wasted effort.

      This is an exact test on the fracture extents, **not** an inference from
      the domain being thin.  A 1 m "unit" dimension whose fractures do not all
      span it is still informative and is still sampled -- which is the case
      that distinguishes a genuinely thin 3D domain from a pseudo-2D one.
      """
      if a.size == 0:
         return False
      return not bool(np.all((a <= self.zn.st(d)) & (b >= self.zn.en(d))))

   def spanning_fraction(self, d):
      """Fraction of this zone's fractures spanning its full extent on axis `d`.

      1.0 together with a small extent is the signature of a pseudo-2D domain;
      anything less means the dimension carries real structure.  Reported per
      zone so the distinction is visible in the output rather than assumed.
      """
      if len(self.d) == 0:
         return float('nan')
      a, b = self.d[:, 2*d], self.d[:, 2*d+1]
      return float(np.count_nonzero((a <= self.zn.st(d)) & (b >= self.zn.en(d)))) / len(self.d)

   def P10( self, dScanLine, nScanLine=None, rng=None ):

      if not nScanLine:
         nScanLine = self.nScan

      # d0, the direction of the scan line; d1 and d2, the axes it is placed on
      d0 = DIR[dScanLine]
      d1 = DIR[PERP[dScanLine][0]]
      d2 = DIR[PERP[dScanLine][1]]

      # only fractures perpendicular to the scan line can be crossed by it;
      # take their in-plane extents once, then test whole scan lines at a time
      onLine = self.perp == d0
      (a1, b1) = ( self.d[onLine, 2*d1], self.d[onLine, 2*d1+1] )
      (a2, b2) = ( self.d[onLine, 2*d2], self.d[onLine, 2*d2+1] )
      iOnLine = np.flatnonzero(onLine)

      # m, the length of one scan line
      m = self.zn.size(d0)
      (cc,cm) = (0,0.0)

      if __VERBOSITY__ > 1:
         print('\nP10-{} for scanline at:'.format(dScanLine))

      if __SAMPLING__ == 'exact':
         # Closed form.  count(c1,c2) is the number of axis-aligned rectangles
         # [a1,b1)x[a2,b2) containing the placement point, so its mean over a
         # uniformly placed scan line is the summed rectangle area (clipped to
         # the zone) divided by the placement area.  See the module docstring.
         A = self.zn.size(d1) * self.zn.size(d2)
         if A <= 0.0 or m <= 0.0:
            return P10Result(dScanLine, m, 0, 0.0, float('inf'))
         area = (_overlap(a1, b1, *self.zn.r(d1))
                 * _overlap(a2, b2, *self.zn.r(d2))).sum()
         cbar = float(area) / A            # expected crossings per scan line
         return P10Result(dScanLine, m, int(round(cbar)), m, cbar / m)

      # placement axes carrying no information are collapsed to their mid-point
      inf1 = self._informative(a1, b1, d1)
      inf2 = self._informative(a2, b2, d2)
      cs1, cs2 = _placements(self.zn.r(d1), self.zn.r(d2), nScanLine,
                             (inf1, inf2), rng)

      for ci in range(nScanLine):
         (c1, c2) = (cs1[ci], cs2[ci])

         hits = (a1 <= c1) & (c1 < b1) & (a2 <= c2) & (c2 < b2)
         count = int(np.count_nonzero(hits))

         if __VERBOSITY__ > 1:

            s = 'P10-{} for scanline at ({},{})=({:.3f},{:.3f})'.format(
                     dScanLine,
                     PERP[dScanLine][0], PERP[dScanLine][1],
                     c1, c2,
                 )

            dens = float(count)/m if m > 0.0 else float('inf')
            spac = 1.0/dens if count > 0 else float('inf')

            print( "{}: {:6.3g}/m {:6.3g}m (count={})".format(
                      s, dens, spac, count ) )

            if __VERBOSITY__ > 2 and count > 0:
                 w = 1+int(log10(count))
                 s = 'Fractures found:\n'
                 for iff,ifx in enumerate(iOnLine[hits], start=1):
                        s += f'{iff:{w}}: {self.fracStr(ifx)}\n'
                 print(s)

         cc += count
         cm += m

      size_1 = self.zn.size(DIR[dScanLine])
      if cm == 0.0:
          return P10Result(dScanLine, size_1, cc, cm, float('inf'))
      else:
          return P10Result(dScanLine, size_1, cc, cm, float(cc) / cm)


   def lengths(self):
      """Return per-axis fracture length statistics for this zone

      Lengths are of the part of each fracture inside the zone when the zone
      was made with `truncateToZone`; a fracture that leaves nothing inside the
      zone is left out.
      """

      ext = self.ext_zone

      # a fracture clipped out of existence in the plane it lies in is not one
      # of this zone's fractures any more
      alive = ~np.any( self.in_plane & (ext <= 0.0), axis=1 )

      stats = {}
      for a,ax in enumerate('xyz'):
         # the extent along an axis is a length only for the fractures lying in
         # a plane containing that axis
         l = ext[alive & self.in_plane[:, a], a]

         stats[ax] = {
            'MIN': float(l.min()) if l.size else 1e100,
            'MAX': float(l.max()) if l.size else 0.0,
            'SUM': float(l.sum()),
            'COUNT': int(l.size),
            }

      return stats

   def P20_P22(self, dperpScanPlane, nScanPlane, rng=None):

      fracEndCount = 0
      fracCount = 0
      fracArea = 0.0
      scanPlaneTotalArea = 0.0

      d = DIR[dperpScanPlane]
      d1 = DIR[PERP[dperpScanPlane][0]]
      d2 = DIR[PERP[dperpScanPlane][1]]

      # only fractures not perpendicular to the scan plane's normal can cut it
      cuts = self.perp != d
      (lo, hi) = ( self.d[cuts, 2*d], self.d[cuts, 2*d+1] )
      cExt = self.ext[cuts]

      # the trace of such a fracture in the scan plane is a line of its in-plane
      # extent, so this is the trace length times the aperture
      cArea = self.ap[cuts] * np.maximum(cExt[:, d1], cExt[:, d2])

      # count each fracture's trace ends that lie within the zone, once
      cEnds = np.zeros(len(cArea), dtype=int)
      for a in (d1, d2):
         (aFrom, aTo) = ( self.d[cuts, 2*a], self.d[cuts, 2*a+1] )
         isEnd = cExt[:, a] > 0.0
         cEnds += isEnd & (aFrom >= self.zn.st(a))
         cEnds += isEnd & (aTo <= self.zn.en(a))

      planeArea = self.zn.size(d1) * self.zn.size(d2)

      if __SAMPLING__ == 'exact':
         # Closed form.  A fracture cuts the plane iff the plane's position
         # falls in its extent [lo,hi) along d, so the mean over a uniformly
         # placed plane weights each fracture by that extent (clipped to the
         # zone) over the zone's length on d.  One 1-D integral per quantity.
         L = self.zn.size(d)
         if L <= 0.0:
            w = np.ones(len(cArea))            # degenerate zone: the plane is the zone
         else:
            w = _overlap(lo, hi, *self.zn.r(d)) / L
         return (float((cEnds * w).sum()) / 2.0,
                 float((cArea * w).sum()),
                 planeArea)

      # scan-plane placement is one-dimensional, so 'lhs' reduces to stratified
      positions, _ = _placements(self.zn.r(d), (0.0, 0.0), nScanPlane,
                                 (self._informative(lo, hi, d), False), rng)

      for plane in range(nScanPlane):
         positionOfPlane = positions[plane]

         # prune to the fractures that intersect this plane
         onPlane = (lo <= positionOfPlane) & (positionOfPlane < hi)

         fracCount += int(np.count_nonzero(onPlane))
         fracEndCount += int(cEnds[onPlane].sum())
         fracArea += float(cArea[onPlane].sum())

         scanPlaneTotalArea += planeArea

      return ( int(fracEndCount/2), fracArea, scanPlaneTotalArea )

   def P20(self, dperpScanPlane, nScanPlane=None, rng=None):
        if not nScanPlane:
            nScanPlane = self.nScan

        if self.zn.size(DIR[dperpScanPlane]) == 0.0:
            nScanPlane = 1

        fCount, fArea, spArea = self.P20_P22(dperpScanPlane, nScanPlane, rng)

        if spArea == 0.0:
            return P20Result(dperpScanPlane, int(round(fCount)), spArea, float('inf'))
        # f_count is a reporting field and is formatted as an integer; the P20
        # value itself keeps the (fractional) expectation under 'exact'
        return P20Result(dperpScanPlane, int(round(fCount)), spArea,
                         float(fCount) / spArea)

   def P22(self, dperpScanPlane, nScanPlane=None, rng=None):
        if not nScanPlane:
            nScanPlane = self.nScan

        if self.zn.size(DIR[dperpScanPlane]) == 0.0:
            nScanPlane = 1

        fCount, fArea, spArea = self.P20_P22(dperpScanPlane, nScanPlane, rng)
        size_1, size_2 = [self.zn.size(_d) for _d in PERP[dperpScanPlane]]

        if spArea == 0.0:
            return P22Result(dperpScanPlane, size_1, size_2,
                             int(round(fCount)), spArea, float('inf'))
        return P22Result(dperpScanPlane, size_1, size_2,
                         int(round(fCount)), spArea, float(fArea) / spArea)

   def P30(self):
        f_count = len(self)
        if self.zn_vol == 0:
            return P30Result(f_count, self.zn_vol, float('inf'))
        return P30Result(f_count, self.zn_vol, f_count / self.zn_vol)

   def P32(self):
        if not hasattr(self, '_fxA'):
            self._fxA = float(self.area.sum())

        if self.zn_vol < 1e-6:
            return P32Result(self.zn_vol, self._fxA, float('inf'))
        return P32Result(self.zn_vol, self._fxA, self._fxA / self.zn_vol)

   def P33(self):
        fx_vol = float(self.vol.sum())
        if self.zn_vol < 1e-6:
            return P33Result(self.zn_vol, fx_vol, float('inf'))
        return P33Result(self.zn_vol, fx_vol, fx_vol / self.zn_vol)

#}}}


##############################################################################
#
#  Read command line, start doing stuff
#


# for multiprocessing
def _run_calc_job(task_args):
    """Unpacks and runs a FractureZone method with its arguments.

    The generator arrives with the task rather than being read off a module
    global, so a job's placements do not depend on which process ran it.
    """
    fzn, (method, arg), rng = task_args

    # If the method doesn't take an extra direction argument (like P30, P32,
    # P33).  Those are exact counts and sums -- no placements, so no generator.
    if arg is None:
        return method(fzn)

    # If it does take an argument (like P10, P20, P22)
    return method(fzn, arg, rng=rng)


def doEverything(args, batchDir=''):

    fxNets = []
    fracFileSubZones = []

    # collect data for JSON as its generated
    dict4json = {}

    # iterate through all files (or problem prefixes) found on command line
    for fnin in args.FILES:
       if __VERBOSITY__:
          print( "========= %s ========="%(fnin))

       fxNet = parse_dfn(fnin)

       # populate
       nfile= fxNet.getFxCounts()
       mima = list( map(float, chain( *fxNet.getBounds() ) ) )

       subZone = SpatialZone(start=( mima[0], mima[2], mima[4] ) ,
                        end=( mima[1], mima[3], mima[5] ) )

       fxNets.append( fxNet )
       fracFileSubZones.append( subZone )

       if __VERBOSITY__:
          print( "Boundaries of fractures: {}".format(subZone) )
          print( "Number of fractures counted: %d" % ( sum(nfile) ) )
          # getFxCounts is indexed by the axis a fracture is perpendicular to,
          # so index 0 counts the fractures in the yz-plane
          for i,ax in enumerate('xyz'):
             print( "Number in %s-plane: %d" % ( PERP[ax], nfile[i] ) )
       del nfile

    # superimpose the networks of all files
    fxNet = fxNets[0] if len(fxNets) == 1 else fxNets[0].merge( *fxNets[1:] )


    #  Determine the size of the domain
    dom = None
    if args.domain:
       dom = SpatialZone(asString=args.domain)
    else:
       dom = SpatialZone()
       for zn in fracFileSubZones:
          dom.expandBoundingBox( zn )


    if not dom:
       print("Could not determine domain size. Specify it with --domain",
             file=sys.stderr)
       sys.exit(1)

    if __VERBOSITY__:
        print( "========= Domain =========" )
        print( "Domain: {}; size: {} x {} x {}".format(str(dom), dom.xSz(),dom.ySz(),dom.zSz()) )

    dict4json['Domain'] = str(dom)

    # determine sample zones
    sampleZn = []
    if args.sample_zones:
       sampleZn =list( map(lambda s:
             SpatialZone(asString=s, truncateToZone=args.truncate_to_sample_zones),
          args.sample_zones.split(';') ) )
    else:
       sampleZn = [ dom ]


    # collect each zone's fractures once; the calculations below re-use them
    fzns = [ FractureZone(zn, fxNet, args.n) for zn in sampleZn ]

    results = []

    _jobs = list(chain(
        [(FractureZone.P10, d,) for d in sorted(DIR)],
        [(FractureZone.P20, d,) for d in sorted(DIR)],
        [(FractureZone.P22, d,) for d in sorted(DIR)],
        [(FractureZone.P30, None,),
         (FractureZone.P32, None,),
         (FractureZone.P33, None,),]
    ))

    if args.max_cpus == 1:
        for (izn, (zn, fzn)) in enumerate(zip(sampleZn, fzns)):

            _d = { 'SubDomain':str(zn), 'nscan':args.n, 'nfracs':len(fzn),
                   'sampling':__SAMPLING__,
                   'spanning_frac':{ d:fzn.spanning_fraction(DIR[d]) for d in sorted(DIR) },
                   'axis_extent':{ d:zn.size(DIR[d]) for d in sorted(DIR) }, }

            _r = list(map(_run_calc_job,
                          ((fzn, j, _job_rng(args.seed, izn, ij))
                           for ij, j in enumerate(_jobs))))

            results.append(_r)
            _d.update(_organize_PXXResults(_r))
            dict4json[f'Zone{izn}'] = _d

    else:

        # Worker processes re-import this module, so module-level settings
        # revert to their defaults there.  Push --sampling across explicitly:
        # without this it is silently ignored whenever --max-cpus > 1, and every
        # zone is computed with the default.  The seed travels per task instead,
        # so that the answer does not depend on which worker took which job.
        with multiprocessing.Pool(args.max_cpus,
                                  initializer=_init_worker,
                                  initargs=(__SAMPLING__,)) as pool:
            for (izn, (zn, fzn)) in enumerate(zip(sampleZn, fzns)):

                _d = { 'SubDomain':str(zn), 'nscan':args.n, 'nfracs':len(fzn),
                       'sampling':__SAMPLING__,
                       'spanning_frac':{ d:fzn.spanning_fraction(DIR[d]) for d in sorted(DIR) },
                       'axis_extent':{ d:zn.size(DIR[d]) for d in sorted(DIR) }, }

                _r = pool.map(_run_calc_job,
                              [(fzn, j, _job_rng(args.seed, izn, ij))
                               for ij, j in enumerate(_jobs)])
                results.append(_r)

                _d.update(_organize_PXXResults(_r))
                dict4json[f'Zone{izn}'] = _d


    # get ready for batch printing
    header="""
Columns:
Directory - the directory
# - the subzone number, as in list below
P10_[xyz] is in [counts/metre]

Sample Zones:
{}
------------------------------------------
""".format(
        '\n'.join(map(lambda z: f"{z[0]}: {z[1]}", enumerate(sampleZn)))
    )
    (FW, FPREC, BDW, ZNW) = (
              12,
              3,
              # '9' is for 'Directory' heading
              max(map(len,args.batch_dir+['Directory',])),
              int(log10(len(sampleZn)))+1,
              )
    rowFmt = f'{{:{BDW}s}} {{:{ZNW}d}}' \
     + 3*f' {{:{FW}.{FPREC}f}}'
    hdrFmt = f'{{:{BDW}s}} {{:{ZNW}s}}' \
     + 3*f' {{:{FW}s}}'
    header += hdrFmt.format('Directory', '#', 'P10_x','P10_y','P10_z',)


    # calc/print stats for sub zones
    if __VERBOSITY__:
       print( "========= stats for fracture network sub-zones =========" )


    # get ready for tecplot printing
    import os
    tecout = ''
    tecout += f'# {os.path.realpath(__file__)} on {datetime.datetime.now()}\n'
    tecout += 'VARIABLES="X","Y","Z"\n'
    tecout += '\n'

    for (izn, (zn, fzn)) in enumerate(zip(sampleZn, fzns)):

        # r is a list of 2-tuples of the data and a formatter
        r = results[izn]

        if args.batch_dir:
            if batchDir == args.batch_dir[0]:
                print(header)
            print(rowFmt.format(batchDir, izn,
                # pick out just the P10s
                r[0][0][1],
                r[1][0][1],
                r[2][0][1],) )

        else:
            print( "--- {} ---".format(str(zn) ) )

        # print results
        print('\n'.join(map(str, r)))


        # zone header
        tecout += f'''ZONE T="{zn!s}" ZONETYPE=ORDERED I=2 J=2 K=2 DATAPACKING=BLOCK\n'''
        # Auxvar
        for i,d in enumerate(sorted(DIR)):
            try :
                mag = int(floor(log10(r[i][-1])))
                _v = f'{round(r[i][1], -mag+1):.{-mag+2}f}'
                _1dv = f'{round(1./r[i][1], mag+3):.{max(0,mag+2)}f}'
            except ValueError:
                _v = '-'
                _1dv = '-'
            tecout += f'''AUXDATA P10{d}="{_v}"\n'''
            tecout += f'''AUXDATA Spacing{d}="{_1dv}"\n'''
        tecout += f'''AUXDATA P32="{r[10].P32:.3g}"\n'''
        tecout += f'''AUXDATA P33="{r[11].P33:.3g}"\n'''

        # length stats ... P21???
        lengths = fzn.lengths()
        for os in sorted(DIR):
            if lengths[os]['COUNT']>0:
                print( "%s-length: min=%7.3f max=%7.3fm avg=%7.3fm (count=%4d)" %
                     (os, lengths[os]['MIN'], lengths[os]['MAX'],
                      lengths[os]['SUM'] / lengths[os]['COUNT'], lengths[os]['COUNT'] ) )
            else:
                print( "%s-length:         (count=%4d)" % (os, 0) )

        #ZONE data
        #import pdb ; pdb.set_trace()
        coordBlks = [ '', '', '', ]
        for z,y,x in product(*reversed(zn.c)):
            coordBlks[0] += f' {x:11.3f}'
            coordBlks[1] += f' {y:11.3f}'
            coordBlks[2] += f' {z:11.3f}'
        tecout += ''.join(f'# {d}\n{v[1:]}\n' for d,v in zip('xyz',coordBlks))

    if 'tp_out' in args and args.tp_out: # not None or ''
        if __VERBOSITY__:
            print(f'==== Writing tecplot file {args.tp_out} ====')
        with open(args.tp_out,'w') as fout:
            fout.write(tecout)


    # do JSON output
    with _get_json_context(args.json_out) as fout:
        if fout:
            save_json(dict4json, fout, indent=2)



def main(argv=None):
    # Both of these are read all over the module -- __SAMPLING__ is even what
    # gets handed to the worker pool as initargs -- so they must be set on the
    # module, not shadowed by locals, exactly as they were at module scope.
    global __VERBOSITY__, __SAMPLING__

    # command line options setup
    parser = argparse.ArgumentParser(
          formatter_class=argparse.RawDescriptionHelpFormatter,
          description= 'Calculates "P-- system" values for orthogonal fracture networks.',
          epilog=__doc__
    )

    parser.add_argument( "-v", "--verbose", action='count', default=0,
           help="Print more detail with each -v on the command line")

    parser.add_argument( "--sampling", choices=('random','lhs','exact'),
            default='lhs',
            help="how scan lines/planes are positioned: 'random' = independent "
                 "uniform draws (the historical rule, error ~N^-1/2, placements "
                 "clump); 'lhs' = Latin hypercube, one jittered placement per "
                 "stratum on every placement axis (default); 'exact' = no "
                 "sampling, the expectation over all placements evaluated in "
                 "closed form (zero variance, and -n is ignored)")

    parser.add_argument( "--use-estimator", dest='sampling',
            action='store_const', const='exact',
            help="alias for --sampling exact")

    parser.add_argument( "--seed", type=int, default=None,
            help="seed the placement RNG, making a sampled run reproducible -- "
                 "and reproducible at any --max-cpus, since each measure's "
                 "placements are derived from the seed rather than from the "
                 "process that happened to run it. Unseeded by default, so "
                 "repeat runs reveal the sampling spread. Irrelevant to "
                 "--sampling exact, which is deterministic")

    parser.add_argument( "-n", type=int, default=10,
           help="Number of scan lines or planes to use")

    # sampling zones
    parser.add_argument( "-s","--sample-zones", default=None,
          help="""Regions for sub-sampling (must be rectanle- or box-shaped).
          Separate subzones with ';'.
          Input format is somewhat flexible, e.g.:
          "(5.0,5.0,5.0)" implies one sub zone from (0,0,0) to (5,5,5);
          One subzone from (0,0,3) to (5,5,4) may be specified as
          "start(0,0,3) end(5,5,4)",
          "st(0,0,3)e(5,5,4)",
          "(0,0,3)(5,5,4)",
          "start(0,0,3) size(5,5,1)", or
          "(0,0,3)si(5,5,1)";
          Three subzones may be specified as
          "(5,5,5);(0,0,2.5)si(5,5,2.5);(0,0,2.5)(5,5,5)".
          If this option is omitted, then one subzone that captures all fractures is assumed.
          """)

    parser.add_argument( "--truncate-to-sample-zones", default=False,
          action='store_true',
          help='Truncate fractures to the bounday of the zone when calculating lengths')

    parser.add_argument( "-d", "--domain", default=None,
           help="""The whole domain. Specify in the same way as the subzones
           above. If this is omitted, then a box that bounds all fractures will be
           used.""")

    parser.add_argument( "FILES", nargs='+',
          help="List of RFGen-style input files, or Fractran problem prefix.")

    parser.add_argument( '--max-cpus', type=int, default=4,
          help='The number of processors to apply to these calculations' )

    parser.add_argument( '-b', '--batch-dir',
          action='append',
          help="""Set to batch mode for FILES in each of the given batch
          directories. Multiple directories may be listed, or unix "glob-style"
          wildcards (*, ", [character range]). e.g. "-b runDirA -b runDirB', or
              '-b runDir[AB]', or '-b runDir*'
          Assumes that network domains are the same size in each directory in
          the batch, and that the same sample-zones can be applied.

          With this mode, 'verbosity' is ignored and results are printed in
          table format.
          IN DEVELOPMENT: Only the P10 values are printed
          """)

    parser.add_argument( '--tp-out', type=str,
          help='Name of the tecplot file to write to.' )

    parser.add_argument( '--json-out', metavar='JSON_FILE',
        type=str,
        nargs='?',
        default=None, const='-',
        help='''The name of a JSON-format output file. If this argument is
        provided with '-' or without a filename, JSON data will be printed to
        stdout''',
    )

    # command line args
    args = parser.parse_args(argv)

    __VERBOSITY__ = args.verbose

    __SAMPLING__ = args.sampling
    # args.seed is not turned into a generator here: each job derives its own
    # from it, so that the result does not depend on --max-cpus.  See _job_rng.

    if args.max_cpus > 1 and args.verbose > 0:
        print(f'Verbosity level {args.verbose} selected. Resetting --max-cpus from {args.max_cpus} to 1.')
        args.max_cpus = 1

    if args.batch_dir:
        __VERBOSITY__ = 0

        # expand any glob-style entries
        allDirs = []
        for d in args.batch_dir:
            if re.search("[*?[]", d):
                allDirs.extend(glob.glob(d))
            else:
                allDirs.append(d)
        args.batch_dir = allDirs

        scriptCallDir = os.getcwd()
        for d in args.batch_dir:
            if not os.path.isdir(d):
                 print(f'Skipping: not a directory {d}', file=sys.stderr)
            try:
                os.chdir(d)
                doEverything(args, batchDir=d)
            except NotValidInputFile:
                print(f'Skipping: no valid inputs {d}', file=sys.stderr)
            finally:
                os.chdir(scriptCallDir)

    else:
        args.batch_dir = [] # "fix" the default 'None'
        try:
            doEverything(args)
        except NotValidInputFile as e:
            print(e, file=sys.stderr)
            return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
