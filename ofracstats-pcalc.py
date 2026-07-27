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
- P32 : fracture area per unit sampled volume.[Σ(length_1*length_2)]/volume_total
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

For linear and planar measures, sample scan lines and scan planes are chosen
uniformly in the sub-zone that is being sampled.

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
except ModuleNotFoundError:
    # accommodate "old style" PYTHONPATHing to within this module
    from ofracs import parse as parse_dfn
    from p_system import *
    from p_system.constants import *

__VERBOSITY__ = 0
"""Module level verbosity"""

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
#  A class for holding and operating on a 3-d spatial region
#
class SpatialZone:            # {{{

   def __init__(self,
         size=None,
         start=None,
         end=None,
         asString=None,
         truncateToZone=False):
      """Make a zone.
         Specify one or two of the { size, start, or end } parameters, or give a
         text string 'asString' that should be parsed to find the appropriate
         info.

         asString must contain one or two (x,y,z) triples. One triple implies
         the size of the zone, which is assumed to begin at (0,0,0).

         Keywords (or abbreviated keyword)
            start | st,
            end | e, or
            size | si
         may preceed triples. In the absence of keywords, the first triple is
         assumed to be the 'start' and the second is assumed to be 'size'.


         truncateToZone : bool
            Causes fracture lengths to be calculated/reported for this spatial
            zone, if it is smaller than the whole fracture domain.

      """

      if not size and not start and not end and not asString:
         mx = sys.float_info.max
         mn = sys.float_info.min
         self.c = ( (mx,mn),(mx,mn),(mx,mn) )
         return

      if asString:
         strSave = asString

         e = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
         triple=r"\( *({}), *({}), *({}) *\)".format(e,e,e)
         word="((?:st(?:art)?)|(?:e(?:nd)?)|(?:si(?:ze)?))"

         coords = list( re.finditer("({})".format(triple), asString) )

         if len(coords) == 2:
            m = re.search("(?:st(?:art)?.*?)?"+triple, asString)
            if m:
               start = tuple( map(float,m.groups()[0:3]) )
               asString = asString[m.end():]
            else:
               start = (0.0,0.0,0.0)
         else:
            start = (0.0,0.0,0.0)


         size, end = None, None,

         m_si = re.search("si(?:ze)?.*?"+triple, asString)
         m_e = re.search("e(?:nd)?.*?"+triple, asString)
         if m_si:
            size = tuple( map(float,m_si.groups()[0:3]) )
         elif m_e:
            end = tuple( map(float,m_e.groups()[0:3]) )

         if not size and not end:
            print( "'{}' did not contain enough start, size or end info".format(strSave), file=sys.stderr )
            sys.exit(1)


      if not size and not end:
         raise Exception("Must specify 'size' or 'end' of the zone.")
      elif start and end:
         self.c = ( (start[0], end[0]), (start[1], end[1]), (start[2], end[2]) )
      elif start and size:
         self.c = ( (start[0], start[0]+size[0]), (start[1], start[1]+size[1]), (start[2], start[2]+size[2]) )
      elif size:
         self.c = ( (0.0, size[0]), (0.0, size[1]), (0.0, size[2]) )
      else:
         self.c = ( (0.0, end[0]), (0.0, end[1]), (0.0, end[2]) )

      # normalize data types
      self.c = ( (float(self.c[0][0]), float(self.c[0][1])),
                 (float(self.c[1][0]), float(self.c[1][1])),
                 (float(self.c[2][0]), float(self.c[2][1])),
              )

      for (c1,c2) in self.c:
         if c2 < c1:
            raise Exception("Bad coordinates specified for this zone.")

      self.truncateToZone = truncateToZone

   def expandBoundingBox(self, other):
      self.c = (
         ( min(self.c[0][0], other.c[0][0]), max(self.c[0][1], other.c[0][1]) ),
         ( min(self.c[1][0], other.c[1][0]), max(self.c[1][1], other.c[1][1]) ),
         ( min(self.c[2][0], other.c[2][0]), max(self.c[2][1], other.c[2][1]) )
         )

   # SiZe in a particular direction
   def xSz(self): return self.c[0][1] - self.c[0][0]
   def ySz(self): return self.c[1][1] - self.c[1][0]
   def zSz(self): return self.c[2][1] - self.c[2][0]
   def size(self,d):
       try:
          int(d)
          return self.c[d][1] - self.c[d][0]
       except ValueError:
          return self.c[DIR[d]][1] - self.c[DIR[d]][0]

   def vol(self):
      return self.size(0) * self.size(1) * self.size(2)

   # STart coord in a particular direction
   def xSt(self): return self.c[0][0]
   def ySt(self): return self.c[1][0]
   def zSt(self): return self.c[2][0]
   def st(self,d): return self.c[d][0]

   # ENd coord in a particular direction
   def en(self,d): return self.c[d][1]

   def start(self):
      """Return the starting coordinate as a tuple"""
      return tuple(map(lambda v: v[0], self.c))

   def end(self):
      """Return the ending coordinate as a tuple"""
      return tuple(map(lambda v: v[1], self.c))

   # Range in a particular direction
   def xR(self): return self.c[0]
   def yR(self): return self.c[1]
   def zR(self): return self.c[2]
   def r(self,d): return self.c[d]

   def __str__(self):
      return "x:{} y:{} z:{}".format( self.c[0], self.c[1], self.c[2] )

# }}}

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

      self.in_plane = np.arange(3) != self.perp[:, np.newaxis]
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

   def P10( self, dScanLine, nScanLine=None ):

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

      for ci in range(nScanLine):
         (c1,c2) = ( uniform(*self.zn.r(d1)), uniform(*self.zn.r(d2)) )

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

   def P20_P22(self, dperpScanPlane, nScanPlane):

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

      for plane in range(nScanPlane):
         positionOfPlane = uniform(*self.zn.r(d))

         # prune to the fractures that intersect this plane
         onPlane = (lo <= positionOfPlane) & (positionOfPlane < hi)

         fracCount += int(np.count_nonzero(onPlane))
         fracEndCount += int(cEnds[onPlane].sum())
         fracArea += float(cArea[onPlane].sum())

         scanPlaneTotalArea += self.zn.size(d1) * self.zn.size(d2)

      return ( int(fracEndCount/2), fracArea, scanPlaneTotalArea )

   def P20(self, dperpScanPlane, nScanPlane=None):
        if not nScanPlane:
            nScanPlane = self.nScan

        if self.zn.size(DIR[dperpScanPlane]) == 0.0:
            nScanPlane = 1

        fCount, fArea, spArea = self.P20_P22(dperpScanPlane, nScanPlane)

        if spArea == 0.0:
            return P20Result(dperpScanPlane, fCount, spArea, float('inf'))
        return P20Result(dperpScanPlane, fCount, spArea, float(fCount) / spArea)

   def P22(self, dperpScanPlane, nScanPlane=None):
        if not nScanPlane:
            nScanPlane = self.nScan

        if self.zn.size(DIR[dperpScanPlane]) == 0.0:
            nScanPlane = 1

        fCount, fArea, spArea = self.P20_P22(dperpScanPlane, nScanPlane)
        size_1, size_2 = [self.zn.size(_d) for _d in PERP[dperpScanPlane]]

        if spArea == 0.0:
            return P22Result(dperpScanPlane, size_1, size_2, fCount, spArea, float('inf'))
        return P22Result(dperpScanPlane, size_1, size_2, fCount, spArea, float(fArea) / spArea)

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
    """Unpacks and runs a FractureZone method with its arguments."""
    fzn, (method, arg) = task_args

    # If the method doesn't take an extra direction argument (like P30, P32, P33)
    if arg is None:
        return method(fzn)

    # If it does take an argument (like P10, P20, P22)
    return method(fzn, arg)


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
          for i,o in enumerate(INDO.values()):
             print( "Number in %s-plane: %d" % ( o, nfile[i] ) )
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

            _d = { 'SubDomain':str(zn), 'nscan':args.n, 'nfracs':len(fzn), }

            _r = list(map(_run_calc_job, ((fzn, j) for j in _jobs)))

            results.append(_r)
            _d.update(_organize_PXXResults(_r))
            dict4json[f'Zone{izn}'] = _d

    else:

        with multiprocessing.Pool(args.max_cpus) as pool:
            for (izn, (zn, fzn)) in enumerate(zip(sampleZn, fzns)):

                _d = { 'range':str(zn), 'nscan':args.n, 'nfracs':len(fzn), }

                _r = pool.map(_run_calc_job, ((fzn, j) for j in _jobs))
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



if __name__ == '__main__':
    # command line options setup
    parser = argparse.ArgumentParser(
          formatter_class=argparse.RawDescriptionHelpFormatter,
          description= 'Calculates "P-- system" values for orthogonal fracture networks.',
          epilog=__doc__
    )

    parser.add_argument( "-v", "--verbose", action='count', default=0,
           help="Print more detail with each -v on the command line")

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
    args = parser.parse_args()

    __VERBOSITY__ = args.verbose

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
            sys.exit(1)

    sys.exit(0)
