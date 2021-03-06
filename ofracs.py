#!/usr/bin/env python3
"""Orthogonal discrete FRACture network container and manipulator

A container and grid generator for 3D, axis-aligned, orthogonal fracture
networks. Typically, text data files generated externally (e.g., by proprietary
codes RFGen, or Fractran) use the data types and container herin to
convert and manipulate networks and grids.

External generator and parsers are usually required. `ofracs.OFrac` provides
basic construction from 7 floating point values, like the following:

 xfrom    xto    yfrom    yto    zfrom   zto  aperture

   0.0     1.0     0.5     0.5     0.0    1.0  0.000100

AUTHOR: Ken Walton, kmwalton@g360group.org.
Licenced under GNU GPLv3
Documentation intended to work with pdoc3.
"""

import sys,warnings,copy,re,os,pickle
from decimal import *
from bisect import bisect_left,bisect_right
from math import log10,floor
from itertools import chain

__DEBUG__ = False
__VERBOSITY__ = 0

# other valid options might be "ignore" "fail"
__BAD_FRACTURE_POLICY__ = "warn"

# set precision of decimal numbers
getcontext().prec = 12
N_COORD_DIG = Decimal('0.001')
N_APERT_DIG = Decimal('0.000001')

def D_CO(v):
    """Return a decimal with the required number of digits for coordinates"""
    return Decimal(v).quantize(N_COORD_DIG)
def D_AP(v):
    """Return a decimal with the required number of digits for apertures"""
    return Decimal(v).quantize(N_APERT_DIG)
DINF = Decimal('infinity')

def toDTuple(someString):
    """return a tuple of Decimals from some coordinate-like or list-like string"""
    return tuple( D_CO(v) for v in re.sub("[(),]",' ',someString).split() )

def numTuple2str( somedt, sep=',', start='(', end=')' ):
    return start+\
        "{}".format(sep.join( str(v) for v in somedt ) ) +\
        end


# handy
def iterpairs( flatList ):
    return zip( flatList[0::2], flatList[1::2] )

class FractureDefinitionError(Exception):
    """Improper OFrac definition input values"""
    def __init__(self, message):
        self.message = message

class FractureCollapseError(Exception):
    """Operation caused a fracture's size to degenerate"""
    def __init__(self, message):
        self.message = message

class FractureCollapseWarning(UserWarning):
    """The informational warning message generated when a fracture collapses"""
    def __init__(self, message):
        self.message = message

class GridlineChangeWarning(UserWarning):
    """The informational warning message generated when something happens with gridlines"""
    def __init__(self, message):
        self.message = message

class NotValidOFracGridError(Exception):
    """If un-pickling fails..."""
    def __init__(self, message):
        self.message = message


__FX_COLLAPSE_POLICIES__ = ['fail', 'warn-omit', 'omit', 'ignore',]
"""Policies useful when nudging fractures.
These determine whether a failure, warning, or nothing will happen when a
collapse occurs.
"""
__FX_COLLAPSE_POLICY__ = __FX_COLLAPSE_POLICIES__[0]
"""The Policy in use. Default 'fail'"""



# class for OFrac objects
class OFrac():
    """An orthogonal fracture object"""

    def __init__(self, *vals, **kwargs):
        if vals:
            # assume initializing from xfrom, xto, yfrom, yto, zfrom, zto, ap
            self.__init_from_vals__( *vals )
        else:
            # assume initializing from another OFrac object
            other = kwargs['fromOFrac']
            self.d = other.d
            self.ap = other.ap
            self.myNet = other.myNet

        if 'myNet' in kwargs:
            self.myNet = kwargs['myNet']

    def __init_from_vals__(self, xfrom, xto, yfrom, yto, zfrom, zto, ap):
        """Initialize axis-aligned fracture object.

        set coordinates
        set aperture
        optionally set the
        """
        self.d = tuple(D_CO(v) for v in (xfrom, xto, yfrom, yto, zfrom, zto,) )
        self.ap = D_AP(ap)

        # calculate the size of the fracture in each dimension
        # make sure there are are two good-length sides
        countSameOrBad = ( int(Decimal(xto-xfrom)<N_COORD_DIG)
            + int(Decimal(yto-yfrom)<N_COORD_DIG)
            + int(Decimal(zto-zfrom)<N_COORD_DIG) )

        countBad = int(xto < xfrom) + int(yto < yfrom) + int(zto < zfrom)

        if countSameOrBad > 1 or countBad > 0:
            raise FractureDefinitionError( 'not a plane: '+str(self) )


    def _checkCollapse(self, operation, newd):
        # check for fracture "collapse" due to rounding
        for a in range(3):
            if self.d[2*a] != self.d[2*a+1] and newd[2*a] >= newd[2*a+1] :
                errmsg = 'Fracture collapsed during {}!:\n'.format(operation)
                errmsg +='before - {}\n'.format(self)

                # change coordinates of this object, rather than create a new
                # temporary one (that will fail because coords bad)
                dsave = self.d
                self.d = newd
                errmsg +='after  - {}'.format(self)
                # return self to previous state
                self.d = dsave
                raise FractureCollapseWarning( errmsg )

    @staticmethod
    def determineFracOrientation(f):
        """return 0 for yz-plane, 1 for xz, 2 for xy
            
        Why this arbitrary assignment? The return value is the index into the
        string 'xyz' that you can remove to get a string describing the parallel
        plane.

        'xyz'
         012
        """
        o=-1
        if   f.d[0]==f.d[1]:    o = 0
        elif f.d[2]==f.d[3]:    o = 1
        elif f.d[4]==f.d[5]:    o = 2
        else:
            raise RuntimeError(
                'Could not determine orientation of {}'.format(f))
        return o

    def determinePerpAxisVal(self):
        """Return the axis perpendicular to this fracture's plane, and its value"""
        o = OFrac.determineFracOrientation(self)
        return ( o, self.d[(o)*2], )

    def __str__(self,wid=8):
        return '({:{w}}->{:{w}}, {:{w}}->{:{w}}, {:{w}}->{:{w}}), ap='.format(
                *self.d, w=wid)+str(self.ap)

    def __repr__(self):
        # TODO output more grungy details here
        return self.__str__()

    def nudge(self, nudgeIncrement):
        """Modify a fracture to new "nudged" coordinates."""

        nudgeIncrement = D_CO(nudgeIncrement)

        def n(v):
            return ((v/nudgeIncrement).quantize(0) * nudgeIncrement).quantize(N_COORD_DIG)
        
        newd = tuple( map( n, self.d ) )

        try:
        self._checkCollapse("nudging", newd)

        except FractureCollapseWarning as e:
            if __FX_COLLAPSE_POLICY__ == 'fail':
                raise
            elif __FX_COLLAPSE_POLICY__ == 'warn-omit':
                print(e, file=sys.stderr)
                returnstatus = False
            elif __FX_COLLAPSE_POLICY__ == 'omit':
                returnstatus = False

        else:
        self.d = newd

        # invalidate the gridlines in the containing network
        if self.myNet is not None:
            self.myNet.invalidateGrid()

        return returnstatus

    def truncate(self, s, e):
        """Modify a fracture's size to fit within a given bounding box
            
        s - the minimum coordinate of the bounding box (numeric-type triple)
        e - the maximum coordinate of the bounding box (numeric-type triple)

        Use "big values" in the s and e coordinates if you do not want to
        truncate in a particular direction.

        No error checking on valid inputs s and e!
        """

        domTruncStr = "truncating to ({})->({})".format(
           ','.join(str(s) for v in s),
           ','.join(str(s) for v in e) )
        
        newd = []
        for a,((v1,v2),mi,ma) in enumerate(zip(iterpairs(self.d), s, e)) :
            if v1 == v2 and ( v1 < mi or v1 > ma ):
                # fracture's plane falls outside domain!
                raise FractureCollapseWarning(
                    "Fracture fell outside domain when {}\n{}={} - {}".format(
                        domTruncStr,
                        'xyz'[a],v1, str(self) ) )

            # fit fracture length to domain
            # or, copy-in v1==v2 plane coordinates
            newd.append( max(v1,mi) )
            newd.append( min(v2,ma) )

        self._checkCollapse( domTruncStr, newd )

        self.d = tuple( newd )
        return self

    def calcElems(self):
        """Calculate the number of grid cells occupied by this fracture
        using the grid in self.myNet"""

        ngl = [1,1,1]

        for a in range(3):
            ngl[a] = max(1, (bisect_left(self.myNet._gl[a], self.d[2*a+1])
                    - bisect_left(self.myNet._gl[a], self.d[2*a])))

        return ngl[0] * ngl[1] * ngl[2]

    def getCentroid(self):
        """Get the centroid"""
        d = self.d
        two = D_CO(2.0)
        return list(map(lambda v:(v[0]+v[1])/two, zip(d[0::2],d[1::2])))

    def iterPoints(self):
        """iterate over the four corner points"""
        (n,v) = self.determinePerpAxisVal()

        d = self.d
        if n == 0:
            yield (d[0], d[2], d[4])
            yield (d[0], d[2], d[5])
            yield (d[0], d[3], d[5])
            yield (d[0], d[3], d[4])
        elif n == 1:
            yield (d[0], d[2], d[4])
            yield (d[0], d[2], d[5])
            yield (d[1], d[2], d[5])
            yield (d[1], d[2], d[4])
        elif n == 2:
            yield (d[0], d[2], d[4])
            yield (d[0], d[3], d[4])
            yield (d[1], d[3], d[4])
            yield (d[1], d[2], d[4])
        else:
            raise RuntimeError('Unexpected (wrong?) value for perpenicular direction')

class OFracGrid():
    """Container/Utility class for an orthogonal fracture network."""


    def __init__(self,
            domainOrigin=None,
            domainSize=None,
            fx=[],
            nudgeTo=0.0,
            gl=[],
            fixedgl=[]
        ):
        """Initialize a network
        
        domainOrigin and domainSize, if specified, will override and filter-out
        any fractures or gridlines passed in the other parameter lists.
            
        """

        self._fx = []
        self._fixedgl = [ set(), set(), set() ]

        self.metadata = {}

        # set origin
        # set size
        self._setDomain(domainOrigin, domainSize)
        
        self._ocounts=[0,0,0]
        self._resetMinMax()

        # apply nudging
        if nudgeTo > 0.0:
            raise NotImplementedError()

        # Set bounding box to truncate fractures and grid lines.
        # The parameters for domainOrigin and domainSize override any fractures
        # and gridlines specified.
        s = 3*[-0.5 * sys.float_info.max,] # start
        e = 3*[ 0.5 * sys.float_info.max,] # end
        for i,(do,ds) in enumerate(zip(self.domainOrigin, self.domainSize)):
            if do.is_finite():
                s[i] = do
                if ds.is_finite():
                    e[i] = do+ds

        messages = []

        # add fractures
        for i,f in enumerate(fx):
            try:
                cf = OFrac( *f, myNet=self )
                if cf.truncate(s,e) is not None:
                    self._fx.append(cf)
                else:
                    messages.append(f'Fracture {i} is out of the domain.')
            except FractureDefinitionError as err:
                if __BAD_FRACTURE_POLICY__ == 'warn':
                    messages.append(f'Fracture {i} in inputs is bad: {err!s}')
                elif __BAD_FRACTURE_POLICY__ == 'ignore':
                    pass
                else:
                    raise RuntimeError( f'ABORT: Fracture {i} bad: {err!s}' )


        if messages:
            nm = len(messages)
            messages.append(f'Found {nm} warnings among {len(fx)} input fractures.')
            print('\n'.join( m for m in messages), file=sys.stderr)

        # store fixed gridlines
        if fixedgl:
            for a in range(3):
                # add gridlines
                self._fixedgl[a] = set(D_CO(d) for d in
                    filter( lambda candidate: s[a] <= candidate <= e[a], fixedgl[a]) )


        # make gridlines, if not given
        self._gl = [ set(), set(), set() ]
        if gl:
            for i,g in enumerate(gl):
                self._gl[i].update( d
                    for d in filter(lambda v: s[i]<=v<=e[i],
                        map(D_CO, g)
                    )
                )

            self._gl = list( sorted(x) for x in self._gl )
            self._gridValid = True
            self._reCountFractures()
            self._remakeMinMax(useFixedGrid=True, useGrid=True, useFx=True)

        else:
            self._remakeMinMax(useFixedGrid=True, useGrid=True, useFx=True)
            self._remakeGridLineLists()

        #import pdb ; pdb.set_trace()

        # re-make the domain origin again (in the case that it wasn't provided
        # initially), presuming that the fracture network and fixed grid lines
        # provide what the users' choices here give the proper size 
        if domainOrigin == None and ( fixedgl or fx ):
            self.domainOrigin=tuple(D_CO(a[0]) for a in self._mima)
        if domainSize == None and ( fixedgl or fx ):
            self.domainSize=tuple(D_CO(a[1]) for a in self._mima)


    def _setDomain(self,domainOrigin=None, domainSize=None):
        """Take a string/tuple/other numeric and convert/store as the size

        If arguments are none, the domain reverts to...
            self.domainOrigin=3*(-infinity,)
            self.domainSize=3*(infinity,)

        Add these domain boundaries to the list of fixed grid lines
        """

        def conv2tup(whatever):
            #import pdb ; pdb.set_trace()
            if type(whatever) is tuple:
                # assume this is already in the format we need.
                return whatever
            elif type(whatever) is str:
                return toDTuple( whatever )
            else:
                # assume whatever is list-like
                return tuple( D_CO(v) for v in whatever )

        if domainOrigin is None or domainOrigin == 3*(-DINF,):
            self.domainOrigin = 3*(-DINF,)
        else:
            self.domainOrigin = conv2tup( domainOrigin )

        if domainSize is None or domainSize == 3*(DINF,):
            self.domainSize = 3*(DINF,)
        else:
            self.domainSize = conv2tup( domainSize )
            if domainOrigin is None:
                # special: if domain Size is specified, but not the origin, assume
                # that the origin is (0,0,0)
                self.domainOrigin = conv2tup([0.,0.,0.])

        for v in self.domainSize:
            if v <= Decimal('0.0'):
                raise ValueError('Must have positive size values for domain')

        for a in range(3):
            s = self.domainOrigin[a]
            ds = self.domainSize[a]

            if s.is_finite():
                self._fixedgl[a].add(s)

                if ds.is_finite():
                    self._fixedgl[a].add(s+ds)

    def _resetMinMax(self):
        """set _mima to invalid range"""
        # reset fx min and max coordinate lengths
        self._mima = [ [DINF,-DINF],[DINF,-DINF],[DINF,-DINF], ]

    def _remakeMinMax_includeFx(self, fx):
        # determine fx net min/max coordinate
        for i in range(3):
            self._mima[i][0] = min(self._mima[i][0], fx.d[2*i  ])
            self._mima[i][1] = max(self._mima[i][1], fx.d[2*i+1])

    def _remakeMinMax(self, **kwargs) :
        """Use the given data source(s) to reset _mima values


            Keyword arguments:
                useFixedGrid : bool
                useGrid : bool
                useFx : bool
        """

        if 'useFixedGrid' in kwargs and kwargs['useFixedGrid']:
            for a,gla in enumerate(self._fixedgl):
                if len(gla) == 0: continue
                self._mima[a][0] = min(self._mima[a][0], min(gla))
                self._mima[a][1] = max(self._mima[a][1], max(gla))

        if 'useGrid' in kwargs and kwargs['useGrid']:
            for a,gla in enumerate(self._gl):
                if len(gla) == 0: continue
                self._mima[a][0] = min(self._mima[a][0], gla[ 0])
                self._mima[a][1] = max(self._mima[a][1], gla[-1])

        if 'useFx' in kwargs and kwargs['useFx']:
            for fx in self._fx:
                self._remakeMinMax_includeFx(fx)


    def _remakeGridLineLists(self, keep_glAsSets=False):
        """Scan fractures and make re-make gridlines & counts"""

        beforeCounts = self.getGridLineCounts()

        # reset grid, copy in set objects
        self._gl = [ s.copy() for s in self._fixedgl ]
        self._resetMinMax()
        self._remakeMinMax( useFixedGrid=True )

        for fx in self._fx:
            # add gridlines
            self._gl[0].update( fx.d[0:2] )
            self._gl[1].update( fx.d[2:4] )
            self._gl[2].update( fx.d[4:6] )

            # determine fx net min/max coordinate
            # (instead of doing it below)
            for i in range(3):
                self._mima[i][0] = min(self._mima[i][0], fx.d[2*i  ])
                self._mima[i][1] = max(self._mima[i][1], fx.d[2*i+1])


        if not keep_glAsSets:
            self._gl = list( sorted(x) for x in self._gl )
            self._gridValid = True

        self._reCountFractures()

        afterCounts = self.getGridLineCounts()
        if __VERBOSITY__>2:
            wid = 1+int(log10(1.0+max( v for v in beforeCounts + afterCounts ) ))
            print( 'Remade grid lines. Grid line counts', file=sys.stderr )
            for k,v in { "Before":beforeCounts, "After":afterCounts }.items():
                print( "   {:8} nx={:{w}}, ny={:{w}}, nz={:{w}}".format(
                    k,*v, w=wid), file=sys.stderr )


    def _reCountFractures(self):
        # reset orientation counts
        self._ocounts = 3 * [ 0, ]
        # rescan fractures
        for fx in self._fx:
            o = OFrac.determineFracOrientation(fx)
            self._ocounts[o] += 1

# domain information
    def getBounds(self):
        if not self._gridValid: self._remakeGridLineLists()
        return copy.deepcopy(self._mima)

    def invalidateGrid(self):
        """Set the gridlines, boundaries, and fracture counts to be invalid."""
        self._gridValid = False

    def setDomainSize(self, domainOr, domSz):
        """Set the domain size, possibly exclude gridlines and fractures"""


        self._setDomain(domainOr, domSz)

        # remove fixed gridlines outside of domain
        s = self.domainOrigin
        e = self.getDomainEnd()

        for a in range(3):
            glsToRemove = set()
            for gl in self._fixedgl[a]:
                if gl < s[a] or e[a] < gl:
                    glsToRemove.add(gl)
                    message = "User-specified gridline at {}={} is being removed!".format('xyz'[a],gl)
                    #warnings.warn(message,UserWarning)
                    if __VERBOSITY__ > 1:
                        print(message, file=sys.stderr)
            if __VERBOSITY__ and len(glsToRemove)>0:
                print("Removed {} user-specifed gridlines in {}".format(len(glsToRemove),'xyz'[a]), file=sys.stderr)
                
            self._fixedgl[a] -= glsToRemove

        # add gridlines representing the size
        for a in range(3):
            self._fixedgl[a].add( s[a] )
            self._fixedgl[a].add( e[a] )

        # cull gridlines
        # assume these are already sorted
        for a,gla in enumerate(self._gl):
            f = bisect_left( gla, s[a] )
            l = bisect_right( gla, e[a] )
            gla = gla[f:l]

            if gla[0] != s[a]:
                gla.insert(0,s[a])

            if gla[-1] != e[a]:
                gla.insert(len(gla),e[a])

            # store-back to self
            self._gl[a] = gla
                
        self._resetMinMax()
        self._remakeMinMax(useFixedGrid=True, useGrid=True)

        # truncate/readd fractures
        nNewFx = 0
        nTruncMsgs = 0
        truncMsgs = ''

        for i,f in enumerate(self._fx):
            try:
                f.truncate(s, e)
            except (FractureCollapseError,FractureCollapseWarning) as ce:
                # capture a string listing which fractures were truncated
                if nTruncMsgs == 0:
                    truncMsgs = '\n'.join( ce.message.split('\n')[:2] )
                else:
                    truncMsgs += '\n'+ce.message.split('\n')[1]
                nTruncMsgs += 1
                o = OFrac.determineFracOrientation(f)
                self._ocounts[o] -= 1
            else:
                #re-store successfully truncated fracture
                self._fx[nNewFx] = f
                nNewFx += 1
                # re-check min-max
                self._remakeMinMax_includeFx(f)

        # clear storage freed-up by out-of-domain fractures
        del self._fx[nNewFx:]

        # TODO: use warnings.warn
        if __VERBOSITY__:
            print( "{} fractures removed when domain size changed.".format(
                    nTruncMsgs) )
            if __VERBOSITY__ > 1:
                print( truncMsgs )


    def getDomainEnd(self):
        def sumInfGuarded(aList):
            if   DINF in map(abs, aList): return DINF
            else:                         return sum(aList)

        return tuple( map( sumInfGuarded, zip(self.domainOrigin,self.domainSize) ) )

# methods for fractures        
    def addFracture( self, candidateOFrac, index=-1 ):
        """Add a given OFrac fracture object"""
        self._cbValid = False

        s = self.domainOrigin
        e = self.getDomainEnd()

        try:
            # make a copy of the candidate
            cf = OFrac( fromOFrac=candidateOFrac, myNet=self )

            if cf.truncate(s,e) is not None:
                self._fx.append(cf)
            else:
                messages.append(f'Fracture {index} is out of the domain.')
        except FractureDefinitionError as err:
            if __BAD_FRACTURE_POLICY__ == 'warn':
                messages.append(f'Fracture {index} in inputs is bad: {err!s}')
            elif __BAD_FRACTURE_POLICY__ == 'ignore':
                pass
            else:
                raise RuntimeError( f'ABORT: Fracture {index} bad: {err!s}' )

    def iterFracs(self):
        """iterate over fractures"""
        for f in self._fx:
            yield f

    def nudgeAll( self, nudgeTo ):
        """Removes fractures or fails depending on __FX_COLLAPSE_POLICY__"""
        failedNudges = []
        for i,of in enumerate(self._fx):
            if not self._fx[i].nudge( nudgeTo ): failedNudges.append(i)

        for i in reversed(failedNudges):
            del self._fx[i]

    def getFxCount(self):
        return len(self._fx)

    def getFxCounts(self):
        return tuple(self._ocounts)

    def getHeader(self):
        """Return the header string for printing lists of fractures"""
        return 'xfrom xto yfrom yto zfrom zto aperture type'

    def calcFxElementCount(self, fx=None):
        """Return an the number of fracture elements for single fracture 'fx' or
            all fractures in this grid (with fx=None)"""

        #import pdb; pdb.set_trace()
        if fx:
            return 0
        else:
            return sum(map(lambda f:f.calcElems(), self._fx))

# methods for grid lines
    def addGridline( self, axis, glvalue ):
        """Add a gridline

        If it is outside the domain, the domain becomes bigger."""
        self._fixedgl[axis].add(D_CO(glvalue))
        self._gridValid = False

    def getGridLineCounts(self):
        """Return a 3-tuple of counts grid lines"""
        return tuple( len(l) for l in self._gl )

    def getGridLineFirstInterval(self, axis):
        """Return the interval between first two gridlines of a given axis"""
        if not self._gridValid: self._remakeGridLineLists()
        return self._gl[axis][1] - self._gl[axis][0]

    def iterGridLines(self, axis):
        """Iterate through grid lines of a given axis"""

        # convert to integer axis
        if type(axis) == str:
            axis = 'xyz'.find(axis.lower())
        if axis < 0 or axis > 2: raise ValueError('bad axis')

        if not self._gridValid: self._remakeGridLineLists()
        for v in self._gl[axis]:
            yield v

    def isUniformGridSpacing(self, axis):
        """Scan grid lines to determine if spacing is uniform"""
        if not self._gridValid:
            self._remakeGridLineLists()

        # convert to integer axis
        if type(axis) == str:
            axis = 'xyz'.find(axis.lower())
        if axis < 0 or axis > 2: raise ValueError('bad axis')

        # trivial
        if len(self._gl[axis]) <= 2: return True

        # check spacing between all pairs
        diff = self._gl[axis][1] - self._gl[axis][0]

        for i in range(2,len(self._gl[axis])-2):
            tdiff = self._gl[axis][i] - self._gl[axis][i-1]
            # fail early
            if abs(tdiff - diff) > 1e-6:
                return False
        return True

    def setMaxGlSpacing( self, maxGlSpacing ):
        """Add new gridlines so that the maximum space between is respected
        
        maxGlSpacing = [ maxX, maxY, maxZ ]
        """

        for a,gla in enumerate(self._gl):

            if not maxGlSpacing[a]:
                continue

            maxS = D_CO(maxGlSpacing[a])
            newGl = []
            eps = D_CO('0.001')

            for i in range(len(gla)-1):
                l1 = gla[i]
                l2 = gla[i+1]

                if l2-l1 > maxS:
                    spac = (l2-l1)/maxS
                    while l1 < l2-eps:
                        newGl.append( l1 )
                        l1 += spac
                else:
                    newGl.append( l1 )

            newGl.append(gla[-1])
            self._gl[a] = newGl

            

    def refineNearFx(self, refList):
        """Add gridlines at specified distance(s) away from fracture planes
            
        Deletes all grid lines.
        Remakes them at fracture planes, plus specified refinements.
        """

        # error check inputs
        errmsg = 'Cannot have negatively sized refinement intervals.'
        for v in refList:
            if v <= 0.0: raise ValueError(errmsg)

        # map inputs to Decimal type
        refList = list( D_CO(v) for v in refList )
        for i in range(1, len(refList)):
            refList[i] += refList[i-1]

        # forget all current grid lines
        # remake grid to fit fractures
        self._remakeGridLineLists( keep_glAsSets=True )
        glSets = self._gl

        beforeCounts = self.getGridLineCounts()

        # add in refinements
        for fx in self._fx:

            (perpAxis, paVal) = fx.determinePerpAxisVal()
            setToAddTo = glSets[perpAxis]
            mima = self._mima[perpAxis]

            for r in refList:
                if paVal - r > mima[0]:
                    setToAddTo.add( paVal-r )
                if paVal + r < mima[1]:
                    setToAddTo.add( paVal+r )

        self._gl = list( sorted(x) for x in glSets )

        self._gridValid = True

        afterCounts = self.getGridLineCounts()
        if __VERBOSITY__:
            wid = 1+int(log10(max( v for v in beforeCounts + afterCounts ) ))
            print( '\nRefined grid lines near Fx planes. Grid line counts', file=sys.stderr )
            for k,v in { "Before":beforeCounts, "After":afterCounts }.items():
                print( "   {:8} nx={:{w}}, ny={:{w}}, nz={:{w}}".format(
                    k,*v, w=wid), file=sys.stderr )


    def __str__(self):
        s = "Orthogonal Fracture Grid with:\n"

        def prod( i ):
            p = 1
            for v in i:
                p*=v
            return p
        
        stuff = {
            "Size":"{}->{}".format(
                numTuple2str(self.domainOrigin),
                numTuple2str(self.getDomainEnd() )),

            "Mins & Maxes":"{}".format(
                    ",".join( numTuple2str(t,sep='->') for t in self._mima)),

            "Grid line counts":"nx={}, ny={}, nz={}".format(
                    *map(len, self._gl) ),

            "PM elements":"{:,}".format(
                    prod(len(a)-1 for a in self._gl) ),

            "Fx Counts":"(yz,xz,xy)={}; {} total".format(
                    numTuple2str( self._ocounts ),
                    len(self._fx) )
            #sizew
            # x y z gridlines
            # frac counts`"
        }

        maxCat = max( len(cat) for cat in stuff.keys() )

        return s + '\n'.join( "   {:{w}} {}".format(kv[0]+":",kv[1],w=maxCat+1) for
                kv in stuff.items() )

    def printTecplot(self, fout=sys.stdout, printFileHeader=True, zoneName='DFN'):
        """Print this network as Tecplot FE Quadrilateral data"""

        # reassign stdout for easier printing
        stdoutsave = sys.stdout
        sys.stdout = fout

        _e = self.getFxCount()
        _n = 4*_e

        if printFileHeader:
            print(f'TITLE="DFN generated by OFracGrid ({os.path.basename(__file__)})"')
            print('VARIABLES="x","y","z","aperture"')

        for k,v in self.metadata.items():
            print(f'DATASETAUXDATA {k} = "{v}"')

        # chunks of info for the header string
        znHdrString = ( f'ZONE T="{zoneName}"', 
            'ZONETYPE=FEQUADRILATERAL DATAPACKING=BLOCK',
            f'NODES={_n} ELEMENTS={_e}',
            'VARLOCATION=([4]=CELLCENTERED)', 
            f'AUXDATA numFracs = "{_e}"'
            )
        print(' '.join(znHdrString))

        if _e < 1:
            raise RuntimeError("Domain has zero fractures! Cannot output a Tecplot zone.")

        blockVals = [_n*[Decimal(0),],_n*[Decimal(0),],_n*[Decimal(0),],]

        for iel,f in enumerate(self.iterFracs()):
            for ino,(x,y,z) in enumerate(f.iterPoints()):
                blockVals[0][4*iel+ino] = x
                blockVals[1][4*iel+ino] = y
                blockVals[2][4*iel+ino] = z
                

        # print x y z-blocks
        for ia,a in enumerate('xyz'):
            print(f'# {a}')
            #import pdb ; pdb.set_trace()
            for vals in zip( blockVals[ia][0::4],
                             blockVals[ia][1::4],
                             blockVals[ia][2::4],
                             blockVals[ia][3::4] ):
                v = ' '.join(str(v) for v in vals)
                print(f'{v}')

        # print aperture block
        print('# apertures')
        for fx in self.iterFracs():
            print(f'{fx.ap}')

        # print element data
        print('# FE data')
        w = floor(log10(_n))+1
        for iel in range(_e):
            # 1-based indices
            v = ' '.join(f'{n:{w}d}' for n in range(1+4*iel,4*iel+5))
            print(f'{v}')

        sys.stdout = stdoutsave


    def merge(self,
            *others:"other OFracGrid objects"
        ):
        """Merge this with these others and return a new OFracGrid"""


        if __VERBOSITY__ > 2:
            print(f'merging {self} with {len(others)} others')

        newGrid = OFracGrid(domainOrigin=self.domainOrigin, domainSize=self.domainSize)

        #import pdb ; pdb.set_trace()

        allOthers = chain( (self,), others )

        def nonInfMin(a,b):
            """return the min, guarding against one member being -Infinity
                (invalid)"""
            # if they're both -DINF, then DINF is returned
            if a == -DINF:   return b
            elif b == -DINF: return a
            else:            return min(a,b)

        def nonInfMax(o1,s1,o2,s2):
            """return the max sum of the pair-wise entries guarding against
                any members being Infinity (invalid)"""
            e1 = DINF
            e2 = DINF

            if abs(o1) != DINF and abs(s1) != DINF:
                e1 = o1+s1
            if abs(o2) != DINF and abs(s2) != DINF:
                e2 = o2+s2

            # if they're both DINF, then DINF is returned
            if   e1 == DINF: return e2
            elif e2 == DINF: return e1
            else:            return max(e1,e2)


        #import pdb ; pdb.set_trace()

        for other in allOthers:

            # don't add anyting for a seemingly default grid
            if not other._gl or not other._fx:
               continue

            #import pdb ; pdb.set_trace()

            domO = tuple(map(lambda v:nonInfMin(v[0],v[1]),
                        zip(newGrid.domainOrigin,other.domainOrigin)))

            domE = tuple(map(lambda v: nonInfMax(v[0],v[1],v[2],v[3]),
                        zip(newGrid.domainOrigin,newGrid.domainSize,other.domainOrigin,other.domainSize)))

            domS = tuple(map(lambda v: v[1]-v[0], zip(domO,domE)))

            for a in range(3):
                newGrid._gl[a] = list(set(newGrid._gl[a])|set(other._gl[a]))
                #newGrid._gl[a].extend( other._gl[a] )
                newGrid._gl[a].sort()

            newGrid._setDomain(domO,domS)
            newGrid._remakeMinMax(useGrid=True)

            #import pdb ; pdb.set_trace()

            if __VERBOSITY__ > 3:
                print(f'merging {len(other._fx)} fractures')

            for i,f in enumerate(other.iterFracs()):
                if __VERBOSITY__ > 5:
                    print(f'adding fracture #{i}: {f}')
                elif __VERBOSITY__ > 4:
                    print('.',end='')

                newGrid._fx.append( OFrac(fromOFrac=f, myNet=newGrid) )

            if __VERBOSITY__ > 4:
                print()


            sep = ', '
            for k,v in other.metadata.items():
                if k in newGrid.metadata:
                    newGrid.metadata[k] += f'{sep}{v}'
                else:
                    newGrid.metadata[k] = f'{v}'

        return newGrid

    @staticmethod
    def pickleTo( ofracObj, filename ):
        with open(filename, 'wb') as fout:
            pickle.dump(ofracObj, fout, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def unpickleFrom( filename ):
        with open(filename, 'rb') as fin:
            return pickle.load(fin)

    class PickleParser:
        def __init__(self, fnin):
            self.fnin = fnin
            try:
                self.myGrid = OFracGrid.unpickleFrom(self.fnin)
            except:
                raise NotValidOFracGridError(f'Failed unpickling from {self.fnin}')

        def getOFracGrid(self):
            return self.myGrid

