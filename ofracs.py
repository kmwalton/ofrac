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

import sys
import warnings
import re
import os
import pickle
import decimal
import copy
from decimal import Decimal,getcontext
from bisect import bisect_left,bisect_right
from math import log10,floor,ceil,prod
import itertools
from itertools import chain,count

import numpy as np

_import_warning_strings = []
"""An array of warnining messages about parsers that have not been found."""

def populate_parsers():
    """Return a list of OFracGrid parser options.

    The list will depen on what other packages are on the system and  accessible
    in PYTHONPATH, like FRACTRAN, HGS, RFGen, or Compflow, which may have
    parsers of orthogonal fracture networks.
    """

    ret = [ OFracGrid.PickleParser, ]

    try:
        import pyhgs
        import pyhgs.parser.fractran as _hgs_parser_fractran
    except ModuleNotFoundError as e:
        if "No module named 'pyhgs'" in str(e):
            pass
        elif 'pyhgs.parser.fractran' not in str(e):
            raise e
        _import_warning_strings.append(
            "Warning: did not find 'pyhgs' or its 'parser_fractran'. "
            +"Cannot parse FRACTRAN-type orthogonal fracture networks."
        )
    else:
        ret += list(_hgs_parser_fractran.iterFractranParsers())

    try:
        import pyhgs
        import pyhgs.parser.rfgen as _hgs_parser_rfgen
    except ModuleNotFoundError as e:
        if "No module named 'pyhgs'" in str(e):
            pass
        elif 'pyhgs.parser.rfgen' not in str(e):
            raise e
        _import_warning_strings.append(
            "Warning: did not find 'pyhgs' or its 'parser_rfgen'. "
            +"Cannot parse RFGen-type orthogonal fracture networks."
            )
    else:
        ret += [
            _hgs_parser_rfgen.RFGenOutFileParser,
            _hgs_parser_rfgen.RFGenFracListParser,
        ]

    try:
        import pyhgs
        import pyhgs.parser.hgseco
    except ModuleNotFoundError as e:
        if "No module named 'pyhgs'" in str(e):
            pass
        elif 'pyhgs.parser.hgseco' not in str(e):
            raise e
        _import_warning_strings.append(
            "Warning: did not find 'pyhgs' or 'pyhgs.parser.hgseco'. "
            +"Cannot parse HGS+RFGen-style orthogonal fracture networks."
            )
    else:
        ret += [pyhgs.parser.hgseco.HGSEcoFileParser,]

    try:
        import parser_rfgen as _lp
    except ModuleNotFoundError as e:
        if 'parser_rfgen' not in str(e):
            raise e
        _import_warning_strings.append(
            "Warning: did not find loose module 'parser_rfgen'. "
            +"Cannot parse RFGen-type orthogonal fracture networks."
            )
    else:
        if hasattr(_lp, 'RFGenOutFileParser'):
            ret += [_lp.RFGenOutFileParser,]
        if hasattr(_lp, 'RFGenFracListParser'):
            ret += [_lp.RFGenFracListParser,]

    return ret

def parse(file_name):
    """Return an OFracGrid using any available parser"""

    errmsg = ''
    fxNet = None
    for ParserClass in populate_parsers():
        try:
            parser = ParserClass(file_name)
            fxNet = parser.getOFracGrid()

        except BaseException as e:
            errmsg += '\n'+ParserClass.__name__+\
                      ' did not work- {}'.format(str(e))
            fxNet = None

        except:
            (t,v,tb) = sys.exc_info()
            print( "Unexpected error: {}\n{}\n\nTraceback:".format(t,v),
                    file=sys.stderr )
            traceback.print_tb(tb)
            sys.exit(-1)

        if fxNet:
            break

    if not fxNet:
        raise NotValidOFracGridError(
            f'ofracs.parse() failed on "{file_name}":\n'
            +errmsg
            +'\n\n' + '\n'.join(_import_warning_strings)
            )

    return fxNet

__DEBUG__ = False
__VERBOSITY__ = 0

# other valid options might be "ignore" "fail"
__BAD_FRACTURE_POLICY__ = "warn"

# set precision of decimal numbers
getcontext().prec = 12
N_COORD_DIG = Decimal('0.001')
N_APERT_DIG = Decimal('0.000001')

def D(v,new_prec):
    """Return a decimal with the specified quantization/precision"""

    ret = v

    try:
        ret = Decimal(v).quantize(new_prec)
    except decimal.InvalidOperation as e:
        if not v.is_finite():
            pass # return v
        else:
            raise ValueError(f'Cannot re-quantize {v}') from e
    except TypeError as e:
        if isinstance(v, np.floating):
            ret = Decimal(float(v)).quantize(new_prec)
        else:
            raise
    except Exception as e:
        raise ValueError(f'Argument {v} of type {type(v)}.') from e

    return ret


def D_CO(v):
    """Return a decimal with the required number of digits for coordinates"""

    return D(v,N_COORD_DIG)

def D_AP(v):
    """Return a decimal with the required number of digits for apertures"""
    return D(v,N_APERT_DIG)

def nudge(v,increment):
    """Nudge v to the nearest multiple of increment."""
    return ((v/increment).quantize(0) * increment).quantize(N_COORD_DIG)

DINF = Decimal('infinity')

def toDTuple(s):
    """Return a tuple of coordinate precision Decimals

    Arguments:
        s : str or list-like
            strs must look like a list of numbers tuple, like '(x,y,z)',
            'x y z', 'x,y z', or '[u, v, w x y z'
            list-likes must be a sequence of number-like things.
    """

    ivals = iter([0,0,0,])

    if type(s) is str:
        ivals = iter(re.sub("[(),]",' ',s).strip().split())
    elif hasattr(s,'__iter__'):
        ivals = iter(s)
    else:
        raise ValueError(f'"toDTuple" expected string or list-like, but got '
                '{type(s)}.')

    return tuple( D_CO(v) for v in ivals )

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

        self.d = 6*[0.,]
        """Position data:, xfrom, xto, yfrom ... zto."""

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


    def asRFGenStr(self,i=None):
        wid = 8
        s= '      '
        if i is not None:
            s = f'{i:6d}'
        s+= (6*' {:{w}}').format(*self.d, w=wid)
        s+= f' {self.ap:.10f}'
        #s+= f' {self.determineFracOrientation()+1}'
        return s

    def __repr__(self):
        return f'{type(self)} at {id(self):#x}'

    def nudge(self, nudgeIncrement):
        """Modify a fracture to new "nudged" coordinates.

        If the `nudgeIncrement` is zero, do nothing and return success (True).
        """

        returnstatus = True

        nudgeIncrement = D_CO(nudgeIncrement)

        if float(nudgeIncrement) == 0.:
            return True

        def myNudger(v):
            return nudge(v,nudgeIncrement)

        newd = tuple(map(myNudger, self.d))

        _policy = __FX_COLLAPSE_POLICY__
        if hasattr(self, 'myNet'):
            _policy=self.myNet.collapse_policy

        try:
            self._checkCollapse("nudging", newd)

        except FractureCollapseWarning as e:
            if _policy == 'fail':
                raise FractureCollapseError(
                    f'{e!s}\nFailing, due to ofracs.__FX_COLLAPSE_POLICY__') \
                    from e
            elif _policy == 'warn-omit':
                print(e, file=sys.stderr)
                returnstatus = False
            elif _policy == 'omit':
                returnstatus = False

        else:
            self.d = newd

        # invalidate the gridlines in the containing network
        if self.myNet is not None:
            self.myNet.invalidateGrid()

        return returnstatus

    class _Truncate_Op_Message:
        """A class to make a string about truncation.
        String formatting is done on-demand at time of use.
        This avoids creating and formatting a string that is never used.
        """
        __slots__ = ['s', 'e',]
        def __init__(self, s, e):
            self.s = s
            self.e = e
        def __str__(self):
            return "truncating to ({})->({})".format(
               ','.join(str(v) for v in self.s),
               ','.join(str(v) for v in self.e) )

    def truncate(self, s, e):
        """Modify a fracture's size to fit within a given bounding box

        s - the minimum coordinate of the bounding box (numeric-type triple)
        e - the maximum coordinate of the bounding box (numeric-type triple)

        Use "big values" in the s and e coordinates if you do not want to
        truncate in a particular direction.

        No error checking on valid inputs s and e!
        """

        domTruncStr = OFrac._Truncate_Op_Message(s,e)

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
        self.collapse_policy = 'warn-omit'

        self.metadata = {}

        # set origin
        # set size
        if domainOrigin is not None:
            domainOrigin = toDTuple(domainOrigin)
        if domainSize is not None:
            domainSize = toDTuple(domainSize)

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

        self._reCountFractures()

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
            if v < Decimal('0.0'):
                raise ValueError('Must have >=0 size values for domain')

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
        if not self._gridValid:
            raise RuntimeError(
                    "getBounds "
                    "called when _gridValid == False")
            #self._remakeGridLineLists()
        return copy.deepcopy(self._mima)

    def invalidateGrid(self):
        """Set the gridlines, boundaries, and fracture counts to be invalid."""
        self._gridValid = False

    def setDomainSize(self, domainOr, domSz):
        """Set the domain size, possibly exclude gridlines and fractures

        Arguments:
            domainOr : list-like
                Three numbers representing the new (x,y,z)-origin
            domSz : list-like
                Three numbers representing the new domain size (|x|,|y|,|z|)
        """

        domainOr = toDTuple(domainOr)
        domSz = toDTuple(domSz)

        self._setDomain(domainOr, domSz)

        # remove fixed gridlines outside of domain
        s = self.domainOrigin
        e = self.getDomainEnd()

        for a in range(3):
            glsToRemove = set()
            for gl in self._fixedgl[a]:
                if gl < s[a] or e[a] < gl:
                    glsToRemove.add(gl)
                    #warnings.warn(message,UserWarning)
                    if __VERBOSITY__ > 1:
                        message = "User-specified gridline at {}={} is being removed!".format('xyz'[a],gl)
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

            if not gla or gla[0] != s[a]:
                gla.insert(0,s[a])

            if gla[-1] != e[a]:
                gla.insert(len(gla),e[a])

            # store-back to self
            self._gl[a] = gla

        self._resetMinMax()
        self._remakeMinMax(useFixedGrid=True, useGrid=True)

        # truncate/readd fractures
        nNewFx = 0
        truncMsgs = []

        for i,f in enumerate(self._fx):
            try:
                f.truncate(s, e)
            except (FractureCollapseError,FractureCollapseWarning) as ce:
                # capture a string listing which fractures were truncated
                truncMsgs.append(ce)
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
                    len(truncMsgs)) )
            if __VERBOSITY__ > 1:
                _msg = '\n'.join( truncMsgs[0].message.split('\n')[:2] )
                for m in truncMsgs[1:]:
                    _msg += '\n'+m.message.split('\n')[1]
                print( _msg )


    def getDomainEnd(self):
        def sumInfGuarded(aList):
            if   DINF in map(abs, aList): return DINF
            else:                         return sum(aList)

        return tuple( map( sumInfGuarded, zip(self.domainOrigin,self.domainSize) ) )

    def scale(self, s):
        """Scale all gridlines and fractures.

        Arguments:
            s : list-like
                Three components of the scaling magnitude.
        """

        s = toDTuple(s)

        for ax,sc in enumerate(s):
           if sc == Decimal('0'):
                raise ValueError(f'Found scaling of zero in {"xyz"[ax]}')

        # move fractures
        for f in self._fx:
            newd = 6*[ None, ]
            for i,sc in zip(count(start=0,step=2),s):
                newd[i  ] = D_CO(f.d[i  ]*sc)
                newd[i+1] = D_CO(f.d[i+1]*sc)
            f.d = tuple(newd)

        self.domainOrigin = toDTuple(map(prod,zip(self.domainOrigin,s)))
        self.domainSize = toDTuple(map(prod,zip(self.domainSize,s)))

        # move grid
        for ax,sc in enumerate(s):

            # move mins and maxes
            for i in range(2):
                self._mima[ax][i] = D_CO(self._mima[ax][i]*sc)

            # times-equals
            def te(v):
                return D_CO(v*sc)

            # move fixed gridlines
            self._fixedgl[ax] = set(map(te,self._fixedgl[ax]))

            # move gridlines, inplace
            self._gl[ax][:] = map(te,self._gl[ax])


    def translate(self, t):
        """Translate all gridlines and fractures.

        Arguments:
            t : list-like
                Three components of the translation magnitude.
        """

        t = toDTuple(t)

        # move fractures
        for f in self._fx:
            newd = 6*[ None, ]
            for i,tv in zip(count(start=0,step=2),t):
                newd[i  ] = f.d[i  ]+tv
                newd[i+1] = f.d[i+1]+tv
            f.d = tuple(newd)

        newOrigin = list(self.domainOrigin) #mutable

        # move grid
        for ax,tv in enumerate(t):

            if tv == Decimal('0'):
                continue

            # plusequals
            def pe_tv(v):
                return v+tv

            # move origin
            newOrigin[ax] += tv

            # move mins and maxes
            self._mima[ax][0] += tv
            self._mima[ax][1] += tv

            # move fixed gridlines
            self._fixedgl[ax] = set(map(pe_tv,self._fixedgl[ax]))

            # move gridlines, inplace
            self._gl[ax][:] = map(pe_tv,self._gl[ax])

        self.domainOrigin = tuple(newOrigin)
        del newOrigin



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

    def delFracture( self, indexList ):
        """Delete fracture(s)

        Arguments:
            indexList : list of int
                Remove fractures at the given indices (based on order of
                OFracGrid.iterFracs
        """

        for i in reversed(sorted(indexList)):
            (o,v) = self._fx[i].determinePerpAxisVal()
            self._ocounts[o] -= 1
            del self._fx[i]


    def iterFracs(self):
        """iterate over fractures"""
        for f in self._fx:
            yield f

    def nudgeAll( self, nudgeTo ):
        """Nudge existing gridlines and all fractures to specified increment.

        Fixed gridlines are not nudged.

        Removes fractures or fails depending on __FX_COLLAPSE_POLICY__

        If the `nudgeTo` is zero, do nothing.
        """

        nudgeInc = D_CO(nudgeTo)

        if float(nudgeInc) == 0.:
            return 

        _gvsave = self._gridValid

        def nudger(v):
            return nudge(v,nudgeInc)

        for a in range(3):
            newGL = set(map(nudger, self._gl[a]))
            newGL.update(self._fixedgl[a])
            self._gl[a] = sorted(newGL)

        failedNudges = []
        for i,of in enumerate(self._fx):
            if not self._fx[i].nudge( nudgeTo ): failedNudges.append(i)

        for i in reversed(failedNudges):
            del self._fx[i]

        # A side-effect of nudging fractures is that the grid becomes invalid.
        # Because we just nudged the grid lines, the grid is ok now only if it
        # was ok before nudging.
        self._gridValid = _gvsave

    def getFxCount(self):
        """Return the number of fractures."""
        return len(self._fx)

    def getFxCounts(self):
        """Return a 3-tuple the number of fractures in each orientation.

        The 3-tuple has the order (N_yz,N_xz,N_xy), where the index in the tuple
        is the index of the axis perpendicular to the fracture.
        """
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
        """Add a gridline to the list of fixed gridlines.

        If it is outside the domain, the domain becomes bigger.
        """

        # maintains status of _gridValid by inserting gridline in the correct
        # spot in the  list of gridlines, and checking that the domain bounding
        # box is still accurate

        v = D_CO(glvalue)
        self._fixedgl[axis].add(v)

        if type(self._gl[axis]) == set:
            self._gl[axis].update(v)
        else:
            i = bisect_left(self._gl[axis], v)
            if i == len(self._gl[axis]):
                self._gl[axis].append(v)
            elif self._gl[axis][i] != v:
                self._gl[axis].insert(i,v)

        self._remakeMinMax(useFixedGrid=True)


    def getGridLines(self, axis='all'):
        """Return a list of grid lines

        Parameters
        ----------
        axis : `str` or `int`
            Default, 'all', which returns a 3-length list of arrays of grid
            lines for the x-, y-, and z-axis, respectively. Otherwise, returns
            an array of gridlines for the requested axis, as 0, 1, 2, or 'x',
            'y', or 'z'
        """
        if axis == 'all':
            return [ np.array(a) for a in self._gl ]
        elif axis in (0, 1, 2):
            return np.array(self._gl[axis])
        elif axis in ('x', 'y', 'z'):
            return np.array(self._gl['xyz'.index(axis)])
        raise ValueError(f'Cannot interperet "{axis}" as an axis')

    def getGridLineCounts(self):
        """Return a 3-tuple of counts grid lines"""
        return tuple( len(l) for l in self._gl )

    def getGridLineFirstInterval(self, axis):
        """Return the interval between first two gridlines of a given axis"""
        if not self._gridValid:
            raise RuntimeError(
                    "getGridLineFirstInterval "
                    "called when _gridValid == False")
            #self._remakeGridLineLists()
        return self._gl[axis][1] - self._gl[axis][0]

    def iterGridLines(self, axis):
        """Iterate through grid lines of a given axis"""

        # convert to integer axis
        if type(axis) == str:
            axis = 'xyz'.find(axis.lower())
        if axis < 0 or axis > 2: raise ValueError('bad axis')

        if not self._gridValid:
            raise RuntimeError(
                    "iterGridLines "
                    "called when _gridValid == False")
            #self._remakeGridLineLists()

        for v in self._gl[axis]:
            yield v

    def isUniformGridSpacing(self, axis):
        """Scan grid lines to determine if spacing is uniform"""
        if not self._gridValid:
            raise RuntimeError(
                    "isUniformGridSpacing " \
                    "called when _gridValid == False")
            #self._remakeGridLineLists()

        # convert to integer axis
        if type(axis) == str:
            axis = 'xyz'.find(axis.lower())
        if axis < 0 or axis > 2: raise ValueError('bad axis')

        # trivial
        if len(self._gl[axis]) <= 2: return True

        # check spacing between all pairs
        diff = self._gl[axis][1] - self._gl[axis][0]

        for i in range(2,len(self._gl[axis])):
            tdiff = self._gl[axis][i] - self._gl[axis][i-1]
            # fail early
            if abs(tdiff - diff) > 1e-6:
                return False
        return True

    def addRegularGlSpacing(self, spacing):
        """Add grid lines at regular intervals from the domainOrigin

        Arguments:
            spacing : list-like
                3-length list of the regular spacing increments. List items that
                are 'None' or zero will cause no change in grid lines in that
                axis.
        """

        for i in range(len(self._gl)):

            # allow for no change
            if not spacing[i]:
                continue

            # error check
            s = float(spacing[i])
            if s < 0.0:
                raise ValueError('Spacing cannot be less than zero')

            gls = set(self._gl[i])

            o = self.domainOrigin[i]
            ngl = int(floor(float(self.domainSize[i])/s))

            s = D_CO(s)
            gls.update( [ o+igl*s for igl in range(1, ngl) ] )

            self._gl[i] = list(sorted(gls))

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
                    nspac = ceil((l2-l1)/maxS)
                    spac = (l2-l1)/nspac
                    while l1 < l2-eps:
                        newGl.append( l1 )
                        l1 += spac
                else:
                    newGl.append( l1 )

            newGl.append(gla[-1])
            self._gl[a] = newGl[:]

        self._gridValid = True


    def refineNearFx(self, refList):
        """Add gridlines at specified distance(s) away from fracture planes

        Arguments:
            refList : list-like
                A sequence of cell sizes away from the fracture plane. e.g., a
                Fracture at 'F' and refList [ 'a', 'b', 'c' ] will have 7
                resultant grid lines F-(a+b+c), F-(a+b), F-a, F, F+a, F+a+b, and
                F+a+b+c.
        """

        # error check inputs
        errmsg = 'Cannot have negatively sized refinement intervals.'
        for v in refList:
            if v <= 0.0: raise ValueError(errmsg)

        # map inputs to Decimal type
        refList = list( D_CO(v) for v in refList )
        for i in range(1, len(refList)):
            refList[i] += refList[i-1]

        glSets = [ set(gll) for gll in self._gl ]

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


    def strDomFromTo(self):
        """Return a formatted string <from>-><to>"""
        st = numTuple2str(self.domainOrigin)
        en = numTuple2str(self.getDomainEnd())
        return f"{st}->{en}"

    def __str__(self):
        s = "Orthogonal Fracture Grid with:\n"

        def prod( i ):
            p = 1
            for v in i:
                p*=v
            return p

        stuff = {
            "Size":self.strDomFromTo(),

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

        _e = self.getFxCount()
        _n = 4*_e

        if printFileHeader:
            print(f'TITLE="DFN generated by OFracGrid ' \
                    f'({os.path.basename(__file__)})"\n' \
                    'VARIABLES="x","y","z","aperture"\n',
                    file=fout)

        for k,v in self.metadata.items():
            print(f'DATASETAUXDATA {k} = "{v}"', file=fout)

        if _e < 1:
            raise RuntimeError(
                "Domain has zero fractures! Cannot output a Tecplot zone.")

        # chunks of info for the header string
        znHdrString = ( f'ZONE T="{zoneName}"',
            'ZONETYPE=FEQUADRILATERAL DATAPACKING=BLOCK',
            f'NODES={_n} ELEMENTS={_e}',
            'VARLOCATION=([4]=CELLCENTERED)',
            f'AUXDATA numFracs = "{_e}"'
            )
        print(' '.join(znHdrString), file=fout)

        blockVals = [_n*[Decimal(0),],_n*[Decimal(0),],_n*[Decimal(0),],]

        for iel,f in enumerate(self.iterFracs()):
            for ino,(x,y,z) in enumerate(f.iterPoints()):
                blockVals[0][4*iel+ino] = x
                blockVals[1][4*iel+ino] = y
                blockVals[2][4*iel+ino] = z


        # print x y z-blocks
        for ia,a in enumerate('xyz'):
            print(f'# {a}', file=fout)
            #import pdb ; pdb.set_trace()
            for vals in zip( blockVals[ia][0::4],
                             blockVals[ia][1::4],
                             blockVals[ia][2::4],
                             blockVals[ia][3::4] ):
                v = ' '.join(str(v) for v in vals)
                print(f'{v}', file=fout)

        # print aperture block
        print('# apertures', file=fout)
        for fx in self.iterFracs():
            print(f'{fx.ap}', file=fout)

        # print element data
        print('# FE data', file=fout)
        w = floor(log10(_n))+1
        for iel in range(_e):
            # 1-based indices
            v = ' '.join(f'{n:{w}d}' for n in range(1+4*iel,4*iel+5))
            print(f'{v}', file=fout)


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
                        zip(newGrid.domainOrigin,newGrid.domainSize,
                            other.domainOrigin,other.domainSize)))

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

            for i in range(3):
                newGrid._ocounts[i] += other._ocounts[i]

            if __VERBOSITY__ > 4:
                print()


            sep = ', '
            for k,v in other.metadata.items():
                if k in newGrid.metadata:
                    newGrid.metadata[k] += f'{sep}{v}'
                else:
                    newGrid.metadata[k] = f'{v}'

        return newGrid

    def choose_nodes_block(self, block_spec):
        """Return a list of nodes within a bounding box

        Parameters
        ----------
        block_spec : array-like
            A 6-valued array of (xfrom, xto, yfrom, yto, zfrom, zto). A
            `str` of comma-separated values is also acceptable.

        Returns
        -------
        A 2-tuple of `numpy.array` of grid (porous medium) node indices and
        fracture node indices. Indices are 0-based.
        """

        grid = self

        if type(block_spec) == str:
            block_spec = re.sub(',',' ',block_spec).strip().split()
        elif len(block_spec) == 6:
            pass
        else:
            raise ValueError('block_spec must be interpretable as 6 floats')
        
        # loading zone 3D block
        coords = toDTuple(block_spec)
        (x1,x2,y1,y2,z1,z2) = coords

        # full domain:
        ngl = self.getGridLineCounts()
        gl = self.getGridLines()

        # layer index increments
        _lii = np.array([1, ngl[0], ngl[0]*ngl[1],])

        # loading zone grid line indices [inclusive, exclusive)
        # [ [ix1, ix2), [iy1, iy2), [iz1, iz2), ]
        lzgl = -np.ones(6, dtype=int)

        icoord = iter(coords)
        for axis in range(3):
            lzgl[2*axis  ]= bisect_left(gl[axis],next(icoord))
            lzgl[2*axis+1]= bisect_right(gl[axis],next(icoord),lo=lzgl[2*axis])

        # trim domain to store just the relevant fractures (reducing the search
        # space)
        # Note that the 'max(N_COORD_DIG...' might select a domain outside the
        # original domain if the given range is on the upper bound of its
        # axis. Therefore, guard each value of the origin so the cutout domain
        # is a sub-zone of the orignal
        cutout = copy.deepcopy(self)
        cutout_o = [ x1, y1, z1 ]
        cutout_sz = list(max(N_COORD_DIG,j-i) for i,j in iterpairs(coords))
        for i,o,v in zip(count(), cutout_o, cutout_sz):
            if o + v > cutout._mima[i][1]:
                cutout_o[i] -= v
        cutout.setDomainSize(cutout_o, cutout_sz)

        # determine the pm nodes list
        def _2slices(blk):
            return [ np.s_[blk[0]:blk[1]],
                     np.s_[blk[2]:blk[3]],
                     np.s_[blk[4]:blk[5]], ]

        def _get_indices_in_gl_block(blk):
            '''Returns all indices, given ranges (is, ie, js, je, ks, ke)'''
            ret = -np.ones(np.prod(blk[1::2]-blk[::2]), dtype=int)
            for i, ijk in enumerate(itertools.product(
                    *[_lsz*np.ogrid[_s]
                        for (_s,_lsz) in reversed(list(zip(_2slices(blk), _lii)))])):
                ret[i] = np.sum(ijk)
            return ret

        pmnodes = _get_indices_in_gl_block(lzgl)

        # determine the fracture nodes
        fxnodes = set()

        # iter fractures and record the node numbers, noting HGS' fortran-style
        # 1-based indexing and x-fastest/z-slowest
        for f in cutout.iterFracs():

            # fracture starting and ending gridlines
            fgl = np.zeros(6, dtype=int)

            for axis,(v1,v2),(g1,g2) in \
                zip(count(),iterpairs(f.d),iterpairs(lzgl)):

                i1 = bisect_left(gl[axis],v1,lo=g1,hi=g2)
                i2 = bisect_right(gl[axis],v2,lo=i1,hi=g2)

                fgl[2*axis  ]= i1
                fgl[2*axis+1]= i2

            fxnodes.update(_get_indices_in_gl_block(fgl))

        return (pmnodes, np.array(sorted(fxnodes),dtype=int))

     
    def ng2ni(self, ng):
        '''Convert each (i, j, k)-row from grid to node index'''
            
        ngl = self.getGridLineCounts()
        _lii = np.array([1, ngl[0], ngl[0]*ngl[1],])

        if isinstance(ng, np.ndarray):
            return np.dot(ng, _lii[:,np.newaxis]).squeeze()

        raise NotImplementedError()

    def ni2ng(self, ni):
        '''Convert each node index value to (i, j, k) grid index'''

        ngl = self.getGridLineCounts()
        lii = np.array([1, ngl[0], ngl[0]*ngl[1],])

        if isinstance(ni, np.ndarray):
            ret = np.zeros((ni.size,3), dtype=int)
            np.divmod(ni, lii[2], ret[:,2], ret[:,1])
            np.divmod(ret[:,1], lii[1], ret[:,1], ret[:,0])
            return ret

        raise NotImplementedError()
 
    @staticmethod
    def pickleTo( ofracObj, f ):
        """Dump to the given filename/file"""
        if type(f) in [ str, os.PathLike ]:
            with open(f, 'wb') as fout:
                pickle.dump(ofracObj, fout, pickle.HIGHEST_PROTOCOL)
        else:
            pickle.dump(ofracObj, f, pickle.HIGHEST_PROTOCOL)

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

