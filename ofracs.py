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

__docformat__='numpy'

_import_warning_strings = []
"""An array of warnining messages about parsers that have not been found."""

def populate_parsers():
    """Return a list of OFracGrid parser options.

    The list will depen on what other packages are on the system and  accessible
    in PYTHONPATH, like FRACTRAN, HGS, RFGen, or Compflow, which may have
    parsers of orthogonal fracture networks.
    """

    #ret = [ OFracGrid.PickleParser, OFracGrid.LegacyUnpickler, ]
    ret = [ OFracGrid.PickleParser, ]

    try:
        import hgstools.pyhgs.parser.fractran as _hgs_parser_fractran
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
        import hgstools.pyhgs.parser.rfgen as _hgs_parser_rfgen
    except ModuleNotFoundError as e:
        if "No module named 'pyhgs'" in str(e):
            pass
        elif 'pyhgs.parser.rfgen' not in str(e):
            raise e
        _import_warning_strings.append(
            "Warning: did not find 'hgstools.pyhgs.parser.parser_rfgen'. "
            +"Cannot parse RFGen-type orthogonal fracture networks."
            )
    else:
        ret += [
            _hgs_parser_rfgen.RFGenOutFileParser,
            _hgs_parser_rfgen.RFGenFracListParser,
        ]

    try:
        import hgstools.pyhgs.parser.eco as _hgs_parser_eco
    except ModuleNotFoundError as e:
        if "No module named 'pyhgs'" in str(e):
            pass
        elif 'pyhgs.parser.eco' not in str(e):
            raise e
        _import_warning_strings.append(
            "Warning: did not find 'hgstools.pyhgs.parser.eco'. "
            +"Cannot parse HGS+RFGen-style orthogonal fracture networks."
            )
    else:
        ret += [_hgs_parser_eco.EcoFile,]

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


N_COORD_EXP = 3
"""Number of decimal digits in `N_COORD_DIG`; `N_COORD_DIG == 10**-N_COORD_EXP`"""

N_APERT_EXP = 6
"""Number of decimal digits in `N_APERT_DIG`; `N_APERT_DIG == 10**-N_APERT_EXP`"""

CO_SCALE = 10**N_COORD_EXP
"""Number of integer storage units per unit of coordinate length"""

AP_SCALE = 10**N_APERT_EXP
"""Number of integer storage units per unit of aperture"""

STORE_DTYPE = np.int32
"""`numpy` dtype of the integer-quantized fracture coordinate and aperture stores

Fractures are stored as integer counts of `N_COORD_DIG` (and `N_APERT_DIG`)
rather than as `Decimal` or floating point values. Integers preserve the exact
equality and exact arithmetic that the rest of this module depends upon --- a
fracture's orientation is found by testing two of its coordinates for equality,
and grid lines are matched to fracture faces through `set` membership --- while
allowing the whole network to be queried with vectorized `numpy` operations.

At 32 bits this spans +/-2,147,483.647 m of coordinate, which is far more than
the physical domains involved. Values arriving from outside raise
`OverflowError` when they exceed it; `OFracGrid.scale` and `OFracGrid.translate`
check their results explicitly, because `numpy` would otherwise wrap silently.
"""

_WIDE_CTX = decimal.Context(prec=40)
"""A context wide enough that re-scaling a quantized `Decimal` cannot round it

The module-wide precision (12) is enough to *hold* a coordinate, but shifting
that coordinate's decimal point to make an integer needs a few more digits.
"""

def _co2i(v):
    """Return coordinate `v` as an exact integer count of `N_COORD_DIG` units"""
    return int(D_CO(v).scaleb(N_COORD_EXP, _WIDE_CTX))

def _i2co(i):
    """Return a coordinate `Decimal` from an integer count of `N_COORD_DIG` units"""
    # scaling an integer by 10**-N_COORD_EXP lands on the coordinate precision
    # exactly, so no further quantization is needed
    return Decimal(int(i)).scaleb(-N_COORD_EXP, _WIDE_CTX)

_CO_INT_LIMIT = Decimal(np.iinfo(STORE_DTYPE).max) / CO_SCALE
"""Largest coordinate magnitude the integer store can hold"""

def _chk_co_range(a, operation):
    """Raise if any value in `a` has left the range `STORE_DTYPE` can hold

    `numpy` integer arithmetic wraps around silently, so operations that move
    fractures check their results here rather than store a corrupt coordinate.
    """
    ii = np.iinfo(STORE_DTYPE)
    if a.size and (a.min() < ii.min or a.max() > ii.max):
        raise ValueError(
            f'{operation} put a fracture outside the '
            f'+/-{ii.max/CO_SCALE:,.3f} coordinate range that '
            f'{np.dtype(STORE_DTYPE).name} fracture storage allows')

def _bound2i(v, default):
    """Return bounding-box coordinate `v` in integer units, or `default`

    Bounding boxes may be given as infinities, or as the enormous floats used
    to mean "do not truncate in this direction"; neither is representable in
    the store, and `default` (a saturating integer bound) stands in for them.
    """
    try:
        if abs(Decimal(v)) > _CO_INT_LIMIT:
            return default
    except (TypeError, ValueError, OverflowError, decimal.InvalidOperation):
        return default

    return _co2i(v)

def _ap2i(v):
    """Return aperture `v` as an exact integer count of `N_APERT_DIG` units"""
    return int(D_AP(v).scaleb(N_APERT_EXP, _WIDE_CTX))

def _i2ap(i):
    """Return an aperture `Decimal` from an integer count of `N_APERT_DIG` units"""
    return Decimal(int(i)).scaleb(-N_APERT_EXP, _WIDE_CTX)


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



class OFracArray():
    """Array-backed store of orthogonal fractures.

    This replaces the `list` of `OFrac` objects that an `OFracGrid` used to
    hold. Coordinates and apertures live in `numpy` integer arrays --- see
    `STORE_DTYPE` for why integers --- alongside a helper array caching each
    fracture's orientation, which the coordinates alone do not state directly.

    The store deliberately behaves like the `list` it replaces: it has a
    length, it is iterable, `store[i]` returns an `OFrac`, and `append`,
    `del store[i]` and `del store[i:]` all work. Unlike a `list`, the `OFrac`
    returned by indexing is a *view* onto a row of the arrays: mutating it
    mutates the store, and it is invalidated by any later insertion or deletion.

    Attributes
    ----------
    net : `OFracGrid` or `None`
        The network owning these fractures, reported as `OFrac.myNet`.
    """

    __slots__ = ('_d', '_ap', '_perp', '_n', 'net',)

    _MIN_CAPACITY = 8
    """Smallest number of rows allocated once a store becomes non-empty"""

    def __init__(self, net=None, capacity=0):
        self._d = np.zeros((max(capacity,0), 6), dtype=STORE_DTYPE)
        """Coordinates, in `N_COORD_DIG` units: xfrom, xto, yfrom ... zto"""

        self._ap = np.zeros(max(capacity,0), dtype=STORE_DTYPE)
        """Apertures, in `N_APERT_DIG` units"""

        self._perp = np.full(max(capacity,0), -1, dtype=np.int8)
        """Helper: index of the axis perpendicular to each fracture, or -1"""

        self._n = 0
        """Number of fractures stored; rows at and beyond this are unused"""

        self.net = net

    # live (as opposed to allocated) views of the stored data
    @property
    def coords(self):
        """An (N,6) `numpy.array` of coordinates, in `N_COORD_DIG` units.

        This is a view: writing to it writes to the store, and it is
        invalidated by any operation that changes the store's length.
        """
        return self._d[:self._n]

    @property
    def apertures(self):
        """An (N,) `numpy.array` of apertures, in `N_APERT_DIG` units (a view)"""
        return self._ap[:self._n]

    @property
    def perp_axes(self):
        """An (N,) `numpy.array` of perpendicular axis indices (a view)"""
        return self._perp[:self._n]

    @property
    def perp_vals(self):
        """An (N,) `numpy.array` of fracture plane coordinates, derived

        The coordinate of a fracture's plane is just whichever of its 'from'
        coordinates lies on the perpendicular axis, so it is computed here
        rather than stored. Unlike the other array properties this is a fresh
        array, not a view.
        """
        n = self._n
        if n == 0:
            return np.zeros(0, dtype=STORE_DTYPE)

        return np.take_along_axis(self._d[:n, 0::2],
            np.maximum(self._perp[:n], 0).astype(np.intp)[:, np.newaxis],
            axis=1).squeeze(axis=1)

    # list-like interface
    def __len__(self):
        return self._n

    def __iter__(self):
        for i in range(self._n):
            yield self._view(i)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [ self._view(i) for i in range(*key.indices(self._n)) ]
        return self._view(self._chkindex(key))

    def __setitem__(self, i, f):
        """Overwrite fracture `i` with a copy of the `OFrac` `f`"""
        i = self._chkindex(i)
        (o, j) = (f._store, f._i)
        self._d[i] = o._d[j]
        self._ap[i] = o._ap[j]
        self._perp[i] = o._perp[j]

    def __delitem__(self, key):
        if isinstance(key, slice):
            self.delete(range(*key.indices(self._n)))
        else:
            self.delete((key,))

    def append(self, f):
        """Append a copy of the `OFrac` `f`; return the index it was given"""
        return self._append_row(f._store._d[f._i], f._store._ap[f._i])

    def append_values(self, *vals):
        """Append one fracture from xfrom, xto, yfrom ... zto, ap

        Returns the index it was given. Raises `FractureDefinitionError` if the
        values do not describe a plane. This skips the temporary single-row
        store that building an `OFrac` first would need.
        """
        return self._append_row(*OFrac.__check_vals__(*vals))

    def reserve(self, n):
        """Pre-allocate room for `n` fractures in total"""
        self._reserve(n)

    def extend(self, other):
        """Append copies of every fracture in the `OFracArray` `other`"""
        m = len(other)
        if m == 0:
            return

        n = self._n
        self._reserve(n+m)
        self._d[n:n+m] = other.coords
        self._ap[n:n+m] = other.apertures
        self._perp[n:n+m] = other.perp_axes
        self._n = n+m

    def truncate_all(self, s, e):
        """Fit every fracture into the bounding box `s`->`e`

        The vectorized equivalent of calling `OFrac.truncate` on each fracture
        in turn. If any fracture falls outside the box, or would be clipped out
        of existence, that fracture is put back through `OFrac.truncate` so the
        `FractureCollapseWarning` is raised exactly as it otherwise would be.

        Parameters
        ----------
        s : array-like
            The minimum coordinate of the bounding box (numeric-type triple)
        e : array-like
            The maximum coordinate of the bounding box (numeric-type triple)
        """

        n = self._n
        if n == 0:
            return

        d = self._d[:n]
        ii = np.iinfo(STORE_DTYPE)
        lo = np.fromiter((_bound2i(v, ii.min) for v in s),
                dtype=STORE_DTYPE, count=3)
        hi = np.fromiter((_bound2i(v, ii.max) for v in e),
                dtype=STORE_DTYPE, count=3)

        # find every fracture the box cannot accommodate, in one pass, so that
        # the one reported is the same one the per-fracture path would reach
        bad = np.zeros(n, dtype=bool)
        for a in range(3):
            (v1, v2) = (d[:, 2*a], d[:, 2*a+1])
            plane = (v1 == v2)

            # a fracture whose plane lies outside the box is gone entirely
            bad |= plane & ((v1 < lo[a]) | (v1 > hi[a]))
            # ...as is one clipped down to nothing
            bad |= ~plane & (
                np.maximum(v1, lo[a]) >= np.minimum(v2, hi[a]))

        if bad.any():
            # re-run the offending fracture to raise with the usual message
            self._view(int(np.argmax(bad))).truncate(s, e)

        np.maximum(d[:, 0::2], lo, out=d[:, 0::2])
        np.minimum(d[:, 1::2], hi, out=d[:, 1::2])

        self._refresh_all()

    def delete(self, indices):
        """Remove the fractures at `indices`, an iterable of `int`

        Repeated indices are removed once. Indices may be given in any order.
        """

        doomed = set(self._chkindex(i) for i in indices)
        if not doomed:
            return

        keep = np.ones(self._n, dtype=bool)
        keep[list(doomed)] = False
        n = int(np.count_nonzero(keep))

        # note the right hand sides are fancy-indexed copies, so these
        # assignments back into the same arrays do not alias
        self._d[:n] = self._d[:self._n][keep]
        self._ap[:n] = self._ap[:self._n][keep]
        self._perp[:n] = self._perp[:self._n][keep]
        self._n = n

    # internals
    def _chkindex(self, i):
        """Return `i` as a non-negative, in-range row index"""
        i = int(i)
        if i < 0:
            i += self._n
        if not 0 <= i < self._n:
            raise IndexError(f'No fracture {i} among {self._n} fractures')
        return i

    def _view(self, i):
        """Return an `OFrac` view of row `i` (no bounds checking)"""
        f = OFrac.__new__(OFrac)
        f._store = self
        f._i = i
        return f

    def _reserve(self, n):
        """Grow the allocation, if needed, to hold at least `n` fractures"""
        cap = self._ap.size
        if n <= cap:
            return

        newcap = max(n, 2*cap, self._MIN_CAPACITY)
        live = self._n

        def _grown(a, fill=0):
            b = np.full((newcap,)+a.shape[1:], fill, dtype=a.dtype)
            b[:live] = a[:live]
            return b

        self._d = _grown(self._d)
        self._ap = _grown(self._ap)
        self._perp = _grown(self._perp, fill=-1)

    def _append_row(self, d, ap):
        """Append raw integer coordinates and aperture; return the new index"""
        i = self._n
        self._reserve(i+1)
        self._n = i+1
        self._d[i] = d
        self._ap[i] = ap
        self._refresh(i)
        return i

    def _refresh(self, i):
        """Recompute the orientation helpers for fracture `i`"""
        row = self._d[i]
        for a in range(3):
            if row[2*a] == row[2*a+1]:
                self._perp[i] = a
                return
        self._perp[i] = -1

    def _refresh_all(self):
        """Recompute the orientation helpers for every fracture"""
        n = self._n
        if n == 0:
            return

        d = self._d[:n]
        # a fracture's plane is perpendicular to the first axis whose 'from'
        # and 'to' coordinates are equal
        eq = d[:, 0::2] == d[:, 1::2]
        self._perp[:n] = np.where(eq.any(axis=1), eq.argmax(axis=1), -1)

    # pickling; store only the rows in use, not the spare capacity
    def __getstate__(self):
        n = self._n
        return {
            '_d': self._d[:n].copy(),
            '_ap': self._ap[:n].copy(),
            '_perp': self._perp[:n].copy(),
            '_n': n,
            'net': self.net,
        }

    def __setstate__(self, state):
        for key, val in state.items():
            setattr(self, key, val)

    def __repr__(self):
        return f'<{type(self).__name__} of {self._n} fractures at {id(self):#x}>'


# class for OFrac objects
class OFrac():
    """An orthogonal fracture object

    An `OFrac` is a lightweight *view* onto one row of an `OFracArray`: it
    stores only that array and a row index, and its `d` and `ap` attributes
    read and write the array on demand. A fracture built directly, rather than
    obtained from an `OFracGrid`, gets its own single-row store; adding it to a
    grid copies its row into the grid's store.
    """

    __slots__ = ('_store', '_i',)

    def __init__(self, *vals, **kwargs):

        net = kwargs.get('myNet', None)

        if vals:
            # assume initializing from xfrom, xto, yfrom, yto, zfrom, zto, ap
            (d, ap) = OFrac.__check_vals__( *vals )
        else:
            # assume initializing from another OFrac object
            other = kwargs['fromOFrac']
            d = other._store._d[other._i].copy()
            ap = other._store._ap[other._i]
            if 'myNet' not in kwargs:
                net = other.myNet

        self._store = OFracArray(net, capacity=1)
        self._i = self._store._append_row(d, ap)

    @staticmethod
    def __check_vals__(xfrom, xto, yfrom, yto, zfrom, zto, ap):
        """Check axis-aligned fracture values.

        Returns the quantized integer coordinates and aperture, ready to be
        stored. Raises `FractureDefinitionError` if the values are not a plane.
        """

        d = tuple(_co2i(v) for v in (xfrom, xto, yfrom, yto, zfrom, zto,))
        ap = _ap2i(ap)

        # calculate the size of the fracture in each dimension
        # make sure there are are two good-length sides
        countSameOrBad = ( int(Decimal(xto-xfrom)<N_COORD_DIG)
            + int(Decimal(yto-yfrom)<N_COORD_DIG)
            + int(Decimal(zto-zfrom)<N_COORD_DIG) )

        countBad = int(xto < xfrom) + int(yto < yfrom) + int(zto < zfrom)

        if countSameOrBad > 1 or countBad > 0:
            raise FractureDefinitionError(
                'not a plane: '+OFrac._strvals(map(_i2co, d), _i2ap(ap)) )

        return (d, ap)

    @property
    def d(self):
        """Position data:, xfrom, xto, yfrom ... zto.

        A 6-tuple of coordinate-precision `Decimal`. Assigning a 6-length
        sequence of numbers writes the (quantized) values back to the store.
        """
        return tuple(map(_i2co, self._store._d[self._i]))

    @d.setter
    def d(self, vals):
        vals = tuple(vals)
        if len(vals) != 6:
            raise ValueError(f'Need 6 coordinates, but got {len(vals)}')

        self._store._d[self._i] = tuple(map(_co2i, vals))
        self._store._refresh(self._i)

    @property
    def ap(self):
        """The fracture's aperture, as an aperture-precision `Decimal`"""
        return _i2ap(self._store._ap[self._i])

    @ap.setter
    def ap(self, v):
        self._store._ap[self._i] = _ap2i(v)

    @property
    def myNet(self):
        """The `OFracGrid` containing this fracture, or `None`"""
        return self._store.net

    @myNet.setter
    def myNet(self, net):
        if net is not self._store.net and len(self._store) > 1:
            raise ValueError(
                'Cannot re-assign the network of a fracture held in a grid')
        self._store.net = net

    def _checkCollapse(self, operation, newd, d=None):
        # check for fracture "collapse" due to rounding
        # ('d' is this fracture's current coordinates; callers that already
        # hold them pass them in rather than pay to rebuild them)
        if d is None:
            d = self.d
        for a in range(3):
            if d[2*a] != d[2*a+1] and newd[2*a] >= newd[2*a+1] :
                errmsg = 'Fracture collapsed during {}!:\n'.format(operation)
                errmsg +='before - {}\n'.format(OFrac._strvals(d, self.ap))
                errmsg +='after  - {}'.format(OFrac._strvals(newd, self.ap))
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
        o = int(f._store._perp[f._i])
        if o < 0:
            raise RuntimeError(
                'Could not determine orientation of {}'.format(f))
        return o

    def determinePerpAxisVal(self):
        """Return the axis perpendicular to this fracture's plane, and its value"""
        o = OFrac.determineFracOrientation(self)
        return ( o, _i2co(self._store._d[self._i, 2*o]), )

    @staticmethod
    def _strvals(d, ap, wid=8):
        """Return the standard string for the given coordinates and aperture"""
        return '({:{w}}->{:{w}}, {:{w}}->{:{w}}, {:{w}}->{:{w}}), ap='.format(
                *d, w=wid)+str(ap)

    def __str__(self,wid=8):
        return OFrac._strvals(self.d, self.ap, wid=wid)

    def __getitem__(self, i):
        if i < 6: return self.d[i]
        if i == 6: return self.ap
        raise IndexError(f'{__class__} has no index {i}')

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

        d = self.d
        newd = tuple(map(myNudger, d))

        _policy = __FX_COLLAPSE_POLICY__
        if self.myNet is not None:
            _policy=self.myNet.collapse_policy

        try:
            self._checkCollapse("nudging", newd, d)

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

        Use "big values" in the s and e coordinates if you do not want to
        truncate in a particular direction.

        No error checking on valid inputs s and e!

        Parameters
        ----------
        s : array-like
            The minimum coordinate of the bounding box (numeric-type triple)
        e : array-like
            The maximum coordinate of the bounding box (numeric-type triple)
        """

        domTruncStr = OFrac._Truncate_Op_Message(s,e)

        d = self.d
        newd = []
        for a,((v1,v2),mi,ma) in enumerate(zip(iterpairs(d), s, e)) :
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

        newd = tuple( newd )

        # A fracture wholly inside the bounding box is the common case; leave
        # the store untouched rather than convert the same coordinates back in.
        # (It cannot have collapsed: its coordinates have not changed.)
        if newd == d:
            return self

        self._checkCollapse( domTruncStr, newd, d )

        self.d = newd
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

    def __getstate__(self):
        return { '_store':self._store, '_i':self._i, }

    def __setstate__(self, state):
        """
        __setstate__ is called when unpickling to set the object's state.
        We need to implement it to handle the __dict__ from the old object.
        """
        # The 'state' dictionary contains the attributes from the pickled object's __dict__.
        if type(state) == tuple:
            state = state[1]
        elif hasattr(state,'items'):
            pass
        else:
            raise ValueError('Bad state: type={type(state)}, value={state!s}')

        if '_store' in state:
            self._store = state['_store']
            self._i = state['_i']
            return

        # A fracture pickled before fractures were array-backed: it carried its
        # own 'd' and 'ap' Decimals. Give it a private store to live in.
        self._store = OFracArray(state.get('myNet', None), capacity=1)
        self._i = self._store._append_row(
            np.fromiter((_co2i(v) for v in state['d']),
                dtype=STORE_DTYPE, count=6),
            _ap2i(state['ap']))

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

        self._fx = OFracArray(self)
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

        # add fractures, straight into the store
        try:
            self._fx.reserve(len(fx))
        except TypeError:
            pass # fx has no len(); let the store grow as it goes

        for i,f in enumerate(fx):
            try:
                self._fx.append_values( *f )
            except FractureDefinitionError as err:
                if __BAD_FRACTURE_POLICY__ == 'warn':
                    messages.append(f'Fracture {i} in inputs is bad: {err!s}')
                elif __BAD_FRACTURE_POLICY__ == 'ignore':
                    pass
                else:
                    raise RuntimeError( f'ABORT: Fracture {i} bad: {err!s}' )

        self._fx.truncate_all(s,e)

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

        if 'useFx' in kwargs and kwargs['useFx'] and len(self._fx) > 0:
            d = self._fx.coords
            for i in range(3):
                self._mima[i][0] = min(
                    self._mima[i][0], _i2co(d[:, 2*i  ].min()))
                self._mima[i][1] = max(
                    self._mima[i][1], _i2co(d[:, 2*i+1].max()))


    def _remakeGridLineLists(self, keep_glAsSets=False):
        """Scan fractures and make re-make gridlines & counts"""

        beforeCounts = self.getGridLineCounts()

        # reset grid, copy in set objects
        self._gl = [ s.copy() for s in self._fixedgl ]
        self._resetMinMax()
        self._remakeMinMax( useFixedGrid=True )

        if len(self._fx) > 0:
            d = self._fx.coords

            # add gridlines; np.unique collapses the (many) repeated fracture
            # face coordinates before any Decimal is built
            for a in range(3):
                self._gl[a].update(map(_i2co, np.unique(d[:, 2*a:2*a+2])))

            # determine fx net min/max coordinate
            self._remakeMinMax( useFx=True )

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
        # rescan fractures, using the store's cached orientations
        perp = self._fx.perp_axes

        if np.any(perp < 0):
            bad = int(np.argmax(perp < 0))
            raise RuntimeError(
                'Could not determine orientation of {}'.format(self._fx[bad]))

        self._ocounts = [ int(c) for c in np.bincount(perp, minlength=3) ]

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


    def getDomainStart(self):
        return tuple( self.domainOrigin )

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
        if len(self._fx) > 0:
            d = self._fx.coords
            for ax,sc in enumerate(s):
                # np.rint rounds halves to even, matching Decimal's default
                scaled = np.rint(d[:, 2*ax:2*ax+2] * float(sc))
                _chk_co_range(scaled, 'Scaling')
                d[:, 2*ax:2*ax+2] = scaled.astype(STORE_DTYPE)
            self._fx._refresh_all()

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
        if len(self._fx) > 0:
            d = self._fx.coords
            for ax,tv in enumerate(t):
                # exact in integer storage units, but widen before adding so
                # that leaving the storage range is caught rather than wrapped
                shifted = d[:, 2*ax:2*ax+2].astype(np.int64) + _co2i(tv)
                _chk_co_range(shifted, 'Translation')
                d[:, 2*ax:2*ax+2] = shifted
            self._fx._refresh_all()

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
                # keep orientation counts in sync (constructor recounts, but
                # addFracture must maintain them incrementally)
                self._ocounts[OFrac.determineFracOrientation(cf)] += 1
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

        doomed = sorted(set(indexList))

        for i in doomed:
            (o,v) = self._fx[i].determinePerpAxisVal()
            self._ocounts[o] -= 1

        self._fx.delete(doomed)


    def iterFracs(self):
        """iterate over fractures"""
        return iter(self._fx)

    def getFxCoordinates(self):
        """Return an (N,6) `numpy.array` of fracture coordinates, as `float`

        Columns are xfrom, xto, yfrom, yto, zfrom, zto. The array is a fresh
        copy; changing it does not change the network.
        """
        return self._fx.coords * (1.0/CO_SCALE)

    def getFxApertures(self):
        """Return an (N,) `numpy.array` of fracture apertures, as `float`

        The array is a fresh copy; changing it does not change the network.
        """
        return self._fx.apertures * (1.0/AP_SCALE)

    def getFxPerpAxes(self):
        """Return an (N,) `numpy.array` of each fracture's perpendicular axis

        Values are 0 for a yz-plane fracture, 1 for xz, and 2 for xy, matching
        `OFrac.determineFracOrientation`. The array is a fresh copy.
        """
        return self._fx.perp_axes.astype(int)

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
            if not of.nudge( nudgeTo ): failedNudges.append(i)

        if failedNudges:
            self._fx.delete(failedNudges)
            # dropped fractures must not be left in the orientation counts
            self._reCountFractures()

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
            *others:'OFracGrid'
        ):
        """Merge this with these others and return a new OFracGrid

        Parameters
        ----------
        others : `ofrac.ofracs.OFracGrid`
            Other grid objects to be merged-in to this one.

        """


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

            if __VERBOSITY__ > 4:
                for i,f in enumerate(other.iterFracs()):
                    if __VERBOSITY__ > 5:
                        print(f'adding fracture #{i}: {f}')
                    else:
                        print('.',end='')

            newGrid._fx.extend( other._fx )

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
 
    def __setstate__(self, state):
        """Restore an unpickled network.

        Networks pickled before fractures were array-backed hold `_fx` as a
        `list` of `OFrac`; convert it to an `OFracArray`.
        """

        self.__dict__.update(state)

        if not isinstance(state.get('_fx', None), OFracArray):
            oldfx = state.get('_fx', None) or []
            self._fx = OFracArray(self, capacity=len(oldfx))
            for f in oldfx:
                if isinstance(f, OFrac):
                    self._fx.append(f)
                else:
                    # some other fracture object from an older version of this
                    # module; take it at its 'd' and 'ap'
                    self._fx._append_row(
                        tuple(_co2i(v) for v in f.d), _ap2i(f.ap))

        self._fx.net = self

    @staticmethod
    def pickleTo( ofracObj, f ):
        """Dump to the given filename/file"""
        if type(f) in [ str, os.PathLike ]:
            with open(f, 'wb') as fout:
                pickle.dump(ofracObj, fout, pickle.HIGHEST_PROTOCOL)
        else:
            pickle.dump(ofracObj, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def unpickleFrom(filename):
        ret = None
        try:
            with open(filename, 'rb') as fin:
                ret = pickle.load(fin)
        except ModuleNotFoundError as e:
            # if the pathing of the pickled module was different than the
            # current pathing, add some aliases....
            if str(e) == "No module named 'ofrac.ofracs'":
                sys.modules['ofrac.ofracs'] = sys.modules[__name__]
            elif str(e) == "No module named 'ofracs'":
                sys.modules['ofracs'] = sys.modules[__name__]
            else:
                raise # something different... raise it
            # try again, or return whatever error was found before
            with open(filename, 'rb') as fin:
                ret = pickle.load(fin)

        return ret

    class PickleParser:
        def __init__(self, fnin):
            self.fnin = fnin
            try:
                self.myGrid = OFracGrid.unpickleFrom(self.fnin)
            except ModuleNotFoundError as e:
                raise NotValidOFracGridError(
                    f'Failed unpickling from {self.fnin}: {e!s}')
            except Exception as e:
                raise NotValidOFracGridError(
                    f'Failed unpickling from {self.fnin}: {type(e)!s}')

        def getOFracGrid(self):
            return self.myGrid


    class LegacyUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            breakpoint()
            # Example: Handle a renamed class
            if module == "old_module_name" and name == "OldClassName":
                module = "new_module_name"
                name = "MyCustomClass" # Assuming MyCustomClass is the new name

            # Allow unpickling of specific classes only
            if module == "__main__" and name == "MyCustomClass":
                return super().find_class(module, name)
            
            # Optionally, raise an error or return a placeholder for unknown classes
            raise pickle.UnpicklingError(f"Global unpickling of {module}.{name} is disallowed.")
