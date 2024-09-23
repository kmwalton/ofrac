"""Support for determining matrix blocks in an OFracGrid"""

import os
import datetime
import warnings
import re
from math import floor, ceil
from operator import attrgetter, mul
from itertools import combinations, product, cycle
from functools import reduce
from time import perf_counter

import scipy
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import ofracs

import logging
logger = logging.getLogger(__name__)
_INFO1 = logging.INFO+1

class MatrixBlockOFracGrid(ofracs.OFracGrid):

    def __init__(self, ofracgrid_obj):
        """Create a MatrixBlockOFracGrid from an existing OFracGrid"""
        self.__dict__.update(ofracgrid_obj.__dict__)

        nperpax = self.getFxCounts()

        logger.info(f'Determining matrix blocks in {ofracgrid_obj}-sized grid')

        obinfx = list(
                -np.ones(nperpax[i], dtype=int) for i in range(3))
        locfx = list(
                np.full(nperpax[i], np.inf, dtype=np.single)
                    for i in range(3))
        nextind = 3*[0,]

        _t0 = perf_counter()

        # make bins for each fracture orientation;
        # classify fractures
        for ifrac,frac in enumerate(self.iterFracs()):
            (perpaxis, axisval) = frac.determinePerpAxisVal()
            iunsorted = nextind[perpaxis]
            obinfx[perpaxis][iunsorted] = ifrac
            locfx[perpaxis][iunsorted] = axisval
            nextind[perpaxis] += 1

        logger.debug(
            f'Put {nextind} fractures into yz, xz, xy orientation bins')
        logger.debug('Sorting fracture bins...')

        self.sorted_fx = 3*[None,]
        self.sorted_fx_locs = 3*[None,]
        for iax in range(3):
            isorted = np.argsort(locfx[iax])
            self.sorted_fx[iax] = obinfx[iax][isorted]
            self.sorted_fx_locs[iax] = locfx[iax][isorted]

        _t1 = perf_counter()

        logger.log(_INFO1, f'Sorting fracture bins...done in {_t1-_t0:.2f} s.')

    def find_block_bounds(self, pt):
        """Find bounding box locations and bounding fracture indices"""

        retbnds = np.zeros(6, dtype=np.single)
        retbnds[::2] = self.domainOrigin
        retbnds[1::2] = retbnds[::2] + np.array(self.domainSize, dtype=np.single)

        retifx = -np.ones(6, dtype=np.int32)

        def _search(iax, pt, sortedifx):

            # do linear search of perpendicular fractures to find intercept
            for ifx in sortedifx:
                
                # fracture bounding box
                fxbb = self._fx[ifx].d

                # examine coordinates in direction
                iscandidate = True
                for iiax in range(3):
                    if iiax != iax:
                        inbb = (fxbb[2*iiax] <= pt[iiax] <= fxbb[2*iiax+1])
                        #breakpoint()
                        logger.debug(
                          f'Fx#{ifx:2d}\'s {"xyz"[iiax]} bounds: '
                          +f'{fxbb[2*iiax]:7.3f} <= '
                          +f'{pt[iiax]:7.3f} <= {fxbb[2*iiax+1]:7.3f}? '
                          + str(inbb))
                        iscandidate &= inbb

                if iscandidate:
                    logger.debug(f'Found intercept along {"xyz"[iax]} of '
                            + f'{pt} with {self._fx[ifx]}')
                    #breakpoint()
                    return fxbb[2*iax], ifx

            return None

        def _searchup(iax, pt, sortedifx):
            ret = _search(iax, pt, sortedifx)
            if ret is None:
                return float(self.domainOrigin[iax]+self.domainSize[iax]), -1,
            return ret

        def _searchdown(iax, pt, ifx, ileft):
            ret = _search(iax, pt, np.flip(ifx[:ileft]))
            if ret is None:
                return float(self.domainOrigin[iax]), -1,
            return ret

        for iax,vax in enumerate(pt):
            ileft = np.searchsorted(self.sorted_fx_locs[iax], vax)

            (_l, _ifx) = _searchdown(iax, pt, self.sorted_fx[iax], ileft)
            retbnds[2*iax  ]=_l
            retifx[2*iax  ]=_ifx

            (_l, _ifx) = _searchup(iax, pt, self.sorted_fx[iax][ileft:])
            retbnds[2*iax+1]=_l
            retifx[2*iax+1]=_ifx
            
        return (retbnds, retifx)

    def find_blocks_regular_grid(self, filter_keywords=[]):
        """Find blocks using a grid, centred at halfway points in the dfn's grid

        Check each block versus the given filters from `MatrixBlock.FILTERS`.
        """

        domo = np.array(self.domainOrigin, dtype=np.single)
        doms = np.array(self.domainSize, dtype=np.single)


        gl = [ np.fromiter(self.iterGridLines(ax), count=nax, dtype=np.single)
                for ax, nax in zip(range(3), self.getGridLineCounts()) ]
        sample_grid = 3*[None,]
        for i in range(3):
            if gl[i] is None:
                sample_grid[i] = np.zeros((1,))
            else:
                sample_grid[i] = (gl[i][1:] + gl[i][:-1])/2
        del gl

        blocks = {}
        for p in product(*sample_grid):
            logger.debug(f'Evaluating grid point {p}')
            bl = MatrixBlock(self, p)

            # skip loop iteration if any filters not met
            for fkw in filter_keywords:
                if not MatrixBlock.FILTERS[fkw](bl):
                    logger.debug(f'Filter {fkw} rejected block at {p}')
                    continue


            blbfx = tuple(bl.bfx)
            if blbfx not in blocks:
                blocks[blbfx] = bl
                logger.debug(f'Adding block at {p} with bounding fractures {blbfx}.')
            else:
                logger.debug(f'Rejecting duplicate block at {p}')

        npts = reduce(mul, map(attrgetter('size'), sample_grid))
        logger.info(f'Processed {npts} matrix blocks; using {len(blocks)}.')

        return list(blocks.values())

    def find_blocks_random(self, npts=5, filter_keywords=[]):
        """Return a list of "random" blocks

        Check each block versus the given filters from `MatrixBlock.FILTERS`.
        """

        domo = np.array(self.domainOrigin, dtype=np.single)
        doms = np.array(self.domainSize, dtype=np.single)

        blocks = []

        for i in range(npts):
            p = domo+doms*np.random.rand(domo.size)
            logger.debug(f'Chose point {p}')

            bl = MatrixBlock(self, p)

            # skip loop iteration if any filters not met
            for fkw in filter_keywords:
                if not MatrixBlock.FILTERS[fkw](bl):
                    logger.debug(f'Filter {fkw} rejected block at {p}')
                    continue

            blocks.append(bl)

        logger.info(f'Processed {npts} matrix blocks')
        return blocks


class MatrixBlock:
    """Representation of a matrix block bounded by fractures



    Attributes include:
    - `pt` Seed point of the volume (x,y,z)
    - `bb` Bounding box of the volume (x1, x2, ..., z1, z2)
    - 'bfx` List of bounding fracture indices
    - 'L' 

    On-demand attributes include

    """

    FILTERS = {
        '4sides':(lambda bl: len(bl.bfx) >=4),
        '5sides':(lambda bl: len(bl.bfx) >=5),
        '6sides':(lambda bl: len(bl.bfx) >=6),
    }

    __slots__ = ['pt', 'bb', 'bfx', 'L', 'Afrac',]

    def __init__(self, dfn, pt):
        """Analyze the dfn to find the block around `pt`"""

        self.pt = pt
        """[x y z]"""

        #breakpoint()
        (_bb, _bfx) = dfn.find_block_bounds(pt)
        #self.bb = _MatrixBlock._dummy_block_bounds(pt, 2.)
        self.bb = _bb
        """[x1 x2 y1 y2 z1 z2]"""

        _nbfx = sum(ifx > -1 for ifx in _bfx)
        self.bfx = sorted(np.fromiter(filter(lambda ifx: ifx>-1, _bfx), count=_nbfx,
                dtype=np.int32))
        """Sorted list of indicies of bounding fractures"""

        _length = _bb[1::2]-_bb[::2]
        self.L = _length
        self.Afrac = -1.

    def __getattr__(self, a):
        try:
            super().getattr(a)
        except AttributeError as e:
            if a == 'V':
                return np.prod(self.L)
            elif a == 'A':
                return 2*sum(np.prod(ll) for ll in combinations(self.L, 2))
            elif re.match(r'a[xyz][xyz]', a, re.I):
                i = 'xyz'.index(a[1].lower())
                j = 'xyz'.index(a[2].lower())
                return self.L[j]/self.L[i]
            raise e

    def iter_pts(self):
        _tmp = np.zeros(3)
        for x in self.bb[:2]:
            _tmp[0] = x
            for y in self.bb[2:4]:
                _tmp[1] = y
                for z in self.bb[4:]:
                    _tmp[2] = z
                    yield _tmp

    @staticmethod
    def pts_to_tecplot(bllist, fn):

        tecout = ''
        tecout += f'# {os.path.realpath(__file__)} on {datetime.datetime.now()}\n'
        tecout += 'VARIABLES="X","Y","Z"\n'
        tecout += '\n'

        tecout += f'ZONE T="Block initial points" '
        tecout += f'ZONETYPE=ORDERED I={len(bllist)} DATAPACKING=POINT'
        tecout += '\n'

        for bl in bllist:
            tecout += ' '.join(f'{v:8.3f}' for v in bl.pt) + '\n'

        with open(fn, 'w') as fout:
            print(tecout, file=fout)

        logger.info(f'Wrote sample points as Tecplot data in {fn}')

    @staticmethod
    def blocks_to_tecplot(bllist, fn):
        tecout = ''
        tecout += f'# {os.path.realpath(__file__)} on {datetime.datetime.now()}\n'
        tecout += 'VARIABLES="X","Y","Z","V","A","X:Y aspect","X:Z aspect","Frac SA"\n'
        tecout += '\n'

        ne = len(bllist)
        tecout += f'ZONE T="Block data" '
        tecout += f'ZONETYPE=FEBRICK '
        tecout += f'NODES={8*ne} ELEMENTS={ne} DATAPACKING=BLOCK '
        tecout += 'VARLOCATION=([4-8]=CELLCENTERED)'
        tecout += '\n'

        # X,Y,Z
        for i in range(3):
            tecout += f'# {"XYZ"[i]}\n'
            for bl in bllist:
                tecout += ' '.join(f'{v[i]:.3f}' for v in bl.iter_pts()) + '\n'
        # values
        for v in ['V', 'A', 'axy', 'axz', 'Afrac',]:
            tecout += '# '+v+'\n'
            tecout += '\n'.join(str(getattr(bl, v)) for bl in bllist) + '\n'

        # connectivity
        for ibl in range(len(bllist)):
            i = 8*ibl
            tecout += ' '.join(str(i+ii) for ii in [1,5,7,3,2,6,8,4])+'\n'


        with open(fn, 'w') as fout:
            print(tecout, file=fout)

        logger.info(f'Wrote blocks as Tecplot data in {fn}')

    @staticmethod
    def _dummy_block_bounds(pt, s):
        """Make a square block irrespecitve of fractures, for testing"""
        bb = np.zeros(6, dtype=np.single)
        bb[0] = pt[0]-s
        bb[1] = pt[0]+s
        bb[2] = pt[1]-s
        bb[3] = pt[1]+s
        bb[4] = pt[2]-s
        bb[5] = pt[2]+s
        return bb

    @staticmethod
    def get_stats(block_list):
        """Return select statistics seen in the histograms as a `dict`

        The `dict` is empty if there are no blocks
        """

        # early exit:
        if not block_list:
            return dict()

        ret = {}
        _nbl = len(block_list)

        ret['N'] = _nbl

        # Volume
        v = np.fromiter(map(attrgetter('V'), block_list), count=_nbl,
                dtype=np.single)
        ret['Volume geo.mean'] = scipy.stats.gmean(v)
        ret['Volume mean'] = np.mean(v)

        # Aspect
        #with warnings.simplefilter('ignore'):
        v = np.fromiter(map(attrgetter('axy'), block_list), count=_nbl,
            dtype=np.single)
        v = np.log10(v)
        ret['log-Aspect mean'] = np.mean(v)

        # x-Length
        # y-Length
        v = np.zeros((_nbl,3),np.single)
        for r in range(_nbl):
            v[r,:] = block_list[r].L

        for iax,ax in enumerate('xyz'):
            ret[ax+'-length geo.mean'] = scipy.stats.gmean(v[:,iax])
            ret[ax+'-length mean'] = np.mean(v[:,iax])
            ret[ax+'-length median'] = np.median(v[:,iax])

        return ret

    @staticmethod
    def to_histogram_pngs(block_list, filename_prefix):
        """Create predefined histogram images.

        Currently, histograms are
            1) block volume, with aritimetic and geometric means
            2) block aspect, with aritimetic mean
            3) block x-length, with arith. geo. and median
            4) block y-length, with arith. geo. and median
        """

        def _make_plot(v, name, xlab, pfx, **kwargs):
            fig, axes = plt.subplots(1)
            plt.title(f'Histogram of Matrix Block {name} ($N$={len(v)})')
            plt.xlabel(xlab)
            plt.ylabel(f'Frequency')

            _hist_kwargs = {}
            if 'range' in kwargs:
                _hist_kwargs['range']=kwargs['range']

            hist = plt.hist(v, rwidth=0.75, **_hist_kwargs)

            _mi,_ma = plt.xlim()
            if 'range' in kwargs:
                _mi, _ma =kwargs['range'][0],kwargs['range'][1]

            plt.xlim(left=floor(_mi), right=ceil(_ma))
            _ymi,_yma = plt.ylim()

            if 'vbars' in kwargs:
                lines = ["-","--","-.",":"]
                linecycler = cycle(lines)

                vlines = []
                for labl,val in kwargs['vbars']:
                    try:
                        ll = labl+f' = {val:.2f}'
                    except:
                        ll = labl

                    vlines.append(
                        plt.vlines(
                            x=val,
                            ymin=_ymi,
                            ymax=_yma,
                            label=ll,
                            color='black',
                            linestyles=next(linecycler)
                    ))

                plt.legend(handles=vlines)

            fn = filename_prefix+'_'+re.sub("\\W+","_",name)+'.png'
            plt.savefig(fn, dpi=300)
            logger.info(f'Saved {name} histogram as {fn}')
            plt.close()
        
        _nbl = len(block_list)

        # Volume
        v = np.fromiter(map(attrgetter('V'), block_list), count=_nbl,
                dtype=np.single)
        pct = np.percentile(v,[95.,])
        stats = scipy.stats.describe(v)
        _make_plot(v, 'Volume', 'Volume [$m^3$]', filename_prefix,
            vbars=[
                ('$\mu_{geo}$',scipy.stats.gmean(v),),
                ('$\mu$',stats.mean,),
            ],
            range=[0., ceil(pct[0])],
            )
        

        # Aspect
        for aspect in ['axy', 'axz', 'ayz',]:
            astr = f'{aspect[2]}:{aspect[1]}'
            v = np.fromiter(map(attrgetter(aspect), block_list), count=_nbl,
                    dtype=np.single)
            v = np.log10(v)
            pct = np.percentile(v, [5., 25., 50., 75., 95.,])
            max_less_extremes = ceil(np.max(np.abs(np.take(pct,[0,4]))))
            _make_plot(v, f'Aspect Ratio, {astr}', '$log_{10}$(Aspect) [-]',
                    filename_prefix,
                    vbars=[
                        ('$\mu$', np.mean(v),),
                        ('$2^{nd} Quartile$', pct[2],),
                        ('$1^{st}$ & $3^{rd}$ Quartile', np.take(pct, [1,3]),),
                    ],
                    range=[-max_less_extremes, max_less_extremes,]
                )

        # x,y,z-Lengths
        v = np.zeros((_nbl,3),np.single)
        for r in range(_nbl):
            v[r,:] = block_list[r].L
        for i,ax in enumerate('xyz'):
            if v[:,i].size < 2: continue
            pct = np.percentile(v[:,i],[25., 75., 95.,])
            _make_plot(v[:,i], f'{ax}-Length', f'{ax}-Length [$m$]', filename_prefix,
                vbars=[
                    ('$\mu_{geo}$',scipy.stats.gmean(v[:,i]),),
                    ('$\mu$',np.mean(v[:,i]),),
                    ('median',np.median(v[:,i]),),
                    ('$1^{st}$ & $3^{rd}$ Quartile', pct[:2],),
                ],
                range=[0.,ceil(pct[-1]),],)
