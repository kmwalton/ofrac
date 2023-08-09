#!/usr/bin/env python

import os
import datetime
import argparse
import tempfile
import subprocess
import shutil
import warnings
from re import sub
from math import floor,ceil
from operator import attrgetter, mul
from itertools import combinations, product, cycle
from functools import reduce

import numpy as np
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt

import ofracs

import logging
logger = logging.getLogger(__file__)
#logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def process_pstats(dfn):

    pcalc = shutil.which('ofracstats-pcalc.py')

    if not pcalc:
        return
    
    with tempfile.NamedTemporaryFile(dir='.') as f:
        ofracs.OFracGrid.pickleTo(dfn, f)
        subprocess.run([pcalc, f.name,])

class _SortedDFN(ofracs.OFracGrid):

    def __init__(self, ofracgrid_obj):
        self.__dict__.update(ofracgrid_obj.__dict__)

        nperpax = self.getFxCounts()
        obinfx = list(
                -np.ones(nperpax[i], dtype=int) for i in range(3))
        locfx = list(
                np.full(nperpax[i], np.inf, dtype=np.single)
                    for i in range(3))
        nextind = 3*[0,]

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
        logger.debug('Sorting bins...')

        self.sorted_fx = 3*[None,]
        self.sorted_fx_locs = 3*[None,]
        for iax in range(3):
            isorted = np.argsort(locfx[iax])
            self.sorted_fx[iax] = obinfx[iax][isorted]
            self.sorted_fx_locs[iax] = locfx[iax][isorted]

        logger.debug('Sorting bins...done.')

    def find_block_bounds(self, pt):
        """Find bounding box locations and bounding fracture indices"""

        retbnds = np.zeros(6, dtype=np.single)
        retbnds[::2] = dfn.domainOrigin
        retbnds[1::2] = retbnds[::2] + np.array(dfn.domainSize, dtype=np.single)

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

        


class _MatrixBlock:

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
        self.V = np.product(_length)
        self.A = 2*sum(np.product(a) for a in combinations(_length, 2))
        self.Afrac = 0.
        self.axy = _length[1]/_length[0]
        self.axz = _length[2]/_length[0]

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
        bb = np.zeros(6, dtype=np.single)
        bb[0] = pt[0]-s
        bb[1] = pt[0]+s
        bb[2] = pt[1]-s
        bb[3] = pt[1]+s
        bb[4] = pt[2]-s
        bb[5] = pt[2]+s
        return bb

def process_blocks_regular(dfn):
    """Find blocks using a grid, centred at halfway points in the dfn's grid"""

    domo = np.array(dfn.domainOrigin, dtype=np.single)
    doms = np.array(dfn.domainSize, dtype=np.single)


    gl = [ np.fromiter(dfn.iterGridLines(ax), count=nax, dtype=np.single)
            for ax, nax in zip(range(3), dfn.getGridLineCounts()) ]
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
        bl = _MatrixBlock(dfn, p)
        blbfx = tuple(bl.bfx)
        if blbfx not in blocks:
            blocks[blbfx] = bl
            logger.debug(f'Adding block at {p} with bounding fractures {blbfx}.')
        else:
            logger.debug(f'Rejecting block at {p}')

    npts = reduce(mul, map(attrgetter('size'), sample_grid))
    logger.info(f'Processed {npts} matrix blocks; using {len(blocks)}.')

    return list(blocks.values())

def process_blocks_random(dfn, npts=5):

    domo = np.array(dfn.domainOrigin, dtype=np.single)
    doms = np.array(dfn.domainSize, dtype=np.single)

    blocks = []

    for i in range(npts):
        p = domo+doms*np.random.rand(3)
        logger.debug(f'Chose point {p}')

        blocks.append(_MatrixBlock(dfn, p))

    logger.info(f'Processed {npts} matrix blocks')
    return blocks

def histograms_to_png(block_list, filename_prefix):

    def _make_plot(v, name, xlab, pfx, **kwargs):
        fig, axes = plt.subplots(1)
        plt.title(f'Histogram of Matrix Block {name}')
        plt.xlabel(xlab)
        plt.ylabel(f'Frequency')

        hist = plt.hist(v, rwidth=0.75)
        _mi,_ma = plt.xlim()
        plt.xlim(left=floor(_mi), right=ceil(_ma))

        if 'vbars' in kwargs:
            lines = ["-","--","-.",":"]
            linecycler = cycle(lines)

            vlines = []
            for labl,val in kwargs['vbars']:
                ll = labl+f' = {val:.2f}'
                vlines.append(
                    plt.axvline(
                        x=val,
                        label=ll,
                        color='black',
                        linestyle=next(linecycler)
                ))

            plt.legend(handles=vlines)

        fn = f'{filename_prefix}_{sub(" ","_",name)}.png'
        plt.savefig(fn)
        logger.info(f'Saved {name} histogram as {fn}')
        plt.clf()
    
    _nbl = len(block_list)

    # Volume
    v = np.fromiter(map(attrgetter('V'), block_list), count=_nbl,
            dtype=np.single)
    stats = scipy.stats.describe(v)
    _make_plot(v, 'Volume', 'Volume [$m^3$]', filename_prefix,
            vbars=[
            ('$\mu_{geo}$',scipy.stats.gmean(v),),
            ('$\mu$',stats.mean,),
            ])
    

    # Aspect
    #with warnings.simplefilter('ignore'):
    v = np.fromiter(map(attrgetter('axy'), block_list), count=_nbl,
        dtype=np.single)
    v = np.log10(v)
    stats = scipy.stats.describe(v)
    _make_plot(v, 'Aspect Ratio', '$log_{10}$(Aspect) [-]', filename_prefix,
            vbars=[('$\mu$',stats.mean,),])

    # x-Length
    # y-Length
    v = np.zeros((_nbl,3),np.single)
    for r in range(_nbl):
        v[r,:] = block_list[r].L
    _make_plot(v[:,0], 'x-Length', 'x-Length [$m$]', filename_prefix,
        vbars=[
            ('$\mu_{geo}$',scipy.stats.gmean(v[:,0]),),
            ('$\mu$',np.mean(v[:,0]),),
            ('median',np.median(v[:,0]),),
        ])
    _make_plot(v[:,1], 'y-Length', 'y-Length [$m$]', filename_prefix,
        vbars=[
            ('$\mu_{geo}$',scipy.stats.gmean(v[:,0]),),
            ('$\mu$',np.mean(v[:,1]),),
            ('median',np.median(v[:,0]),),
        ])


_FILTERS = {
    '4sides':(lambda bl: len(bl.bfx) >=4),

}
if __name__ == '__main__':

    argp = argparse.ArgumentParser()

    argp.add_argument('-n', '--nblocks',
        metavar='N|{"reg"}',
        help='''Number of matrix blocks in random sample, or "reg[ular]" for a
        regular grid. Default 10.''',
        default=10,
        )
    argp.add_argument('-f', '--block-filter',
        choices=list(_FILTERS.keys()),
        default=None,
        help='''Filter the blocks by the named criterion: "4sides" ensures that
        fractures bound the block on at least four sizes.
        ''',
        )
    argp.add_argument('--nudge',
        metavar='INCREMENT',
        help='Nudge fractures to increments of this value',
        )
    argp.add_argument('-d', '--sub-domain',
        metavar='x1 x2 y1 y2 z1 z2',
        help='Use the given sub-domain and shift its origin to (0,0,0)',
        )

    argp.add_argument('-p', '--prefix',
        metavar='path/to/prefix',
        help='The prefix of output files',
        )

    argp.add_argument('-l', '--lay-to-jpg',
        metavar='LAYOUT',
        default=None,
        help='Make a jpg using the tecplot layout.'
        )

    argp.add_argument('--random-seed',
        metavar='X',
        nargs='?',
        default=0,
        const=True,
        help='''Specify a seed (integer) for the random number generator. 
         Default seed value is 0 (for repeatability of sampling). If this
         option is used without a value "x", then numpy's default seeding
         procedure will be used.''',
        )


    argp.add_argument('FILE',
        help='Name of the orthogonal fracture network datafile.')

    args = argp.parse_args()

    if not args.prefix:
        args.prefix = os.path.splitext(os.path.basename(args.FILE))[0]

    if isinstance(args.random_seed, bool) and args.random_seed == True:
        pass # system time or something is used
    else:
        np.random.seed(int(args.random_seed))

    # process input dfn and modifications to it
    dfn = ofracs.parse(args.FILE)
    if args.sub_domain is not None:
        try:
            bb = np.fromiter(args.sub_domain.split(), count=6, dtype=float)
        except:
            argp.error('Error processing subdomain')
        dfn.setDomainSize(bb[::2], bb[1::2])
        dfn.translate(-bb[::2])

    if args.nudge:
        dfn.nudgeAll(args.nudge)

    # convert DFN to _SortedDFN for processing
    dfn = _SortedDFN(dfn)

    # argument fixup
    try:
        _tmp = int(args.nblocks)
    except ValueError:
        pass
    else:
        args.nblocks = _tmp

    # determine sample of matrix blocks
    blocks = None
    if isinstance(args.nblocks, int):
        blocks = process_blocks_random(dfn, args.nblocks)
    elif isinstance(args.nblocks, str) and \
            args.nblocks.lower()[:3]=='reg':
        blocks = process_blocks_regular(dfn)
    else:
        argp.error('Invalid value for --nblocks')

    # apply filter
    if args.block_filter is not None:
        nprev = len(blocks)
        blocks = list(filter(_FILTERS[args.block_filter], blocks))
        logger.info(
          f'Filter "{args.block_filter}" removed {nprev-len(blocks)} blocks.')


    # output to speed up next computation:
    # write subzone of fractures
    if args.FILE != args.prefix+'.pkl':
        with open(args.prefix+'_subfracs.pkl','wb') as f:
            ofracs.OFracGrid.pickleTo(dfn, f)
            logger.info(f'Wrote subdomain fractures to {f.name}.')
        with open(args.prefix+'_subfracs.dat','w') as f:
            dfn.printTecplot(f)
            logger.info(f'Wrote Tecplot format fractures to {f.name}.')

    # Useful output:

    # write points datafile
    #process_pstats(dfn)
    _MatrixBlock.pts_to_tecplot(blocks, args.prefix+'_points.dat')
    _MatrixBlock.blocks_to_tecplot(blocks, args.prefix+'_blocks.dat')

    histograms_to_png(blocks, args.prefix)

    if args.lay_to_jpg:
        try:
            import tecplot as tp
            tp.load_layout(args.lay_to_jpg)
            fn = args.prefix+'.jpg'
            tp.save_fig(fn)
            logger.info(f'Saved {args.lay_to_jpg} to {fn}.')
        except:
            logger.warning(f'Failed to use {args.lay_to_jpg}.')

