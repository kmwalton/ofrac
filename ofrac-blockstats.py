#!/usr/bin/env python
"""Identify matrix blocks in a given DFN; visualize blocks & statistics.

Specifically, produces:
     - Tecplot-readable datasets of blocks and seed points
     - png images of block lengths, aspect ratios,  and volumes
     - OFracGrid objects (as .pkl files) of the sampled DFN (if it was modified
       from the original by sub-domaining or nudging).

"""

import os
import argparse
import tempfile
import subprocess
import shutil

import numpy as np

from time import perf_counter
from itertools import chain

from .ofracs import numTuple2str as t2s
from .ofrac.matrixblock import MatrixBlockOFracGrid, MatrixBlock

import logging
logger = logging.getLogger(__file__)
#logger.setLevel(logging.DEBUG)
_INFO1 = logging.INFO+1

def process_pstats(dfn):

    pcalc = shutil.which('ofracstats-pcalc.py')

    if not pcalc:
        return
    
    with tempfile.NamedTemporaryFile(dir='.') as f:
        ofracs.OFracGrid.pickleTo(dfn, f)
        subprocess.run([pcalc, f.name,])

class BlockHow(argparse.Action):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.n=-1
        self.do_regular = False

    def __call__(self, parser, namespace, values, option_string):

        try:
            self.n = int(values)
            setattr(namespace, self.dest,
                    lambda dfn: dfn.find_blocks_random(self.n))
            return

        except ValueError as e:
            if isinstance(values, str) and values.lower()=='reg':
                self.do_regular = True
                setattr(namespace, self.dest,
                        lambda dfn: dfn.find_blocks_regular_grid())
                return

        parser.error(f'Bad value for {option_string}')

if __name__ == '__main__':

    argp = argparse.ArgumentParser()

    argp.add_argument('-n', '--nblocks',
        metavar='N|{"reg"}',
        help='''Number of matrix blocks in random sample, or "reg[ular]" for a
        regular grid. Default 10.''',
        nargs='?',
        dest='make_blocks',
        action=BlockHow,
        )
    argp.add_argument('-f', '--block-filter',
        choices=list(MatrixBlock.FILTERS.keys()),
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
        metavar=('x1','x2', 'y1', 'y2', 'z1', 'z2',),
        default=None,
        nargs=6,
        type=float,
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

    argp.add_argument('-v', '--verbose',
        default=0,
        action='count',
        help='Increase the verbosity. Default 0, maximum 4.',
        )

    argp.add_argument('FILE',
        help='Name of the orthogonal fracture network datafile.')

    args = argp.parse_args()

    if args.verbose:
        liblogger = logging.getLogger('ofrac.matrixblock')
        if args.verbose > 3:
            liblogger.setLevel(logging.DEBUG)
            logger.setLevel(logging.DEBUG)
        elif args.verbose > 1:
            liblogger.setLevel(logging.INFO-args.verbose+1)
            logger.setLevel(logging.INFO-args.verbose+1)
        elif args.verbose == 1:
            logger.setLevel(logging.INFO)
        liblogger.addHandler(logging.StreamHandler())
        logger.addHandler(logging.StreamHandler())


    if not args.prefix:
        args.prefix = os.path.splitext(os.path.basename(args.FILE))[0]

    if isinstance(args.random_seed, bool) and args.random_seed == True:
        pass # system time or something is used
    else:
        np.random.seed(int(args.random_seed))

    # process input dfn and modifications to it
    ofracs.__FX_COLLAPSE_POLICY__ = 'omit'
    dfn = ofracs.parse(args.FILE)
    logger.info(
      f'Read original domain {t2s(dfn.domainOrigin)}-{t2s(dfn.domainSize)}.')
    if args.sub_domain is not None:
        #try:
        #    bb = np.fromiter(args.sub_domain.split(), count=6, dtype=float)
        #except:
        #    argp.error('Error processing subdomain')
        bb = np.array(args.sub_domain)
        dfn.setDomainSize(bb[::2], bb[1::2])
        dfn.translate(-bb[::2])
        logger.info(f'Resized to {t2s(dfn.domainSize,"x","","")}.')

    if args.nudge:
        dfn.nudgeAll(args.nudge)

    # convert DFN to _SortedDFN for processing
    dfn = MatrixBlockOFracGrid(dfn)

    logger.log(_INFO1, 'Identifying blocks...')
    _t0 = perf_counter()

    # determine sample of matrix blocks
    blocks = args.make_blocks(dfn)

    _t1 = perf_counter()
    bb = list(map(float, chain.from_iterable(dfn.getBounds())))
    logger.info(
      f'Identified {len(blocks)} matrix blocks '
      +f'in subdomain {t2s(bb[::2])}-{t2s(bb[1::2])} '
      +f'in {_t1-_t0:.2f} s.')

    # apply filter
    if args.block_filter is not None:
        nprev = len(blocks)
        blocks = list(filter(MatrixBlock.FILTERS[args.block_filter], blocks))
        logger.info(
          f'Filter "{args.block_filter}" removed {nprev-len(blocks)} blocks;'
          +f' {len(blocks)} remain.')


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
    MatrixBlock.pts_to_tecplot(blocks, args.prefix+'_points.dat')
    MatrixBlock.blocks_to_tecplot(blocks, args.prefix+'_blocks.dat')

    MatrixBlock.to_histogram_pngs(blocks, args.prefix)

    if args.lay_to_jpg:
        try:
            import tecplot as tp
            tp.load_layout(args.lay_to_jpg)
            fn = args.prefix+'.jpg'
            tp.save_fig(fn)
            logger.info(f'Saved {args.lay_to_jpg} to {fn}.')
        except:
            logger.warning(f'Failed to use {args.lay_to_jpg}.')

