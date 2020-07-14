#!/usr/bin/env python3
"""Groups fractures in bins based on orientation and aperture and reports
count of fractures per bin (frequency)

AUTHOR: Ken Walton, kmwalton@g360group.org.
Licenced under GNU GPLv3
Documentation intended to work with pdoc3.
"""

import argparse,sys,re,copy,traceback,glob
from decimal import Decimal
from bisect import bisect
from math import log10

import numpy as np
from scipy.stats import describe,gmean

import ofracs
from ofracs import OFracGrid,NotValidOFracGridError



__VERBOSITY__ = 0

# create a list of parser types
parserOptions = [ OFracGrid.PickleParser, ]

try:
    import parser_fractran
    parserOptions += list(parser_fractran.iterFractranParsers())
except ImportError as e:
    print("Warning: did not find 'parser_fractran'. Cannot parse FRACTRAN-type orthogonal fracture networks.", file=sys.stderr)

try:
    import parser_rfgen
    parserOptions += [ parser_rfgen.RFGenOutFileParser, ]
except ImportError as e:
    print("Warning: did not find 'parser_rfgen'. Cannot parse RFGen-type orthogonal fracture networks.", file=sys.stderr)

try:
    import parser_hgs_rfgeneco
    #parserOptions += list(parser_hgs_rfgeneco.??? )
except ImportError as e:
    print("Warning: did not find 'parser_hgs_rfgeneco'. Cannot parse HGS+RFGen-style orthogonal fracture networks.", file=sys.stderr)


__VERBOSITY__ = 0
"""Module level verbosity"""


class Binner:


    """
    bins - list of strings representing aperture values in units microns
    """
    def __init__(self, files, bins):
        toM = Decimal('1e-6')
        self.bins = list(map(lambda v:v*toM, sorted(map(Decimal,bins))))

        # store an empty OFracGrid
        self.grid = OFracGrid()


        # process all input files to make an aggregate fracture network
        for fnin in args.FILES:
           if __VERBOSITY__:
              print( "========= %s ========="%(fnin))

           fxNet = None

           errmsg = ''

           # try some different parsers
           for ParserClass in parserOptions:
              try:
                 parser = ParserClass(fnin)
                 fxNet = parser.getOFracGrid()
                   
              except BaseException as e:
                 errmsg += '\n'+ParserClass.__name__+' did not work- {}'.format(str(e))
                 fxNet = None

              except:
                  (t,v,tb) = sys.exc_info()
                  print( "Unexpected error: {}\n{}\n\nTraceback:".format(t,v), file=sys.stderr )
                  traceback.print_tb(tb)
                  sys.exit(-1)

              if fxNet:
                 break

           if not fxNet:
              raise NotValidOFracGridError('Could not parse input file "{}":\n{}\n'.format(fnin,errmsg))
           
           else:
              self.grid = self.grid.merge(fxNet)

    def makeBinningReport(self):

        conv = Decimal('1e6') # convert to microns

        bins = (len(self.bins)+1)*[0,]
        printBins = list(map( lambda v: v*conv, self.bins ) )

        for f in self.grid.iterFracs():
            bins[ bisect(self.bins, f.ap) ] += 1


        width = 2+int(6+log10(float(self.bins[-1])))

        rpt = '{:>{w}} - {!s:>{w}}: {}\n'.format('0',printBins[0], bins[0],w=width)

        for i,bc in enumerate(bins[1:-1]):
            #rpt = '{:w} - {:w} : {}'.format('>'+str(self.bins[-,self.bins[0], bins[0])
            rpt += '{:>{w}} - {!s:>{w}}: {}\n'.format(
                f'>{printBins[i]!s}', printBins[i+1], bc, w=width)

        rpt += '{:>{w}} - {!s:>{w}}: {}\n'.format(
            f'>{printBins[-1]!s}', '', bins[-1], w=width)

        return rpt


    def makeDescStats(self):
        aps = np.fromiter(map( lambda f: f.ap, self.grid.iterFracs()), dtype=np.float_)

        (N,(apMin,apMax),mean,variance,skewness,kurtosis) = describe(aps)

        s = ''
        s += f'N               = {N}\n'
        s += f'Arithmetic mean = {mean}\n'
        s += f'Geometric mean  = {gmean(aps)}\n'
        s += f'Variance        = {variance}\n'
        s += f'Skewness        = {skewness}\n'
        return s



    def __str__(self):
        return "bins = " + ",".join(str(v) for v in self.bins) + '\n' + str(self.grid)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument( '--bins',
            type=str,
            default="100",
            help="a comma or space separated list of apertures (in microns)")

    parser.add_argument( '-v', '--verbosity',
            default=0,
            action='count',
            help="Increase the verbosity of this operation with increasing '-vvv's or integer parameter value")

    parser.add_argument( 'FILES',
            nargs='+',
            help='fracture network input files (or fractran prefix)' )

    # TODO add sample/sub-sample zones, per ofracstats-pcalc

    args = parser.parse_args()
    __VERBOSITY__ = args.verbosity
    ofracs.__VERBOSITY__ = args.verbosity

    try:
        b = Binner(args.FILES, args.bins.replace(',',' ').split())
    except NotValidOFracGridError as e:
        print(str(e), file=sys.stderr)
        sys.exit(-1)

    print(b.makeBinningReport())
    print(b.makeDescStats())
