#!/usr/bin/env python3
"""Groups fractures in bins based on orientation and aperture and reports
count of fractures per bin (frequency)

AUTHOR: Ken Walton, kmwalton@g360group.org.
Licenced under GNU GPLv3
Documentation intended to work with pdoc3.
"""

import argparse,sys,os,re,copy,traceback,glob,datetime
from itertools import count,accumulate
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

        self.datafns = []

        # process all input files to make an aggregate fracture network
        for fnin in files:
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
              self.datafns.append( os.path.basename(fnin) )
              self.grid = self.grid.merge(fxNet)

        #
        # do the statistics
        #

        # frequencies
        self.freq = (len(self.bins)+1)*[0,]
        for f in self.grid.iterFracs():
            self.freq[ bisect(self.bins, f.ap) ] += 1


        # descriptive statistics
        aps = np.fromiter(map( lambda f: f.ap, self.grid.iterFracs()), dtype=np.float_)
        (N,(apMin,apMax),mean,variance,skewness,kurtosis) = describe(aps)

        self.descStats = {
                'N':N,
                'Arithmetic mean':mean,
                'Geometric mean':gmean(aps),
                'Variance':variance,
                'Skewness':skewness }

        # cumulative density function
        self.cdf = list( v/N for v in accumulate(self.freq) )

    def printTecplot(self, fout):
        stdoutSave = sys.stdout
        sys.stdout = fout

        print(f'# {os.path.realpath(__file__)} on {datetime.datetime.now()}')
        print(f'VARIABLES="Bin [um]","Frequency","CDF"')
        print(f'''ZONE T="{','.join(self.datafns)}" I={len(self.bins)+1}''')

        # nice values for the descStats
        print(f'AUXDATA N="{self.descStats["N"]}"')
        print(f'AUXDATA ARITHMETRICMEAN="{self.descStats["Arithmetic mean"]*1e6:.0f}"')
        print(f'AUXDATA GEOMETRICMEAN="{self.descStats["Geometric mean"]*1e6:.0f}"')
        print(f'AUXDATA VARIANCE="{self.descStats["Variance"]:.4g}"')
        print(f'AUXDATA SKEWNESS="{self.descStats["Skewness"]:.4g}"')

        for i,f,c in zip(count(1), self.freq, self.cdf):
            print(f'{i} {f:10d} {c:10.5f}')

        conv = Decimal('1e6') # convert to microns
        binsMicrons = list(map( lambda v: v*conv, self.bins ) )
        foo='", "'.join(
                f'{a!s}-{b!s}' for (a,b) in
                    zip([0,]+binsMicrons[:-1], binsMicrons) )
        foo += f'", ">{binsMicrons[-1]}'
        print(f'CUSTOMLABELS "{foo}"')


        sys.stdout = stdoutSave

    def makeBinningReport(self):

        conv = Decimal('1e6') # convert to microns
        printBins = list(map( lambda v: v*conv, self.bins ) )

        width = 2+int(6+log10(float(self.bins[-1])))

        rpt = '{:>{w}} - {!s:>{w}}: {}\n'.format('0',printBins[0], self.freq[0],w=width)

        for i,bc in enumerate(self.freq[1:-1]):
            rpt += '{:>{w}} - {!s:>{w}}: {}\n'.format(
                f'>{printBins[i]!s}', printBins[i+1], bc, w=width)

        rpt += '{:>{w}} - {!s:>{w}}: {}\n'.format(
            f'>{printBins[-1]!s}', '', self.freq[-1], w=width)

        return rpt


    def makeDescStats(self):
        aps = np.fromiter(map( lambda f: f.ap, self.grid.iterFracs()), dtype=np.float_)
        (N,(apMin,apMax),mean,variance,skewness,kurtosis) = describe(aps)

        s = '\n'.join( f'{k:15} = {v!s}' for (k,v) in self.descStats.items() )
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

    parser.add_argument( '--tecplot-out',
            nargs='?',
            default=None,
            help="Filename (or stdout, default if blank or '-') for Tecplot-formatted output")

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

    if args.tecplot_out == None:
        pass
    elif args.tecplot_out == '' or args.tecplot_out == '-':
        b.printTecplot(sys.stdout)
    else:
        with open(args.tecplot_out,'w') as fout:
            b.printTecplot(fout)
