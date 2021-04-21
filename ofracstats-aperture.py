#!/usr/bin/env python
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
    """Produce statistics of fracture apertures.
    """


    def __init__(self, files, bins):
    """
        Arguments:
            files : list-like
                A list of file names of ofracs-parsable DFNs. These will be
                merged.
            bins : list-like
                A list of strings representing aperture values in units microns
    """
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
            'Skewness':skewness,
            'Max. Frequency':max(self.freq),
        }

        # cumulative density function
        self.cdf = list( v/N for v in accumulate(self.freq) )

    def strTecplotHeader(self):
        """Return a string for a Tecplot ASCII file header"""
        s = ''
        s += f'# {os.path.realpath(__file__)} on {datetime.datetime.now()}\n'
        s += f'VARIABLES="Bin [um]","Frequency","CDF","Frequency (%)"\n'
        s += """#Notes:
# Frequency (%) is the normalized frequency, i.e. the count of each bin divided
# by the total count of fractures.
# The AUXDATA FREQ_MAX in each zone contains the maximum value frequency (count)
# value of the bins.
"""

        return s

    def strTecplotZone(self):
        """Return a string for ASCII Tecplot zone data"""
        s = ''

        # zone header
        s += f'''ZONE T="{','.join(self.datafns)}" I={len(self.bins)+1}\n'''

        # aux data
        # nice values for the descStats
        s += f'AUXDATA N="{self.descStats["N"]}"\n'
        s += f'AUXDATA ARITHMETICMEAN="{self.descStats["Arithmetic mean"]*1e6:.0f}"\n'
        s += f'AUXDATA GEOMETRICMEAN="{self.descStats["Geometric mean"]*1e6:.0f}"\n'
        s += f'AUXDATA VARIANCE="{self.descStats["Variance"]:.4g}"\n'
        s += f'AUXDATA SKEWNESS="{self.descStats["Skewness"]:.4g}"\n'
        s += f'AUXDATA FREQ_MAX="{self.descStats["Max. Frequency"]:.4g}"\n'
        s += f'AUXDATA REGION="{self.grid.strDomFromTo()}"\n'
        s += f'''AUXDATA DATAFILES="{','.join(self.datafns)}"\n'''

        # zone data
        n = self.descStats["N"]
        for i,f,c in zip(count(1), self.freq, self.cdf):
            normf = f/n
            s += f'{i:<2d} {f:10d} {c:10.6f} {normf:10.6f}\n'

        return s

    def strTecplotFooter(self):
        """Return the CUSTOMLABELS string, etc, as tecplot footer info"""

        conv = Decimal('1e6') # convert to microns
        binsMicrons = list(map( lambda v: v*conv, self.bins ) )
        foo='", "'.join(
                f'{a!s}-{b!s}' for (a,b) in
                    zip([0,]+binsMicrons[:-1], binsMicrons) )
        foo += f'", ">{binsMicrons[-1]}'

        return f'CUSTOMLABELS "{foo}"'

    def printTecplot(self, fout):
        stdoutSave = sys.stdout
        sys.stdout = fout

        print(self.strTecplotHeader())
        print(self.strTecplotZone())
        print(self.strTecplotFooter())

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
