#!/usr/bin/env python
"""Groups fractures in bins based on lengths and reports
count of fractures per bin (frequency)

AUTHOR: Ken Walton, kmwalton@g360group.org.
Licenced under GNU GPLv3
Documentation intended to work with pdoc3.
"""

import sys
import os
import argparse
import datetime
import warnings
from itertools import count
from operator import itemgetter

import numpy as np
import scipy.stats
from tabulate import tabulate

from decimal import Decimal

try:
    from ofrac import ofracs
    from ofrac.ofracs import OFracGrid,NotValidOFracGridError
except ImportError:
    # 'ofrac' can also resolve to the sub-package beside this script, which
    # holds no 'ofracs'; that is an ImportError rather than a missing module
    # accommodate "old style" PYTHONPATHing to within this module
    import ofracs
    from ofracs import OFracGrid,NotValidOFracGridError

__VERBOSITY__ = 0

class OFracBinner():

    def __init__(self, var, binLabels):
        self.binVar = var
        self.binLabels = sorted(list( Decimal(b) for b in binLabels ))

    def strTecplotHeader(self):
        """Return a string for a Tecplot ASCII file header"""
        s = ''
        s += f'# {os.path.realpath(__file__)} on {datetime.datetime.now()}\n'
        s += f'VARIABLES="{self.binVar}","Frequency","CDF","Frequency (%)"\n'
        s += """#Notes:
# Frequency (%) is the normalized frequency, i.e. the count of each bin divided
# by the total count of sampled values.
# The AUXDATA FREQ_MAX in each zone contains the maximum value frequency (count)
# value of the bins.
"""
        return s

    def get_bin_bounds(self):
        """Return list of string representations of bins.

        There is one more bin than there are divisions: the first holds the
        values below the first division, which would otherwise go unreported.
        """
        b = self.binLabels
        foo=[ f'0-{b[0]!s}', ] \
            + [ f'{a!s}-{b!s}' for (a,b) in zip(b[:-1], b[1:]) ] \
            + [ f'>{b[-1]}', ]
        return foo

    def strTecplotFooter(self):
        """Return the CUSTOMLABELS string, etc, as tecplot footer info"""
        foo = '", "'.join(b for b in self.get_bin_bounds())
        return f'CUSTOMLABELS "{foo}"'

    def strTecplotZone(self):
        raise NotImplementedError('Subclass implementation?')

    def printTecplot(self, fout):
        stdoutSave = sys.stdout
        sys.stdout = fout

        print(self.strTecplotHeader())
        print(self.strTecplotZone())
        print(self.strTecplotFooter())

        sys.stdout = stdoutSave

class LengthBinner(OFracBinner):

    def __init__(self, files, bins):

        super().__init__('Length [m]', bins,)

        self.bins = [ float(v) for v in bins ]

        # merge every input file into one network, in one pass
        self.grid = OFracGrid().merge( *map(ofracs.parse, files) )

        # a fracture is a length along each of the two axes in its plane, and
        # nothing along the axis it is perpendicular to
        inPlane = self.grid.getFxPlaneAxes()
        lengths = self.grid.getFxLengths()

        # bin on the stored coordinates, so that a length equal to a bin's edge
        # falls in the bin above it
        lengthBins = self.grid.getFxLengthBins(self.bins)

        self.histo = {}
        self.auxd = dict( (c,[]) for c in 'xyz' )
        for i,c in enumerate('xyz'):
            length_data = lengths[inPlane[:, i], i]

            # bin 0 holds the lengths below the first division; it is reported
            # like any other, so that every length counted in N appears
            self.histo[c] = np.bincount(
                    lengthBins[inPlane[:, i], i],
                    minlength=len(self.bins)+1)

            if len(length_data) == 0:
                self.auxd[c].extend( [
                    ('N',0,''),
                    ('ARITHMETICMEAN',f'-',''),
                    ('GEOMEAN',f'-',''),
                    ('MIN_MAX',f'-..-',''),
                ] )
                continue

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",
                        category=RuntimeWarning, message="Precision loss occurred")
                stats = scipy.stats.describe(length_data)
            gmean = scipy.stats.gmean(length_data)

            self.auxd[c].extend( [
                ('N',stats.nobs,''),
                ('ARITHMETICMEAN',f'{stats.mean:.2f}',''),
                ('GEOMEAN',f'{gmean:.2f}',''),
                ('MIN_MAX',f'{stats.minmax[0]:.1f}..{stats.minmax[1]:.1f}',''),
            ] )


    def _totable(self, histo):
        n = max(1, np.sum(histo))
        normf = histo/n
        return list(zip(count(1), self.get_bin_bounds(), histo, normf,
                    np.cumsum(normf)))

    def __str__(self):
        s = ''
        for i,c in enumerate('xyz'):
            s += f'''{c}-lengths data:\n'''

            for k,v,comm in self.auxd[c]:
                s += f'{k}{c}={v}\n'

            s += tabulate(self._totable(self.histo[c]),
                headers=['BinID', 'Bin', 'Frequency', 'Norm.Frequency', 'CDF'],
                ) + '\n'

        return s


    def strTecplotZone(self):
        s = ''

        for i,c in enumerate('xyz'):
            s += f'''ZONE T="{c}-length" I={len(self.bins)+1}\n'''

            for k,v,comm in self.auxd[c]:
                if comm:
                    s += f'# {comm}:\n'
                s += f'AUXDATA {k}="{v}"\n'

            s += tabulate(map(itemgetter(0,2,3,4), self._totable(self.histo[c])),
                    tablefmt='plain') + '\n'

        return s


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument( '--bins',
            type=str,
            default="100",
            help="a comma or space separated list of length bin divisions")

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
        b = LengthBinner(args.FILES, args.bins.replace(',',' ').split())
    except NotValidOFracGridError as e:
        print(str(e), file=sys.stderr)
        sys.exit(-1)


    if args.tecplot_out == None:
        pass
    elif args.tecplot_out == '' or args.tecplot_out == '-':
        b.printTecplot(sys.stdout)
    else:
        with open(args.tecplot_out,'w') as fout:
            b.printTecplot(fout)

    print(b)

    exit(0)
