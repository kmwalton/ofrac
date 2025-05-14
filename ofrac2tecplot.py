#!/usr/bin/env python
"""Convert some orthogonal fracture network to tecplot format."""

import sys

import argparse

try:
    import ofrac.ofracs as ofracs
except ModuleNotFoundError:
    # accommodate "old style" PYTHONPATHing to within this module
    import ofracs

if __name__ == '__main__':

    argp = argparse.ArgumentParser()

    argp.add_argument('FILE_IN',
        help='input file name',
        )

    argp.add_argument( 'FILE_OUT',
        nargs='?',
        default='-',
        help="Filename for output (or stdout if blank or '-')",
        )

    args = argp.parse_args()

    g = ofracs.parse(args.FILE_IN)

    if not args.FILE_OUT or args.FILE_OUT == '-':
        g.printTecplot()
    else:
        with open(args.FILE_OUT,'w') as fout:
            g.printTecplot(fout=fout)
            
