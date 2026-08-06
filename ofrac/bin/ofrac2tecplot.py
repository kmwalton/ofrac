#!/usr/bin/env python
"""Convert some orthogonal fracture network to tecplot format."""

import sys

import argparse

import ofrac.ofracs as ofracs

def main(argv=None):

    argp = argparse.ArgumentParser()

    argp.add_argument('FILE_IN',
        help='input file name',
        )

    argp.add_argument( 'FILE_OUT',
        nargs='?',
        default='-',
        help="Filename for output (or stdout if blank or '-')",
        )

    args = argp.parse_args(argv)

    g = ofracs.parse(args.FILE_IN)

    if not args.FILE_OUT or args.FILE_OUT == '-':
        g.printTecplot()
    else:
        with open(args.FILE_OUT,'w') as fout:
            g.printTecplot(fout=fout)

    return 0


if __name__ == '__main__':
    sys.exit(main())

