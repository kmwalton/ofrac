# ofrac
Orthogonal discrete FRACture network analysis tools

A collection of tools to manipulate and analyze orthogonal, discrete fracture networks that are used in various groundwater flow and transport numerical simulators.

## Installing

Requires Python 3.9 or newer.

```sh
pip install .          # from a clone of this repository
pip install -e .       # editable, for working on the tools themselves
```

Two groups of dependencies are optional, because they are only needed by one
tool each and are heavy or separately licensed:

```sh
pip install '.[shp]'      # shapely, geojson, geopandas -- needed by ofrac2shp
pip install '.[tecplot]'  # pytecplot -- optional extra output from ofracstats-block
```

### Reading formats other than pickles

Every parser except the pickled-network reader comes from `hgstools`, which is
not on PyPI and so cannot be installed by the command above. Without it the
tools still run, and read
networks that were pickled by `ofracs.OFracGrid`, but any other input fails
with a message naming the formats it cannot read:

```
Warning: did not find 'hgstools.pyhgs.parser.parser_rfgen'. Cannot parse
RFGen-type orthogonal fracture networks.
```

Install `hgstools` from source, and make sure `regex` is installed alongside
it, to read RFGen, FRACTRAN and HGS+RFGen networks.

## The tools

Installing puts nine commands on the PATH:

| command | what it does |
| --- | --- |
| `ofracstats-pcalc` | "P-system" fracture abundance measures (P10, P20, P32, ...) |
| `ofracstats-aperture` | fracture counts binned by orientation and aperture |
| `ofracstats-length` | fracture counts binned by length |
| `ofracstats-spacing` | P10 and raw fracture spacing data |
| `ofracstats-block` | matrix block identification, statistics and plots |
| `ofrac-percolation` | does the network connect the requested boundary faces? |
| `ofrac-rfgenwrapper` | run RFGen over seeds/inputs and pickle the results |
| `ofrac2shp` | convert a network to shapefile/GeoJSON in real-world coordinates |
| `ofrac2tecplot` | convert a network to Tecplot format |

Every command takes `--help`. `ofrac-percolation` also reports its answer
through the exit status (0 percolates, 1 does not), so it can be used in a
script with `--quiet`.

The command names are hyphenated; the modules behind them use underscores
(`ofrac.bin.ofracstats_pcalc`), because a hyphen is not legal in a Python
module name.

### Deprecated names

Four tools were renamed. The old file names remain in `ofrac/bin/` as shims
that print a warning and then run the real tool. They are scheduled for
removal after **2027-02-05**.

| old | new |
| --- | --- |
| `ofracstats-aperture.py` | `ofracstats_aperture.py` |
| `ofracstats-length.py` | `ofracstats_length.py` |
| `ofracstats-pcalc.py` | `ofracstats_pcalc.py` |
| `ofrac-blockstats.py` | `ofracstats_block.py` |

## Running without installing

The checkout and the Python package are two different directories with
similar names, and it matters which one you point at. Calling the checkout
`ofrac_repo` here to keep them apart:

```
ofrac_repo/            <- the checkout; put THIS on PYTHONPATH
├── pyproject.toml
├── ofrac/             <- the Python package; importable as `ofrac`
│   ├── ofracs.py
│   └── bin/
└── testing/
```

```sh
export PYTHONPATH=/path/to/ofrac_repo   # the directory holding pyproject.toml
python "$PYTHONPATH/ofrac/bin/ofracstats_pcalc.py" --help
```

So it is the checkout that must be importable -- not its parent, and not the
package directory inside it. The checkout may be named anything, `ofrac_repo`
or otherwise; that was not true before the package moved into `ofrac/`.

Running a script by its path puts that script's own directory on `sys.path`,
not your working directory, so `PYTHONPATH` has to be right even when you are
standing in the checkout.

One failure here is silent. If a second copy of the package is reachable on
`PYTHONPATH` in the older layout -- with its `__init__.py` at the top of the
checkout rather than one level in -- Python treats that copy as a regular
package and prefers it over `ofrac_repo`, which is only a namespace portion.
`import ofrac` then binds to the other copy and nothing reports an error. If
you keep a stable checkout beside a working one, put the one you mean to use
earlier on `PYTHONPATH`.

## Tests

```sh
python -m unittest discover -s testing -t .
```

from the repository root. `testing/` sits outside the package and is not
included in an install.

## Licence

GNU GPL v3 or later; see `LICENSE`.
