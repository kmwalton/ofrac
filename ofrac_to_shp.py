#!/python
"""Convert an OFracGrid DFN to real-world coordinates

Parameters for transforming the model coordinate-space DFN to real-world
coordinates are specific to the Hydrite site.

The output is a GeoJSON formatted file in WGS84, "epsg:4326", coordinate
projection (per GeoJSON spec).
"""

import sys
import argparse
import math
from io import StringIO

import numpy as np

import shapely
from shapely.affinity import rotate, translate
import geojson
from geojson import FeatureCollection, Feature, LineString
import geopandas as gpd

import ofracs


_mu = 1.003e-3
"""Viscosity of water, [kg m^-2 s^-1]"""

if __name__ == '__main__':

    argp = argparse.ArgumentParser()
    argp.add_argument('-t', '--transform',
        dest='t',
        nargs=3,
        type=float,
        metavar=('PHI +X +Y'.split()),
        default=(-0.79, 305800., 4770100.),
        help='''Rotation angle, X-translation and Y-translation.
        Defaults set to rotation of -0.79 radians, translation of 
        (+305800, +4770100) m.''',
        )
    argp.add_argument('-c', '--input-coordinate-system',
        dest='crs',
        default='epsg:26916',
        help='''The coordinate system of the transformation. Default
        "epsg:26916" for 'NAD 1983 UTM Zone 16N' in units of meters.
        '''
        )

    argp.add_argument('--geo-json',
        default=False,
        action='store_true',
        help='''Output GeoJSON format text of the data (converted to "epsg:4326"
            per GeoJSON specification. The file will be named
            'OUT_FILE_PFX.geojson'.
        ''',
    )
    argp.add_argument('DFN_FILE')
    argp.add_argument('OUT_FILE_PFX', 
        default='dfn',
        nargs='?',
        type=str,
        help='''Prefix for the output shapefile.''',
    )

    argv = argp.parse_args()

    # inputs
    dfn = ofracs.parse(argv.DFN_FILE)
    print(f'Loaded {argv.DFN_FILE}. Found {dfn.getFxCount()} fractures.')

    # convert to geometric objects and data
    #geo_dfn = FeatureCollection([])
    data = np.zeros(
        dfn.getFxCount(),
        dtype=[('aperture m', 'f4'), ('K_f m/s', 'f4'), ('Length m','f4'),],
        )
    geoms = dfn.getFxCount()*[None,]
    for i,f in enumerate(dfn.iterFracs()):

        c = [float(v) for v in f.d]
        #breakpoint()
        geom = shapely.LineString([c[0:4:2], c[1:4:2],])
        geom = rotate(geom, math.degrees(argv.t[0]), origin=(0,0))
        geom = translate(geom, *argv.t[1:])

        geoms[i] = geom
        data[i] = (float(f.ap), float(f.ap)**2/12./_mu,shapely.length(geom),)

        #Disused:
        '''props = {
            'ap m':float(f.ap),
            'K_f m/day':float(f.ap)**2/12./_mu*86400,
            'Length m'=shapely.length(geom),
        }

        feat = Feature(
            geometry=geom,
            properties=props,
        )

        geo_dfn.features.append(feat)
        '''

    # make the geo database
    geo_db = gpd.GeoDataFrame(data=data, geometry=geoms, crs=argv.crs.upper())
    '''
    geo_db = gpd.read_file(StringIO(geojson.dumps(geo_dfn)))
    geo_db = geo_db.set_crs(argv.crs, allow_override=True)
    '''

    # output as GeoJSON
    if argv.geo_json is not None:
        print(f'Outputting GeoJSON FeatureCollection with '
                f'{len(geo_dfn.features)} fractures.')
        with open(argv.OUT_FILE_PFX+'.geojson', 'w') as fout:
            print(geo_db.to_json(to_wgs84=True), file=fout)


    # output as shp
    shpfn = argv.OUT_FILE_PFX+'.shp'
    print(f'Outputting {shpfn}')
    geo_db.to_file(shpfn)
