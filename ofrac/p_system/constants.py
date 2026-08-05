# direction string to direction index mapping
DIR  = { "x":0, "y":1, "z":2 }
"""Mapping of an axis (string) to its index (int)."""

# RFGen orientation index to orientation string mapping
INDO = { 1:"xy", 2:"xz", 3:"yz" }
"""Mapping of RFGen's orientation index to a coordinate plane."""

# orientation string to RFGen orientation index mapping
OIND = { "xy":1, "xz":2, "yz":3 , \
         "yx":1, "zx":2, "zy":3 }
"""Mapping of a coordinate plane (orientation) to RFGen's index."""

PERP = {2: 'xy', 1: 'xz', 0: 'yz', 'x': 'yz', 'y': 'xz', 'z': 'xy'}
"""Mapping of an axis to the two perpendicular axes"""
