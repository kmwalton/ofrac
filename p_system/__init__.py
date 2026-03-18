"""
p_system.py - Fracture Network P-System Metrics and Serialization

This module defines the data structures for P-system fracture metrics
(P10, P20, P22, P30, P32, P33) and provides a seamless JSON interface
to save and load these objects.

## Example Downstream Usage

```python
import ofrac.p_system
from ofrac.p_system.constants import PERP, DIR

# Load the fully parsed data with a single function call
data = p_system.load_json('fracture_output.json')

# Access the flat dictionary using the metric-direction keys
zone_0_data = data['Zone0']
p10_x = zone_0_data['P10-x']

# Use the objects directly (math attributes are preserved)
if p10_x.P10 > 2.5:
    print(f"High intensity detected! Count: {p10_x.f_count}")

# Printing the object automatically formats it
print(p10_x)  
# Output: P10-x :        12.34 /m   spacing-x:    0.08 m (count=  42, d= 3.4m)

print(data['Zone0']['P32'])
# Output: P32:         2.45      (A= 24.5m^2 / V=10.0m^3)
```
"""

import json
from typing import NamedTuple
from math import log10, floor

from .constants import PERP

# ==========================================
# 1. THE DATA MODELS
# ==========================================

class P10Result(NamedTuple):
    d_scan: str
    '''Direction of the scan line'''
    size_1: float
    '''Length of the zone in the direction of the scan line'''
    f_count: int
    '''Count of fractures intersected by scan lines'''
    length_scan: float
    '''Cumulative length of scan lines'''
    P10: float
    '''P10 value'''

    def __str__(self):
        if self.size_1 < 1e-6:
            return ''

        try:
            mag = int(floor(log10(self.P10)))
        except ValueError:
            mag = -1

        _p10 = f'{round(self.P10, -mag+1):.{-mag+2}f}'

        p01 = float('inf')
        if self.f_count > 0:
            p01 = self.length_scan / float(self.f_count)
            
        _p01 = f'{round(p01, mag+3):.{max(0,mag+2)}f}'

        return (f"P10-{self.d_scan} : {_p10: >12} /m   "
                f"spacing-{self.d_scan}: {_p01: >7} m "
                f"(count={self.f_count:4d}, d={self.length_scan:4.1f}m)")

class P20Result(NamedTuple):
    d_perp: int
    '''Direction perpendicular to the scan plane'''
    f_count: int
    '''Count of fractures intersected by scan planes'''
    sp_area: float
    '''Cumulative area of the scan planes'''
    P20: float
    '''P20 value'''

    def __str__(self):
        if self.sp_area == 0.0:
            return ''
            
        _dir = PERP.get(self.d_perp, str(self.d_perp))
        return f"P20-{_dir:2s}: {self.P20:12.3f} /m^2 (count={self.f_count:6d}, A={self.sp_area:4.1f}m^2)"

class P22Result(NamedTuple):
    d_perp: int
    '''Direction perpendicular to the scan plane'''
    size_1: float
    '''Length of the zone along axis 1 of the scan plane'''
    size_2: float
    '''Length of the zone along axis 2 of the scan plane'''
    f_count: int
    '''Count of fractures intersected by scan planes'''
    sp_area: float
    '''Cumulative area of the scan planes'''
    P22: float
    '''P22 value'''

    def __str__(self):
        if self.size_1 == 0.0 or self.size_2 == 0.0:
            return ''
            
        _dir = PERP.get(self.d_perp, str(self.d_perp))
        return f"P22-{_dir:2s}: {self.P22:12.6f}      (count={self.f_count:6d}, A={self.sp_area:4.1f}m^2)"

class P30Result(NamedTuple):
    f_count: int
    '''Count of fractures intersecting the zone'''
    zn_vol: float
    '''Volume of zone'''
    P30: float
    '''P30 value'''

    def __str__(self):
        if self.zn_vol <= 1.0e-6:
            return ''
            
        return f"P30:    {self.P30:12.3f} /m^3 (count={self.frac_count:6d}, V={self.zn_vol:4.1f}m^3)"

class P32Result(NamedTuple):
    zn_vol: float
    '''Volume of zone'''
    fx_area: float
    '''Cumulative area of fractures intersecting the zone'''
    P32: float
    '''P32 value'''

    def __str__(self):
        if self.zn_vol <= 1.0e-6:
            return ''
            
        return f"P32:    {self.P32:12.5g}      (A={self.fx_area:.5g}m^2 / V={self.zn_vol:4.1f}m^3)"

class P33Result(NamedTuple):
    zn_vol: float
    '''Volume of zone'''
    fx_vol: float
    '''Cumulative volume of fractures intersecting the zone'''
    P33: float
    '''P33 value'''

    def __str__(self):
        if self.zn_vol <= 1.0e-6:
            return ''
            
        return f"P33:    {self.P33:12.5g}      (V_frac={self.fx_vol:.5g}m^3 / V={self.zn_vol:4.1f}m^3)"


# Private registry mapping JSON strings back to our Python classes
_REGISTRY = {
    'P10Result': P10Result,
    'P20Result': P20Result,
    'P22Result': P22Result,
    'P30Result': P30Result,
    'P32Result': P32Result,
    'P33Result': P33Result,
}
import json
from typing import NamedTuple
from math import log10, floor

# Relative import pulling PERP from your new constants.py file in the same folder
from .constants import PERP 

# ==========================================
# 1. THE DATA MODELS
# ==========================================

class P10Result(NamedTuple):
    d_scan: str
    size_1: float
    f_count: int
    length_scan: float
    P10: float

    def __str__(self):
        if self.size_1 < 1e-6:
            return ''
        try:
            mag = int(floor(log10(self.P10)))
        except ValueError:
            mag = -1
        _p10 = f'{round(self.P10, -mag+1):.{-mag+2}f}'
        p01 = float('inf')
        if self.f_count > 0:
            p01 = self.length_scan / float(self.f_count)
        _p01 = f'{round(p01, mag+3):.{max(0,mag+2)}f}'
        return (f"P10-{self.d_scan} : {_p10: >12} /m   "
                f"spacing-{self.d_scan}: {_p01: >7} m "
                f"(count={self.f_count:4d}, d={self.length_scan:4.1f}m)")

class P20Result(NamedTuple):
    d_perp: str
    f_count: int
    sp_area: float
    P20: float

    def __str__(self):
        if self.sp_area == 0.0:
            return ''
        _dir = PERP.get(self.d_perp, str(self.d_perp))
        return f"P20-{_dir:2s}: {self.P20:12.3f} /m^2 (count={self.f_count:6d}, A={self.sp_area:4.1f}m^2)"

class P22Result(NamedTuple):
    d_perp: str
    size_1: float
    size_2: float
    f_count: int
    sp_area: float
    P22: float

    def __str__(self):
        if self.size_1 == 0.0 or self.size_2 == 0.0:
            return ''
        _dir = PERP.get(self.d_perp, str(self.d_perp))
        return f"P22-{_dir:2s}: {self.P22:12.6f}      (count={self.f_count:6d}, A={self.sp_area:4.1f}m^2)"

class P30Result(NamedTuple):
    frac_count: int
    zn_vol: float
    P30: float

    def __str__(self):
        if self.zn_vol <= 1.0e-6:
            return ''
        return f"P30:    {self.P30:12.3f} /m^3 (count={self.frac_count:6d}, V={self.zn_vol:4.1f}m^3)"

class P32Result(NamedTuple):
    zn_vol: float
    fx_area: float
    P32: float

    def __str__(self):
        if self.zn_vol <= 1.0e-6:
            return ''
        return f"P32:    {self.P32:12.5g}      (A={self.fx_area:.5g}m^2 / V={self.zn_vol:4.1f}m^3)"

class P33Result(NamedTuple):
    zn_vol: float
    fx_vol: float
    P33: float

    def __str__(self):
        if self.zn_vol <= 1.0e-6:
            return ''
        return f"P33:    {self.P33:12.5g}      (V_frac={self.fx_vol:.5g}m^3 / V={self.zn_vol:4.1f}m^3)"

# Private registry mapping the prefix of your keys back to our Python classes
_REGISTRY = {
    'P10': P10Result,
    'P20': P20Result,
    'P22': P22Result,
    'P30': P30Result,
    'P32': P32Result,
    'P33': P33Result,
}

# ==========================================
# 2. THE DECODER
# ==========================================

def _result_decoder(dct):
    """
    Hook that catches dictionaries as they are loaded.
    It looks at the keys (e.g., 'P10-x' or 'P30') to figure out which 
    NamedTuple to rebuild, and unpacks the list directly into it.
    """
    for key, value in dct.items():
        if isinstance(value, list):
            metric_prefix = key.split('-')[0]
            if metric_prefix in _REGISTRY:
                dct[key] = _REGISTRY[metric_prefix](*value)
    return dct

# ==========================================
# 3. THE PUBLIC API
# ==========================================

def save_json(data_dict, file_or_path, indent=4):
    """
    Saves a nested dictionary to a JSON format using standard list serialization.
    Accepts a string path, pathlib.Path, or an already open file-like object.
    """
    if hasattr(file_or_path, 'write'):
        json.dump(data_dict, file_or_path, indent=indent)
    else:
        with open(file_or_path, 'w') as f:
            json.dump(data_dict, f, indent=indent)

def load_json(file_or_path):
    """
    Loads JSON data and dynamically reconstructs all P-system Result objects
    from their list representations.
    """
    if hasattr(file_or_path, 'read'):
        return json.load(file_or_path, object_hook=_result_decoder)
    else:
        with open(file_or_path, 'r') as f:
            return json.load(f, object_hook=_result_decoder)
