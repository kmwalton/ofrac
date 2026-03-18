"""
p_system.py - Fracture Network P-System Metrics and Serialization

This module defines the data structures for P-system fracture metrics
(P10, P20, P22, P30, P32, P33) and provides a seamless JSON interface
to save and load these objects.

## Example Downstream Usage

```python
import p_system

# Load the fully parsed data with a single function call
data = p_system.load_json('ofracstats_pcalc_output.json')

# Access the data using standard dictionary keys
zone_0_data = data['Zone0']
p10_x = zone_0_data['P10']['x']

# Use the objects directly (math attributes are preserved)
if p10_x.P10 > 2.5:
    print(f"High intensity detected! Count: {p10_x.f_count}")

# Printing the object automatically formats it
print(p10_x)  
# Output: P10-x :        12.34 /m   spacing-x:    0.08 m (count=  42, d= 3.4m)

print(data['Zone0']['P32']['All'])
# Output: P32:         2.45      (A= 24.5m^2 / V=10.0m^3)
```
"""

import json
from typing import NamedTuple
from math import log10, floor

# TODO: Replace this stub with your actual PERP dictionary or import it:
# from your_constants_file import PERP
PERP = {0: 'xy', 1: 'xz', 2: 'yz', 'x': 'yz', 'y': 'xz', 'z': 'xy'}

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
    d_perp: int
    f_count: int
    sp_area: float
    P20: float

    def __str__(self):
        if self.sp_area == 0.0:
            return ''
            
        _dir = PERP.get(self.d_perp, str(self.d_perp))
        return f"P20-{_dir:2s}: {self.P20:12.3f} /m^2 (count={self.f_count:6d}, A={self.sp_area:4.1f}m^2)"

class P22Result(NamedTuple):
    d_perp: int
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


# Private registry mapping JSON strings back to our Python classes
_REGISTRY = {
    'P10Result': P10Result,
    'P20Result': P20Result,
    'P22Result': P22Result,
    'P30Result': P30Result,
    'P32Result': P32Result,
    'P33Result': P33Result,
}

# ==========================================
# 2. THE ENCODER & DECODER
# ==========================================

class _ResultEncoder(json.JSONEncoder):
    """Custom encoder that converts NamedTuples to dicts and tags them."""
    def default(self, obj):
        obj_type = type(obj).__name__
        if obj_type in _REGISTRY:
            # Convert tuple to a clean dictionary
            d = obj._asdict()
            # Inject a hidden type marker
            d['__result_type__'] = obj_type
            return d
        return super().default(obj)

def _result_decoder(dct):
    """Hook that catches tagged dicts and rebuilds the NamedTuples."""
    if '__result_type__' in dct:
        cls_name = dct.pop('__result_type__')
        if cls_name in _REGISTRY:
            return _REGISTRY[cls_name](**dct)
    return dct

# ==========================================
# 3. THE PUBLIC API
# ==========================================

def save_json(data_dict, file_or_path, indent=4):
    """
    Saves a nested dictionary to a JSON format.
    Accepts a string path, pathlib.Path, or an already open file-like object.
    Automatically handles formatting of P-system Result objects.
    """
    # Duck typing: if it quacks like a file (has a write method), it is a file.
    if hasattr(file_or_path, 'write'):
        json.dump(data_dict, file_or_path, cls=_ResultEncoder, indent=indent)
    else:
        # Otherwise, treat it as a path string or pathlib.Path
        with open(file_or_path, 'w') as f:
            json.dump(data_dict, f, cls=_ResultEncoder, indent=indent)

def load_json(filepath):
    """
    Loads a JSON file and dynamically reconstructs all P-system Result objects.
    Returns the parsed dictionary.
    """
    with open(filepath, 'r') as f:
        return json.load(f, object_hook=_result_decoder)

