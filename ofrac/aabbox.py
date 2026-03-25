"""Axis-Aligned Bounding Box"""

import numpy as np

class AABBox:
    """Axis-Aligned Bounding Box"""
    def __init__(self, x0, y0, z0, x1, y1, z1):
        self._data = np.array([x0, y0, z0, x1, y1, z1], dtype=np.float64)

    @classmethod
    def from_blockspec(cls, blocspec):
        """
        Parses a block specification string formatted as 'x0 x1 y0 y1 z0 z1' 
        and returns a BBox instance.
        """
        # 1. Parse the string into a list of 6 floats
        # x0, x1, y0, y1, z0, z1
        vals = list(map(float, blocspec.split()))
        
        if len(vals) != 6:
            raise ValueError(f"BBox string must contain exactly 6 values, got {len(vals)}")

        # 2. Reorder for the standard __init__: (x0, y0, z0, x1, y1, z1)
        # We take indices 0, 2, 4 (mins) then 1, 3, 5 (maxs)
        reordered = [vals[0], vals[2], vals[4], vals[1], vals[3], vals[5]]
        
        return cls(*reordered)    # Allow direct attribute access

    def to_blockspec(self, precision=3):
        """
        Returns the BBox as a string in 'x0 x1 y0 y1 z0 z1' format.
        """
        # Reshape to (2,3) -> [[x0, y0, z0], [x1, y1, z1]]
        # Transpose (.T) -> [[x0, x1], [y0, y1], [z0, z1]]
        # Ravel -> [x0, x1, y0, y1, z0, z1]
        interleaved = self._data.reshape((2, 3)).T.ravel()
        
        fmt = f".{precision}f"
        return " ".join(f"{v:{fmt}}" for v in interleaved)

    @property
    def x0(self): return self._data[0]
    @property
    def y0(self): return self._data[1]
    @property
    def z0(self): return self._data[2]
    @property
    def x1(self): return self._data[3]
    @property
    def y1(self): return self._data[4]
    @property
    def z1(self): return self._data[5]

    # --- THE MAGIC METHODS ---

    def __setitem__(self, key, value):
        """Allows box[:3] = [1, 2, 3] or box[0] = 5.0"""
        self._data[key] = value

    def __getitem__(self, item):
        """Allows box[0], box[1:3], etc."""
        return self._data[item]

    def __len__(self):
        """Allows len(box) -> 6"""
        return len(self._data)

    def __array__(self, dtype=None, copy=None):
        """Allows np.add(box, 1) or np.array(box)"""
        if dtype:
            return self._data.astype(dtype, copy=copy)
        return self._data

    def __repr__(self):
        v = self._data
        #return f"AABBox(x0={v[0]:.3f}, y0={v[1]:.3f}, z0={v[2]:.3f}, x1={v[3]:.3f}, y1={v[4]:.3f}, z1={v[5]:.3f})"
        return f'AABBox({", ".join("{}{}={:.3f}".format(*_) for _ in zip("xyzxyz","000111",v))})'

    def __str__(self):
        v = self._data
        fr = ', '.join("{:.3f}".format(_) for _ in v[:3])
        to = ', '.join("{:.3f}".format(_) for _ in v[3:])
        return f'({fr})->({to})'

    def shrink(self, v):
        """Returns a new AABBox that is smaller by v on all sides"""
        if any((self._data[3:]-self._data[:3])<=v):
            raise ValueError(f'Cannot shrink {self} by {v} -- gets too small')
        return AABBox(*(self._data[:3]+v), *(self._data[3:]-v))


    def find_inner_grid_indices(self, x_grids, y_grids, z_grids):
        """Returns (xs, xe, ys, ye, zs, ze) for gridlines inside the box."""
        grids = [np.asanyarray(g) for g in (x_grids, y_grids, z_grids)]
        mins = [self.x0, self.y0, self.z0]
        maxs = [self.x1, self.y1, self.z1]

        indices = []
        for i in range(3):
            start_idx = np.searchsorted(grids[i], mins[i], side='left')
            end_idx = np.searchsorted(grids[i], maxs[i], side='right') - 1
            indices.extend([start_idx, end_idx])
        return tuple(indices)

    def to_centroid_bbox(self, x_grids, y_grids, z_grids):
        """Returns a new AABBox bounded by the first and last inner cell centroids."""

        grids = [np.asanyarray(g) for g in (x_grids, y_grids, z_grids)]
        #cgrids = [(g[1:]+g[:-1])/2. for g in (x_grids, y_grids, z_grids)]
        idx = self.find_inner_grid_indices(*grids)

        c = []
        for i in range(3):
            s, e = idx[i*2], idx[i*2+1]
            # Safety check: need at least two gridlines to form one cell
            if s >= e:
                raise ValueError(f"AABBox too small to contain a cell in axis {i}")

            # x0' = (G[s] + G[s+1])/2 | x1' = (G[e-1] + G[e])/2
            c.append((grids[i][s] + grids[i][s+1]) / 2.0)
            c.append((grids[i][e-1] + grids[i][e]) / 2.0)

        # Unpack directly into new instance: x0', y0', z0', x1', y1', z1'
        return AABBox(c[0], c[2], c[4], c[1], c[3], c[5])


    def reshape(self, shape):
        """Pass through to numpy array to support the user's string formatting."""
        return self._data.reshape(shape)

    def iter_face_bbox(self, dim_mask=(True,True,True)):
        """
        Yields pairs of AABBox objects representing the planes
        on the opposite faces of the box for each active axis.
        """
        # Get indices of active axes (e.g., [0, 1, 2])
        active_axes = np.arange(3)[dim_mask]

        for ax in active_axes:
            # Create copies of current box data
            d0 = self._data.copy()
            d1 = self._data.copy()

            # For the current axis, collapse the thickness to zero to create a plane
            # Face 0: Set 'max' slot to match 'min' slot
            d0[ax + 3] = d0[ax]

            # Face 1: Set 'min' slot to match 'max' slot
            d1[ax] = d1[ax + 3]

            yield AABBox(*d0), AABBox(*d1)

    def iter_layer_bbox(self, grids, dim_mask=(True,True,True)):
        """
        Yields pairs of AABBox objects representing the grid layers
        on the opposite faces of the box for each active axis.
        """
        # Get indices of active axes (e.g., [0, 1, 2])
        active_axes = np.arange(3)[dim_mask]

        for ax in active_axes:
            # Create copies of current box data
            d0 = self._data.copy()
            d1 = self._data.copy()

            # Face 0: find the outside gridline and go one inwards
            i = np.searchsorted(grids[ax], d0[ax], side='right')
            d0[ax + 3] = grids[ax][i]

            # Face 1: find the outside gridline and go one inwards
            j = np.searchsorted(grids[ax], d1[ax+3], side='left')
            d1[ax] = grids[ax][j-1]

            yield AABBox(*d0), AABBox(*d1)
