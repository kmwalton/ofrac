"""Tests related to OFracGrid objects and their operation

WARNING. VERY INCOMPLETE IN TERMS OF COVERAGE OF TESTS.
"""

import unittest
import itertools
import numpy as np

from ofrac.ofracs import (OFrac, OFracGrid)


class TestOFracGrid(unittest.TestCase):

    def assertArrayEqual(self, a, b, msg=None):
        if not msg:
            msg = f'Failed. Arrays not equal:\n{a}\n{b}'
        #try:
        np.testing.assert_array_equal(a, b)
        #except AssertionError as e:
        #    raise self.failureException(msg)

    def _make_1x1_domain(self):
        return OFracGrid( domainSize=(1.,1.,1.), fx=[
            (0., 1., 0., 1., 0.5, 0.5, 0.001),
        ],)

    def test_choose_nodes(self):
        g = self._make_1x1_domain()

        with self.subTest('x=0 face'):
            ipm, ifx = g.choose_nodes_block('0, 0, 0, 1, 0, 1')
            self.assertArrayEqual(ipm, [0,2,4,6,8,10])
            self.assertArrayEqual(ifx, [4,6])

        with self.subTest('x=1 face'):
            ipm, ifx = g.choose_nodes_block('1, 1, 0, 1, 0, 1')
            self.assertArrayEqual(ipm-1, [0,2,4,6,8,10])
            self.assertArrayEqual(ifx-1, [4,6])

        g.addGridline(1,0.5)
        with self.subTest('1/2 x=0 face, extra gridline'):
            ipm, ifx = g.choose_nodes_block('0, 0, 0, 0.5, 0, 1')
            self.assertArrayEqual(ipm, [0,2,6,8,12,14])
            self.assertArrayEqual(ifx, [6,8])


    def test_ni2ng(self):

        g = self._make_1x1_domain()

        ni = np.arange(12)
        expng = np.array(list([i, j, k] for k, j, i in 
                    itertools.product(range(3), range(2), range(2))))

        self.assertArrayEqual(g.ni2ng(ni), expng)


    def test_getGridLines(self):

        g = self._make_1x1_domain()

        gl = g.getGridLines()
        self.assertArrayEqual(gl[0], [0., 1.])
        self.assertArrayEqual(gl[1], [0., 1.])
        self.assertArrayEqual(gl[2], [0., 0.5, 1.])

        self.assertArrayEqual(g.getGridLines(0), [0., 1.])
        self.assertArrayEqual(g.getGridLines('x'), [0., 1.])

if __name__ == '__main__':
    unittest.main()
