import sys
import unittest

import pandas as pd

sys.path.append('../src')
from clustering import *

###############################################################################

class TestBCMapClass(unittest.TestCase):

    # ---------------------------------------------- #
    #  tests for the initialization of BCMap class   #
    # ---------------------------------------------- #

    def test_map_init_X_EMPTY(self):
        barcodes = pd.DataFrame({'x_coor': [], 'y_coor': []})
        bc_map = BCMap(barcodes)
        solution = []
        self.assertEqual(bc_map.x_sorted, solution)


    def test_map_init_Y_EMPTY(self):
        barcodes = pd.DataFrame({'x_coor': [], 'y_coor': []})
        bc_map = BCMap(barcodes)
        solution = []
        self.assertEqual(bc_map.y_sorted, solution)


    def test_map_init_X(self):
        barcodes = pd.DataFrame({'x_coor': [1, 4, 5, 2], 
                                    'y_coor': [7, 9, 3, 2]})
        bc_map = BCMap(barcodes)
        solution = [(1,7,0), (2,2,3), (4,9,1), (5,3,2)]
        self.assertEqual(bc_map.x_sorted, solution)


    def test_map_init_Y(self):
        barcodes = pd.DataFrame({'x_coor': [1, 4, 5, 2], 
                                    'y_coor': [7, 9, 3, 2]})
        bc_map = BCMap(barcodes)
        solution = [(2,2,3), (5,3,2), (1,7,0), (4,9,1)]
        self.assertEqual(bc_map.y_sorted, solution)


    # --------------------------------------- #
    #  tests for the function binary search   #
    # --------------------------------------- #

    def test_binary_search_EMPTY(self):
        bc_map = BCMap(pd.DataFrame({'x_coor': [], 'y_coor': []}))
        self.assertRaises(ValueError, bc_map.binary_search, 
                            sorted_values=[], index=1, start=23)
    

    def test_binary_search_1_0_LOWER(self):
        bc_map = BCMap(pd.DataFrame({'x_coor': [], 'y_coor': []}))
        sorted_values = [(10, 0, 0)]
        index=0
        start=2
        result = bc_map.binary_search(sorted_values, index, start)
        self.assertEqual(result, 0)


    def test_binary_search_1_0_EQUAL(self):
        bc_map = BCMap(pd.DataFrame({'x_coor': [], 'y_coor': []}))
        sorted_values = [(10, 20, 20)]
        index=0
        start=10
        result = bc_map.binary_search(sorted_values, index, start)
        self.assertEqual(result, 0)


    def test_binary_search_1_0_HIGHER(self):
        bc_map = BCMap(pd.DataFrame({'x_coor': [], 'y_coor': []}))
        sorted_values = [(10, 20, 20)]
        index=0
        start=15
        result = bc_map.binary_search(sorted_values, index, start)
        self.assertIsNone(result)
    

    def test_binary_search_2_0_LOWER(self):
        bc_map = BCMap(pd.DataFrame({'x_coor': [], 'y_coor': []}))
        sorted_values = [(10, 20, 20), (15, 20, 20)]
        index=0
        start=2
        result = bc_map.binary_search(sorted_values, index, start)
        self.assertEqual(result, 0)


    def test_binary_search_2_0_EQUAL_0(self):
        bc_map = BCMap(pd.DataFrame({'x_coor': [], 'y_coor': []}))
        sorted_values = [(10, 20, 20), (15, 20, 20)]
        index=0
        start=10
        result = bc_map.binary_search(sorted_values, index, start)
        self.assertEqual(result, 0)


    def test_binary_search_2_0_MIDDLE(self):
        bc_map = BCMap(pd.DataFrame({'x_coor': [], 'y_coor': []}))
        sorted_values = [(10, 20, 20), (15, 20, 20)]
        index=0
        start=12
        result = bc_map.binary_search(sorted_values, index, start)
        self.assertEqual(result, 0)


    def test_binary_search_2_0_EQUAL_1(self):
        bc_map = BCMap(pd.DataFrame({'x_coor': [], 'y_coor': []}))
        sorted_values = [(10, 20, 20), (15, 20, 20)]
        index=0
        start=15
        result = bc_map.binary_search(sorted_values, index, start)
        self.assertEqual(result, 1)


    def test_binary_search_2_0_HIGHER(self):
        bc_map = BCMap(pd.DataFrame({'x_coor': [], 'y_coor': []}))
        sorted_values = [(10, 20, 20), (15, 20, 20)]
        index=0
        start=20
        result = bc_map.binary_search(sorted_values, index, start)
        self.assertIsNone(result)


    def test_binary_search_3_0_LOWER(self):
        bc_map = BCMap(pd.DataFrame({'x_coor': [], 'y_coor': []}))
        sorted_values = [(0, 20, 20), (10, 20, 20), (15, 20, 20)]
        index=0
        start=-5
        result = bc_map.binary_search(sorted_values, index, start)
        self.assertEqual(result, 0)


    def test_binary_search_3_0_EQUAL_0(self):
        bc_map = BCMap(pd.DataFrame({'x_coor': [], 'y_coor': []}))
        sorted_values = [(0, 20, 20), (10, 20, 20), (15, 20, 20)]
        index=0
        start=0
        result = bc_map.binary_search(sorted_values, index, start)
        self.assertEqual(result, 0)


    def test_binary_search_3_0_MIDDLE_0_1(self):
        bc_map = BCMap(pd.DataFrame({'x_coor': [], 'y_coor': []}))
        sorted_values = [(0, 20, 20), (10, 20, 20), (15, 20, 20)]
        index=0
        start=5
        result = bc_map.binary_search(sorted_values, index, start)
        self.assertEqual(result, 0)


    def test_binary_search_3_0_EQUAL_1(self):
        bc_map = BCMap(pd.DataFrame({'x_coor': [], 'y_coor': []}))
        sorted_values = [(0, 20, 20), (10, 20, 20), (15, 20, 20)]
        index=0
        start=10
        result = bc_map.binary_search(sorted_values, index, start)
        self.assertEqual(result, 1)


    def test_binary_search_3_0_MIDDLE_1_2(self):
        bc_map = BCMap(pd.DataFrame({'x_coor': [], 'y_coor': []}))
        sorted_values = [(0, 20, 20), (10, 20, 20), (15, 20, 20)]
        index=0
        start=12
        result = bc_map.binary_search(sorted_values, index, start)
        self.assertEqual(result, 1)


    def test_binary_search_3_0_EQUAL_2(self):
        bc_map = BCMap(pd.DataFrame({'x_coor': [], 'y_coor': []}))
        sorted_values = [(0, 20, 20), (10, 20, 20), (15, 20, 20)]
        index=0
        start=15
        result = bc_map.binary_search(sorted_values, index, start)
        self.assertEqual(result, 2)
    

    def test_binary_search_3_0_HIGHER(self):
        bc_map = BCMap(pd.DataFrame({'x_coor': [], 'y_coor': []}))
        sorted_values = [(0, 20, 20), (10, 20, 20), (15, 20, 20)]
        index=0
        start=18
        result = bc_map.binary_search(sorted_values, index, start)
        self.assertIsNone(result)


    def test_binary_search_1_1_LOWER(self):
        bc_map = BCMap(pd.DataFrame({'x_coor': [], 'y_coor': []}))
        sorted_values = [(0, 10, 0)]
        index=1
        start=2
        result = bc_map.binary_search(sorted_values, index, start)
        self.assertEqual(result, 0)


    def test_binary_search_1_1_EQUAL(self):
        bc_map = BCMap(pd.DataFrame({'x_coor': [], 'y_coor': []}))
        sorted_values = [(20, 10, 20)]
        index=1
        start=10
        result = bc_map.binary_search(sorted_values, index, start)
        self.assertEqual(result, 0)


    def test_binary_search_1_1_HIGHER(self):
        bc_map = BCMap(pd.DataFrame({'x_coor': [], 'y_coor': []}))
        sorted_values = [(20, 10, 20)]
        index=1
        start=15
        result = bc_map.binary_search(sorted_values, index, start)
        self.assertIsNone(result)
    

    def test_binary_search_2_1_LOWER(self):
        bc_map = BCMap(pd.DataFrame({'x_coor': [], 'y_coor': []}))
        sorted_values = [(0, 5, 0), (0, 10, 0)]
        index=1
        start=2
        result = bc_map.binary_search(sorted_values, index, start)
        self.assertEqual(result, 0)


    def test_binary_search_2_1_EQUAL_0(self):
        bc_map = BCMap(pd.DataFrame({'x_coor': [], 'y_coor': []}))
        sorted_values = [(0, 10, 0), (0, 20, 0)]
        index=1
        start=10
        result = bc_map.binary_search(sorted_values, index, start)
        self.assertEqual(result, 0)


    def test_binary_search_2_1_MIDDLE(self):
        bc_map = BCMap(pd.DataFrame({'x_coor': [], 'y_coor': []}))
        sorted_values = [(0, 10, 0), (0, 20, 0)]
        index=1
        start=12
        result = bc_map.binary_search(sorted_values, index, start)
        self.assertEqual(result, 0)


    def test_binary_search_2_1_EQUAL_1(self):
        bc_map = BCMap(pd.DataFrame({'x_coor': [], 'y_coor': []}))
        sorted_values = [(0, 10, 0), (0, 15, 0)]
        index=1
        start=15
        result = bc_map.binary_search(sorted_values, index, start)
        self.assertEqual(result, 1)


    def test_binary_search_2_1_HIGHER(self):
        bc_map = BCMap(pd.DataFrame({'x_coor': [], 'y_coor': []}))
        sorted_values = [(0, 10, 0), (0, 15, 0)]
        index=1
        start=20
        result = bc_map.binary_search(sorted_values, index, start)
        self.assertIsNone(result)


    def test_binary_search_3_1_LOWER(self):
        bc_map = BCMap(pd.DataFrame({'x_coor': [], 'y_coor': []}))
        sorted_values = [(-10, 0, -10), (-10, 5, -10), (-10, 10, -10)]
        index=1
        start=-5
        result = bc_map.binary_search(sorted_values, index, start)
        self.assertEqual(result, 0)


    def test_binary_search_3_1_EQUAL_0(self):
        bc_map = BCMap(pd.DataFrame({'x_coor': [], 'y_coor': []}))
        sorted_values = [(-10, 0, -10), (-10, 5, -10), (-10, 10, -10)]
        index=1
        start=0
        result = bc_map.binary_search(sorted_values, index, start)
        self.assertEqual(result, 0)


    def test_binary_search_3_1_MIDDLE_0_1(self):
        bc_map = BCMap(pd.DataFrame({'x_coor': [], 'y_coor': []}))
        sorted_values = [(0, 0, 0), (0, 10, 0), (0, 15, 0)]
        index=1
        start=5
        result = bc_map.binary_search(sorted_values, index, start)
        self.assertEqual(result, 0)


    def test_binary_search_3_1_EQUAL_1(self):
        bc_map = BCMap(pd.DataFrame({'x_coor': [], 'y_coor': []}))
        sorted_values = [(0, 5, 0), (0, 10, 0), (0, 15, 0)]
        index=1
        start=10
        result = bc_map.binary_search(sorted_values, index, start)
        self.assertEqual(result, 1)


    def test_binary_search_3_1_MIDDLE_1_2(self):
        bc_map = BCMap(pd.DataFrame({'x_coor': [], 'y_coor': []}))
        sorted_values = [(0, 5, 0), (0, 10, 0), (0, 15, 0)]
        index=1
        start=12
        result = bc_map.binary_search(sorted_values, index, start)
        self.assertEqual(result, 1)


    def test_binary_search_3_1_EQUAL_2(self):
        bc_map = BCMap(pd.DataFrame({'x_coor': [], 'y_coor': []}))
        sorted_values = [(0, 5, 0), (0, 10, 0), (0, 15, 0)]
        index=1
        start=15
        result = bc_map.binary_search(sorted_values, index, start)
        self.assertEqual(result, 2)
    

    def test_binary_search_3_1_HIGHER(self):
        bc_map = BCMap(pd.DataFrame({'x_coor': [], 'y_coor': []}))
        sorted_values = [(0, 5, 0), (0, 10, 0), (0, 15, 0)]
        index=1
        start=18
        result = bc_map.binary_search(sorted_values, index, start)
        self.assertIsNone(result)


###############################################################################

if __name__ == '__main__':
    unittest.main()