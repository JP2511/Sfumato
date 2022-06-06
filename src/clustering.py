import numpy as np
import numba as nb
import pandas as pd


###############################################################################
# distance functions

@nb.njit
def eucledian_distance(center: np.ndarray, barcode: np.ndarray) -> float:
    """Calculates the eucledian distance between the coordinates of the center
    and of the barcode.

    Args:
        center (np.ndarray): spatial coordinates of the center.
        barcode (np.ndarray): spatial coordinates of the barcode.

    Returns:
        float: eucledian distance between the coordinates of the center and of
            the barcode.
    """
    return np.sum((center - barcode)**2)


@nb.njit
def barcode_distance(center_pos: np.ndarray, center_genes: np.ndarray, 
                        barcode_pos: np.ndarray, barcode_genes: np.ndarray,
                        S: float, m: float) -> float:
    """Calculates the distance between a barcode and a center.

    Args:
        center_pos (np.ndarray): spatial coordinates of the center.
        center_genes (np.ndarray): gene counts of the center.
        barcode_pos (np.ndarray): spatial coordinates of the barcode.
        barcode_genes (np.ndarray): gene counts of the barcode.
        S (float): grid interval to consider in the map.
        m (float): weight that evaluates the importance of the gene count simi-
            larity in comparison with the spatial closeness.

    Returns:
        float: measure of distance between the center and the barcode.
    """

    sq_d_c = eucledian_distance(center_genes, barcode_genes)
    sq_d_s = eucledian_distance(center_pos, barcode_pos)
    return np.sqrt(sq_d_c + (sq_d_s/(S**2))*(m**2))


###############################################################################

class BCMap:
    """Class that creates and handles a map of the barcodes. This map is purely
    spatial, so it only takes into account the x and y coordinates of each 
    barcode and not the gene counts.
    """

    def __init__(self, barcodes: pd.DataFrame, k: int):
        """Initiates the map of the barcodes.

        Args:
            barcodes (pd.DataFrame): dataset that contains the coordinates of
                each barcode.
            k (int): number of pixels or cluster centers to use during the 
                clustering of the barcodes.
        
        Requires:
            barcodes: the x coordinate should be stored in a column named 
                \"x_coor\".
            barcodes: the y coordinate should be stored in a column named
                \"y_coor\".
        """
        xy = [(x, y, i) for i, (x, y) in enumerate(zip(barcodes.x_coor, 
                                                        barcodes.y_coor))]
        self.x_sorted = sorted(xy, key=lambda triple: triple[0])
        self.y_sorted = sorted(xy, key=lambda triple: triple[1])
        self.S = np.sqrt(len(barcodes)/k)


    def bs_mechanism(self, sorted_values: list, index: int, start: float, 
                        i: int, j: int) -> int:
        """Performs the main mechanism of binary search.

        Args:
            sorted_values (list[tuple]): sorted list of values.
            index (int): index of the value of the element of the list to 
                consider in the comparisons.
            start (float): value that determines the index of the list to 
                return.
            i (int): lower index to consider in the binary search.
            j (int): upper index to consider in the binary search.

        Returns:
            int: index of the element in the list whose value is just below or 
                just above the start value. 
        """
        
        mid = int(np.ceil((i+j)/2))
        if start == sorted_values[mid][index]:
            return mid
        elif mid == i or mid == j:
            return i
        elif start < sorted_values[mid][index]:
            return self.bs_mechanism(sorted_values, index, start, i, mid)
        else:
            return self.bs_mechanism(sorted_values, index, start, mid, j)


    def binary_search(self, sorted_values: list, index: int, 
                        start: float) -> int:
        """Performs binary search on the list given, to find the index of the
        element whose value is the minimum value of the list above the start or
        the maximum just below start.

        Args:
            sorted_values (list[tuple]): sorted list of elements.
            index (int): index of the value of the element of the list with 
                which comparisons are made.
            start (float): value that determines the index of which element of
                the list to return.

        Raises:
            ValueError: in case the list is empty, it raises a ValueError as, 
                it indicates that there are no barcodes to build a map, which is
                problematic to say the least.

        Returns:
            int: index of the element of the list whose value at the index given
                is either just below or just above the start value.
        """

        if len(sorted_values) == 0:
            raise ValueError("No values to search through. There are no" +
                                " barcodes at this point.")
        
        j = len(sorted_values) - 1

        if start < sorted_values[0][index]:
            return 0
        elif start > sorted_values[j][index]:
            return None

        return self.bs_mechanism(sorted_values, index, start, 0, j)


    def find_bc_range(self, sorted_values: list, index: int, start: float, 
                        stop: float) -> set:
        """Obtains the elements of the given list that have values between the
        start and stop values.

        Args:
            sorted_values (list[tuple]): list that contains the elements to be 
                chosen.
            index (int): index of the element of the tuple (element of the 
                list) whose value is to be between start and stop values.
            start (float): lower bound on the value of the elements selected.
            stop (float): tight upper bound on the value of the elements 
                selected-

        Returns:
            ranged_barcodes (set): elements of the sorted_values, whose value
                on the specified index, is between start and stop values.
        
        Requires:
            sorted_values: must be sorted in increasing order.
        """
        
        initial_index = self.binary_search(sorted_values, index, start)
        ranged_barcodes = set()
        
        while sorted_values[initial_index] <= stop:
            ranged_barcodes.add(sorted_values[initial_index])
            initial_index += 1
        
        return ranged_barcodes


    def find_bc_in_region(self, center: tuple) -> set:
        """Finds the barcodes that are within 2S units from the center in the
        spatial component.

        Args:
            center (tuple): x and y coordinates of the center.

        Returns:
            bc_range (set): coordinates of the barcodes that are in 2S units 
                from the center.
        """

        x_c, y_c = center
        bc_range = self.find_bc_range(self.x_sorted, x_c - 2*self.S, 
                                        x_c + 2*self.S)
        bc_range &= self.find_bc_range(self.y_sorted, y_c - 2*self.S, 
                                        y_c + 2*self.S)
        
        return bc_range



###############################################################################