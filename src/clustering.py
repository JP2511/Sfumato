import numpy as np
import numba as nb
import pandas as pd

###############################################################################
# distance functions

@nb.njit
def eucledian_distance(center: np.ndarray, barcode: np.ndarray) -> float:
    return np.sum((center - barcode)**2)


@nb.njit
def barcode_distance(center_pos: np.ndarray, center_genes: np.ndarray, 
                        barcode_pos: np.ndarray, barcode_genes: np.ndarray,
                        S: float, m: float) -> float:
    sq_d_c = eucledian_distance(center_genes, barcode_genes)
    sq_d_s = eucledian_distance(center_pos, barcode_pos)
    return np.sqrt(sq_d_c + (sq_d_s/(S**2))*(m**2))
