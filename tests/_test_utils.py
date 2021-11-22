from typing import Union

import numpy as np


def equal_arrays(
    arr_a: Union[np.ndarray, list], arr_b: Union[np.ndarray, list]
) -> bool:
    return all([a == b for a, b in zip(arr_a, arr_b)])
