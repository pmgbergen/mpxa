import pytest
import numpy as np

import mpxa


@pytest.mark.parametrize("num_components", [1, 3, 6])
@pytest.mark.parametrize("num_cells", [1, 3])
@pytest.mark.parametrize("dim", [2, 3])
def test_tensor_bindings(num_components, num_cells, dim):
    """For now, we just test that the tensor bindings can be used without error."""
    data = 1 + np.arange(num_cells)

    if num_components == 1:
        tensor = mpxa.SecondOrderTensor(dim, num_cells, data)
    elif num_components == 3:
        tensor = mpxa.SecondOrderTensor(dim, num_cells, data, data, data)
    else:
        if dim == 3:
            tensor = mpxa.SecondOrderTensor(
                dim, num_cells, data, data, data, data, data, data
            )
