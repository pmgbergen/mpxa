#ifndef FV_DISCR_H
#define FV_DISCR_H

#include <memory>  // std::unique_ptr

#include "../grid/grid.h"
#include "../utils/compressed_storage.h"
#include "../utils/tensor.h"

struct ScalarDiscretization
{
    std::unique_ptr<CompressedDataStorage<double>> flux;
    std::unique_ptr<CompressedDataStorage<double>> bound_flux;
};

ScalarDiscretization tpfa(const Grid& grid, const SecondOrderTensor& tensor);

#endif  // FV_DISCR_H