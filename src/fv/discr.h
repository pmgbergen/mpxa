#ifndef FV_DISCR_H
#define FV_DISCR_H

#include <map>     // std::map
#include <memory>  // std::unique_ptr

#include "../grid/grid.h"
#include "../utils/compressed_storage.h"
#include "../utils/tensor.h"

enum class BoundaryCondition
{
    Dirichlet,
    Neumann,
    Robin,
};

struct ScalarDiscretization
{
    std::unique_ptr<CompressedDataStorage<double>> flux;
    std::unique_ptr<CompressedDataStorage<double>> bound_flux;
};

ScalarDiscretization tpfa(const Grid& grid, const SecondOrderTensor& tensor,
                          const std::map<int, BoundaryCondition>& bc_map);

#endif  // FV_DISCR_H