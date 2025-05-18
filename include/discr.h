#ifndef FV_DISCR_H
#define FV_DISCR_H

#include <map>     // std::map
#include <memory>  // std::unique_ptr

#include "../include/compressed_storage.h"
#include "../include/grid.h"
#include "../include/tensor.h"

enum class BoundaryCondition
{
    Dirichlet,
    Neumann,
    Robin,
};

struct ScalarDiscretization
{
    std::shared_ptr<CompressedDataStorage<double>> flux;
    std::shared_ptr<CompressedDataStorage<double>> bound_flux;
    std::shared_ptr<CompressedDataStorage<double>> vector_source;
    std::shared_ptr<CompressedDataStorage<double>> bound_vector_source;
};

ScalarDiscretization tpfa(const Grid& grid, const SecondOrderTensor& tensor,
                          const std::map<int, BoundaryCondition>& bc_map);

#endif  // FV_DISCR_H