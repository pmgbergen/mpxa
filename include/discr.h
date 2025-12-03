#ifndef FV_DISCR_H
#define FV_DISCR_H

#include <unordered_map>     // std::unordered_map
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
    std::shared_ptr<CompressedDataStorage<double>> bound_pressure_vector_source;
    std::shared_ptr<CompressedDataStorage<double>> bound_pressure_cell;
    std::shared_ptr<CompressedDataStorage<double>> bound_pressure_face;
};

ScalarDiscretization tpfa(const Grid& grid, const SecondOrderTensor& tensor,
                          const std::unordered_map<int, BoundaryCondition>& bc_map);

ScalarDiscretization mpfa(const Grid& grid, const SecondOrderTensor& tensor,
                          const std::unordered_map<int, BoundaryCondition>& bc_map);

#endif  // FV_DISCR_H