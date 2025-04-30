#include <pybind11/pybind11.h>

#include "../../include/tensor.h"

namespace py = pybind11;

// PYBIND11_MODULE(tensor, m)
void init_tensor(py::module_ &m)
{
    py::class_<SecondOrderTensor, std::shared_ptr<SecondOrderTensor>>(m, "SecondOrderTensor")
        // Constructor for isotropic tensor.
        .def(py::init(
            [](int dim, int num_cells, py::array_t<double> k_xx)
            {
                // Convert numpy array to std::vector
                std::vector<double> k_xx_vec(k_xx.data(), k_xx.data() + k_xx.size());
                return std::make_shared<SecondOrderTensor>(dim, num_cells, k_xx_vec);
            }))
        // Constructor for tensor with three components. Will be intepreted as a full 2d tensor
        // if dim == 2, and as a diagonal tensor if dim == 3.
        .def(py::init(
            [](int dim, int num_cells, py::array_t<double> k_xx, py::array_t<double> k_yy,
               py::array_t<double> k_zz)
            {
                // Convert numpy arrays to std::vector
                std::vector<double> k_xx_vec(k_xx.data(), k_xx.data() + k_xx.size());
                std::vector<double> k_yy_vec(k_yy.data(), k_yy.data() + k_yy.size());
                std::vector<double> k_zz_vec(k_zz.data(), k_zz.data() + k_zz.size());
                auto tensor = std::make_shared<SecondOrderTensor>(dim, num_cells, k_xx_vec);
                tensor->with_kyy(k_yy_vec);
                if (dim == 2)
                {
                    tensor->with_kxy(k_zz_vec);
                }
                else
                {
                    tensor->with_kzz(k_zz_vec);
                }
                return tensor;
            }))
        // Constructor for full tensor. Must be 3D.
        .def(py::init(
            [](int dim, int num_cells, py::array_t<double> k_xx, py::array_t<double> k_yy,
               py::array_t<double> k_zz, py::array_t<double> k_xy, py::array_t<double> k_xz,
               py::array_t<double> k_yz)
            {
                // Convert numpy arrays to std::vector
                std::vector<double> k_xx_vec(k_xx.data(), k_xx.data() + k_xx.size());
                std::vector<double> k_yy_vec(k_yy.data(), k_yy.data() + k_yy.size());
                std::vector<double> k_zz_vec(k_zz.data(), k_zz.data() + k_zz.size());
                std::vector<double> k_xy_vec(k_xy.data(), k_xy.data() + k_xy.size());
                std::vector<double> k_xz_vec(k_xz.data(), k_xz.data() + k_xz.size());
                std::vector<double> k_yz_vec(k_yz.data(), k_yz.data() + k_yz.size());

                auto tensor = std::make_shared<SecondOrderTensor>(dim, num_cells, k_xx_vec);
                tensor->with_kyy(k_yy_vec);
                tensor->with_kzz(k_zz_vec);
                tensor->with_kxy(k_xy_vec);
                tensor->with_kxz(k_xz_vec);
                tensor->with_kyz(k_yz_vec);
                return tensor;
            }));
    // .def("is_isotropic", &SecondOrderTensor::is_isotropic)
    // .def("is_diagonal", &SecondOrderTensor::is_diagonal)
    // .def("dim", &SecondOrderTensor::dim)
    // .def("isotropic_data", &SecondOrderTensor::isotropic_data)
    // .def("diagonal_data", &SecondOrderTensor::diagonal_data)
    // .def("full_data", &SecondOrderTensor::full_data);
}
