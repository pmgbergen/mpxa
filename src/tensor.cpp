#include "../include/tensor.h"

#include <stdexcept>

SecondOrderTensor::SecondOrderTensor(const int dim, const int num_cells,
                                     const std::vector<double>& k_xx)
    : m_dim(dim),
      m_num_cells(num_cells)
{
    if (k_xx.size() != static_cast<size_t>(num_cells))
    {
        throw std::invalid_argument("Size of k_xx does not match num_cells.");
    }
    m_is_isotropic = true;
    m_is_diagonal = true;

    m_k_full.resize(num_cells * DATA_PER_CELL, 0.0);
    for (auto i{0}; i < num_cells; ++i) {
        m_k_full[i * DATA_PER_CELL + K_XX_OFFSET] = k_xx[i];
        m_k_full[i * DATA_PER_CELL + K_YY_OFFSET] = k_xx[i];
        m_k_full[i * DATA_PER_CELL + K_ZZ_OFFSET] = k_xx[i];
    }
}

SecondOrderTensor::~SecondOrderTensor() = default;

bool SecondOrderTensor::is_isotropic() const
{
    return m_is_isotropic;
}

bool SecondOrderTensor::is_diagonal() const
{
    return m_is_diagonal;
}

void SecondOrderTensor::set_component(size_t offset, const std::vector<double>& values,
                                      bool affects_diagonal)
{
    if (values.size() != static_cast<size_t>(m_num_cells))
    {
        throw std::invalid_argument("Component size does not match num_cells.");
    }
    for (auto i{0}; i < m_num_cells; ++i) {
        m_k_full[i * DATA_PER_CELL + offset] = values[i];
    }
    m_is_isotropic = false;
    if (affects_diagonal)
    {
        m_is_diagonal = false;
    }
}

// Setter methods

SecondOrderTensor& SecondOrderTensor::with_kyy(const std::vector<double>& k_yy)
{
    set_component(K_YY_OFFSET, k_yy, false);
    return *this;
}

SecondOrderTensor& SecondOrderTensor::with_kxy(const std::vector<double>& k_xy)
{
    set_component(K_XY_OFFSET, k_xy, true);
    return *this;
}

SecondOrderTensor& SecondOrderTensor::with_kzz(const std::vector<double>& k_zz)
{
    if (m_dim == 2)
    {
        throw std::runtime_error("Cannot set zz component for 2D tensor.");
    }
    set_component(K_ZZ_OFFSET, k_zz, false);
    return *this;
}

SecondOrderTensor& SecondOrderTensor::with_kxz(const std::vector<double>& k_xz)
{
    if (m_dim == 2)
    {
        throw std::runtime_error("Cannot set xz component for 2D tensor.");
    }
    set_component(K_XZ_OFFSET, k_xz, true);
    return *this;
}

SecondOrderTensor& SecondOrderTensor::with_kyz(const std::vector<double>& k_yz)
{
    if (m_dim == 2)
    {
        throw std::runtime_error("Cannot set yz component for 2D tensor.");
    }
    set_component(K_YZ_OFFSET, k_yz, true);
    return *this;
}

// Getter methods
int SecondOrderTensor::dim() const
{
    return m_dim;
}
