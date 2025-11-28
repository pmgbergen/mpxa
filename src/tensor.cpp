#include "../include/tensor.h"

#include <algorithm>
#include <iostream>
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
    // Filling the diagonal components.
    for (auto i{0}; i < num_cells; ++i) {
        m_k_full[i * 6 + K_XX_OFFSET] = k_xx[i];
        m_k_full[i * 6 + K_YY_OFFSET] = k_xx[i];
        m_k_full[i * 6 + K_ZZ_OFFSET] = k_xx[i];
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

// Setter methods

SecondOrderTensor& SecondOrderTensor::with_kyy(const std::vector<double>& k_yy)
{
    if (k_yy.size() != static_cast<size_t>(m_num_cells))
    {
        throw std::invalid_argument("Size of k_yy does not match num_cells.");
    }
    for (auto i{0}; i < m_num_cells; ++i) {
        m_k_full[i * 6 + K_YY_OFFSET] = k_yy[i];
    }
    m_is_isotropic = false;
    return *this;
}

SecondOrderTensor& SecondOrderTensor::with_kxy(const std::vector<double>& k_xy)
{
    if (k_xy.size() != static_cast<size_t>(m_num_cells))
    {
        throw std::invalid_argument("Size of k_xy does not match num_cells.");
    }

    for (auto i{0}; i < m_num_cells; ++i) {
        m_k_full[i * 6 + K_XY_OFFSET] = k_xy[i];
    }
    m_is_isotropic = false;
    m_is_diagonal = false;

    return *this;
}

SecondOrderTensor& SecondOrderTensor::with_kzz(const std::vector<double>& k_zz)
{
    if (m_dim == 2)
    {
        throw std::runtime_error("Cannot set zz component for 2D tensor.");
    }
    if (k_zz.size() != static_cast<size_t>(m_num_cells))
    {
        throw std::invalid_argument("Size of k_zz does not match num_cells.");
    }
    for (auto i{0}; i < m_num_cells; ++i) {
        m_k_full[i * 6 + K_ZZ_OFFSET] = k_zz[i];
    }
    m_is_isotropic = false;
    return *this;
}

SecondOrderTensor& SecondOrderTensor::with_kxz(const std::vector<double>& k_xz)
{
    if (m_dim == 2)
    {
        throw std::runtime_error("Cannot set xz component for 2D tensor.");
    }
    if (k_xz.size() != static_cast<size_t>(m_num_cells))
    {
        throw std::invalid_argument("Size of k_xz does not match num_cells.");
    }
    for (auto i{0}; i < m_num_cells; ++i) {
        m_k_full[i * 6 + K_XZ_OFFSET] = k_xz[i];
    }
    m_is_isotropic = false;
    m_is_diagonal = false;

    return *this;
}

SecondOrderTensor& SecondOrderTensor::with_kyz(const std::vector<double>& k_yz)
{
    if (m_dim == 2)
    {
        throw std::runtime_error("Cannot set yz component for 2D tensor.");
    }
    if (k_yz.size() != static_cast<size_t>(m_num_cells))
    {
        throw std::invalid_argument("Size of k_yz does not match num_cells.");
    }
    for (auto i{0}; i < m_num_cells; ++i) {
        m_k_full[i * 6 + K_YZ_OFFSET] = k_yz[i];
    }
    m_is_isotropic = false;
    m_is_diagonal = false;


    return *this;
}

// Getter methods
const int SecondOrderTensor::dim() const
{
    return m_dim;
}
