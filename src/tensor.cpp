#include "../include/tensor.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

SecondOrderTensor::SecondOrderTensor(const int dim, const int num_cells,
                                     const std::vector<double>& k_xx)
    : m_dim(dim),
      m_num_cells(num_cells),
      m_k_xx(k_xx),
      m_k_yy(),
      m_k_xy(),
      m_k_zz(),
      m_k_xz(),
      m_k_yz()
{
    if (k_xx.size() != static_cast<size_t>(num_cells))
    {
        throw std::invalid_argument("Size of k_xx does not match num_cells.");
    }
    m_is_isotropic = true;
    m_is_diagonal = true;

    // Initialize a vector of zeros for the off-diagonal components.
    m_zeros.resize(num_cells, 0.0);
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
    m_k_yy = k_yy;
    m_is_isotropic = false;
    return *this;
}

SecondOrderTensor& SecondOrderTensor::with_kxy(const std::vector<double>& k_xy)
{
    if (k_xy.size() != static_cast<size_t>(m_num_cells))
    {
        throw std::invalid_argument("Size of k_xy does not match num_cells.");
    }
    m_k_xy = k_xy;
    m_is_diagonal = false;

    if (m_k_yy.empty())
    {
        m_k_yy = m_k_xx;
    }
    if (m_k_zz.empty() && m_dim == 3)
    {
        m_k_zz = m_k_xx;
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
    m_k_zz = k_zz;
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
    m_k_xz = k_xz;
    m_is_diagonal = false;

    if (m_k_yy.empty())
    {
        m_k_yy = m_k_xx;
    }
    if (m_k_zz.empty())
    {
        m_k_zz = m_k_xx;
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
    m_k_yz = k_yz;
    m_is_diagonal = false;

    if (m_k_yy.empty())
    {
        m_k_yy = m_k_xx;
    }
    if (m_k_zz.empty())
    {
        m_k_zz = m_k_xx;
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

double SecondOrderTensor::isotropic_data(int cell) const
{
    return m_k_xx[cell];
}

std::vector<double> SecondOrderTensor::diagonal_data(int cell) const
{
    // Always return a vector of size 3: [xx, yy, zz], zero-padded for 2D
    std::vector<double> diag(3);
    diag[0] = m_k_xx[cell];
    diag[1] = m_k_yy.empty() ? m_k_xx[cell] : m_k_yy[cell];
    diag[2] = m_k_zz.empty() ? m_k_xx[cell] : m_k_zz[cell];
    return diag;
}

std::vector<double> SecondOrderTensor::full_data(int cell) const
{
    // Always return a vector of size 6: [xx, yy, zz, xy, xz, yz], zero-padded for 2D
    std::vector<double> tensor(6);
    tensor[0] = m_k_xx[cell];
    tensor[1] = m_k_yy.empty() ? m_k_xx[cell] : m_k_yy[cell];
    tensor[2] = m_k_zz.empty() ? m_k_xx[cell] : m_k_zz[cell];
    tensor[3] = m_k_xy.empty() ? 0.0 : m_k_xy[cell];
    tensor[4] = m_k_xz.empty() ? 0.0 : m_k_xz[cell];
    tensor[5] = m_k_yz.empty() ? 0.0 : m_k_yz[cell];
    return tensor;
}