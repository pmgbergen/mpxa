#include "tensor.h"

#include <algorithm>
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
      m_k_yz(),
      m_diagonal_data(3, nullptr),
      m_full_data(6, nullptr)
{
    if (k_xx.size() != static_cast<size_t>(num_cells))
    {
        throw std::invalid_argument("Size of k_xx does not match num_cells.");
    }
    m_is_isotropic = true;
    m_is_diagonal = true;
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

const std::vector<double>& SecondOrderTensor::isotropic_data() const
{
    return m_k_xx;
}

const std::vector<const double*> SecondOrderTensor::diagonal_data() const
{
    if (m_dim == 2)
    {
        m_diagonal_data[0] = m_k_xx.data();
        m_diagonal_data[1] = m_k_yy.empty() ? nullptr : m_k_yy.data();
        m_diagonal_data[2] = nullptr;
    }
    else
    {
        m_diagonal_data[0] = m_k_xx.data();
        m_diagonal_data[1] = m_k_yy.empty() ? nullptr : m_k_yy.data();
        m_diagonal_data[2] = m_k_zz.empty() ? nullptr : m_k_zz.data();
    }
    return m_diagonal_data;
}

const std::vector<const double*> SecondOrderTensor::full_data() const
{
    if (m_dim == 2)
    {
        m_full_data[0] = m_k_xx.data();
        m_full_data[1] = m_k_yy.empty() ? nullptr : m_k_yy.data();
        m_full_data[2] = m_k_xy.empty() ? nullptr : m_k_xy.data();
        m_full_data[3] = nullptr;
        m_full_data[4] = nullptr;
        m_full_data[5] = nullptr;
    }
    else
    {
        m_full_data[0] = m_k_xx.data();
        m_full_data[1] = m_k_yy.empty() ? nullptr : m_k_yy.data();
        m_full_data[2] = m_k_xy.empty() ? nullptr : m_k_xy.data();
        m_full_data[3] = m_k_zz.empty() ? nullptr : m_k_zz.data();
        m_full_data[4] = m_k_xz.empty() ? nullptr : m_k_xz.data();
        m_full_data[5] = m_k_yz.empty() ? nullptr : m_k_yz.data();
    }
    return m_full_data;
}