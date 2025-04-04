#include "tensor.h"

#include <memory>
#include <stdexcept>

SecondOrderTensor::SecondOrderTensor(const int dim, const int num_cells, const double* k_xx)
    : m_dim(dim),
      m_num_cells(num_cells),
      m_k_xx(std::make_unique<double[]>(num_cells)),
      m_k_yy(nullptr),
      m_k_xy(nullptr),
      m_k_zz(nullptr),
      m_k_xz(nullptr),
      m_k_yz(nullptr)
{
    std::copy(k_xx, k_xx + num_cells, m_k_xx.get());
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

SecondOrderTensor& SecondOrderTensor::with_kyy(const double* k_yy)
{
    m_k_yy = std::make_unique<double[]>(m_num_cells);
    std::copy(k_yy, k_yy + m_num_cells, m_k_yy.get());
    m_is_isotropic = false;
    return *this;
}

SecondOrderTensor& SecondOrderTensor::with_kxy(const double* k_xy)
{
    m_k_xy = std::make_unique<double[]>(m_num_cells);
    std::copy(k_xy, k_xy + m_num_cells, m_k_xy.get());
    m_is_diagonal = false;

    if (!m_k_yy)
    {
        m_k_yy = std::make_unique<double[]>(m_num_cells);
        std::copy(m_k_xx.get(), m_k_xx.get() + m_num_cells, m_k_yy.get());
    }
    if (!m_k_zz && m_dim == 3)
    {
        m_k_zz = std::make_unique<double[]>(m_num_cells);
        std::copy(m_k_xx.get(), m_k_xx.get() + m_num_cells, m_k_zz.get());
    }
    m_is_isotropic = false;
    m_is_diagonal = false;

    return *this;
}

SecondOrderTensor& SecondOrderTensor::with_kzz(const double* k_zz)
{
    if (m_dim == 2)
    {
        throw std::runtime_error("Cannot set zz component for 2D tensor.");
    }
    m_k_zz = std::make_unique<double[]>(m_num_cells);
    std::copy(k_zz, k_zz + m_num_cells, m_k_zz.get());
    m_is_isotropic = false;
    return *this;
}

SecondOrderTensor& SecondOrderTensor::with_kxz(const double* k_xz)
{
    if (m_dim == 2)
    {
        throw std::runtime_error("Cannot set xz component for 2D tensor.");
    }
    m_k_xz = std::make_unique<double[]>(m_num_cells);
    std::copy(k_xz, k_xz + m_num_cells, m_k_xz.get());
    m_is_diagonal = false;

    if (!m_k_yy)
    {
        m_k_yy = std::make_unique<double[]>(m_num_cells);
        std::copy(m_k_xx.get(), m_k_xx.get() + m_num_cells, m_k_yy.get());
    }
    if (!m_k_zz)
    {
        m_k_zz = std::make_unique<double[]>(m_num_cells);
        std::copy(m_k_xx.get(), m_k_xx.get() + m_num_cells, m_k_zz.get());
    }
    m_is_isotropic = false;
    m_is_diagonal = false;
    return *this;
}

SecondOrderTensor& SecondOrderTensor::with_kyz(const double* k_yz)
{
    if (m_dim == 2)
    {
        throw std::runtime_error("Cannot set yz component for 2D tensor.");
    }
    m_k_yz = std::make_unique<double[]>(m_num_cells);
    std::copy(k_yz, k_yz + m_num_cells, m_k_yz.get());
    m_is_diagonal = false;

    if (!m_k_yy)
    {
        m_k_yy = std::make_unique<double[]>(m_num_cells);
        std::copy(m_k_xx.get(), m_k_xx.get() + m_num_cells, m_k_yy.get());
    }
    if (!m_k_zz)
    {
        m_k_zz = std::make_unique<double[]>(m_num_cells);
        std::copy(m_k_xx.get(), m_k_xx.get() + m_num_cells, m_k_zz.get());
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
// Get the isotropic data
const double* SecondOrderTensor::isotropic_data() const
{
    return m_k_xx.get();
}
// Get the diagonal data
const double* const* SecondOrderTensor::diagonal_data() const
{
    if (m_dim == 2)
    {
        m_diagonal_data[0] = m_k_xx.get();
        m_diagonal_data[1] = m_k_yy ? m_k_yy.get() : nullptr;
        m_diagonal_data[2] = nullptr;
    }
    else
    {
        m_diagonal_data[0] = m_k_xx.get();
        m_diagonal_data[1] = m_k_yy ? m_k_yy.get() : nullptr;
        m_diagonal_data[2] = m_k_zz ? m_k_zz.get() : nullptr;
    }
    return m_diagonal_data;
}
// Get the full data
const double* const* SecondOrderTensor::full_data() const
{
    if (m_dim == 2)
    {
        m_full_data[0] = m_k_xx.get();
        m_full_data[1] = m_k_yy ? m_k_yy.get() : nullptr;
        m_full_data[2] = m_k_xy ? m_k_xy.get() : nullptr;
        m_full_data[3] = nullptr;
        m_full_data[4] = nullptr;
        m_full_data[5] = nullptr;
    }
    else
    {
        m_full_data[0] = m_k_xx.get();
        m_full_data[1] = m_k_yy ? m_k_yy.get() : nullptr;
        m_full_data[2] = m_k_xy ? m_k_xy.get() : nullptr;
        m_full_data[3] = m_k_zz ? m_k_zz.get() : nullptr;
        m_full_data[4] = m_k_xz ? m_k_xz.get() : nullptr;
        m_full_data[5] = m_k_yz ? m_k_yz.get() : nullptr;
    }
    return m_full_data;
}