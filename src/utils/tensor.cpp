#include "tensor.h"

#include <stdexcept>

SecondOrderTensor::SecondOrderTensor(const int dim, const int num_cells, const double* k_xx)
    : m_dim(dim),
      m_num_cells(num_cells),
      m_k_xx(new double[num_cells]),
      m_k_yy(nullptr),
      m_k_xy(nullptr),
      m_k_zz(nullptr),
      m_k_xz(nullptr),
      m_k_yz(nullptr)
{
    std::copy(k_xx, k_xx + num_cells, m_k_xx);
    m_is_isotropic = true;
    m_is_diagonal = true;
    m_k_yy = nullptr;
    m_k_xy = nullptr;
    m_k_zz = nullptr;
    m_k_xz = nullptr;
    m_k_yz = nullptr;
}

SecondOrderTensor::~SecondOrderTensor()
{
    delete[] m_k_xx;
    delete[] m_k_yy;
    delete[] m_k_xy;
    delete[] m_k_zz;
    delete[] m_k_xz;
    delete[] m_k_yz;
}

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
    delete[] m_k_yy;                              // Free existing memory
    m_k_yy = new double[m_num_cells];             // Allocate new memory
    std::copy(k_yy, k_yy + m_num_cells, m_k_yy);  // Copy data
    // If the yy component is set, we assume it is not the same as the xx component. The
    // tensor is no longer isotropic (but it can still be diagonal).
    m_is_isotropic = false;
    return *this;
}

SecondOrderTensor& SecondOrderTensor::with_kxy(const double* k_xy)
{
    delete[] m_k_xy;                              // Free existing memory
    m_k_xy = new double[m_num_cells];             // Allocate new memory
    std::copy(k_xy, k_xy + m_num_cells, m_k_xy);  // Copy data
    // If the xy component is set, the tensor is no longer diagonal.
    m_is_diagonal = false;

    // We will now treat this as a full tensor. If the yy component is not set, we set
    // it to be the same as the xx component. If the tensor is 3d, we also set the zz
    // component to be the same as the xx component.
    if (m_k_yy == nullptr)
    {
        m_k_yy = new double[m_num_cells];
        std::copy(m_k_xx, m_k_xx + m_num_cells, m_k_yy);
    }
    if (m_k_zz == nullptr && m_dim == 3)
    {
        m_k_zz = new double[m_num_cells];
        std::copy(m_k_xx, m_k_xx + m_num_cells, m_k_zz);
    }
    // After this operation, the tensor is no longer considered isotropic.
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
    delete[] m_k_zz;                              // Free existing memory
    m_k_zz = new double[m_num_cells];             // Allocate new memory
    std::copy(k_zz, k_zz + m_num_cells, m_k_zz);  // Copy data
    // If the zz component is set, we assume it is not the same as the xx component. The
    // tensor is no longer isotropic (but it can still be diagonal).
    m_is_isotropic = false;
    return *this;
}

SecondOrderTensor& SecondOrderTensor::with_kxz(const double* k_xz)
{
    if (m_dim == 2)
    {
        throw std::runtime_error("Cannot set xz component for 2D tensor.");
    }
    delete[] m_k_xz;                              // Free existing memory
    m_k_xz = new double[m_num_cells];             // Allocate new memory
    std::copy(k_xz, k_xz + m_num_cells, m_k_xz);  // Copy data
    // If the xz component is set, the tensor is no longer diagonal.
    m_is_diagonal = false;

    // We will now treat this as a full tensor. If the yy component is not set, we set
    // it to be the same as the xx component. If the zz component is not set, we set
    // it to be the same as the xx component.
    if (m_k_yy == nullptr)
    {
        m_k_yy = new double[m_num_cells];
        std::copy(m_k_xx, m_k_xx + m_num_cells, m_k_yy);
    }
    if (m_k_zz == nullptr)
    {
        m_k_zz = new double[m_num_cells];
        std::copy(m_k_xx, m_k_xx + m_num_cells, m_k_zz);
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
    delete[] m_k_yz;                              // Free existing memory
    m_k_yz = new double[m_num_cells];             // Allocate new memory
    std::copy(k_yz, k_yz + m_num_cells, m_k_yz);  // Copy data
    // If the yz component is set, the tensor is no longer diagonal.
    m_is_diagonal = false;

    // We will now treat this as a full tensor. If the yy component is not set, we set
    // it to be the same as the xx component. If the zz component is not set, we set
    // it to be the same as the xx component.
    if (m_k_yy == nullptr)
    {
        m_k_yy = new double[m_num_cells];
        std::copy(m_k_xx, m_k_xx + m_num_cells, m_k_yy);
    }
    if (m_k_zz == nullptr)
    {
        m_k_zz = new double[m_num_cells];
        std::copy(m_k_xx, m_k_xx + m_num_cells, m_k_zz);
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
    return m_k_xx;
}
// Get the diagonal data
const double* const* SecondOrderTensor::diagonal_data() const
{
    if (m_dim == 2)
    {
        m_diagonal_data[0] = m_k_xx;
        m_diagonal_data[1] = m_k_yy;
        m_diagonal_data[2] = nullptr;
    }
    else
    {
        m_diagonal_data[0] = m_k_xx;
        m_diagonal_data[1] = m_k_yy;
        m_diagonal_data[2] = m_k_zz;
    }
    return m_diagonal_data;
}
// Get the full data
const double* const* SecondOrderTensor::full_data() const
{
    if (m_dim == 2)
    {
        m_full_data[0] = m_k_xx;
        m_full_data[1] = m_k_yy;
        m_full_data[2] = m_k_xy;
        m_full_data[3] = nullptr;
        m_full_data[4] = nullptr;
        m_full_data[5] = nullptr;
    }
    else
    {
        m_full_data[0] = m_k_xx;
        m_full_data[1] = m_k_yy;
        m_full_data[2] = m_k_xy;
        m_full_data[3] = m_k_zz;
        m_full_data[4] = m_k_xz;
        m_full_data[5] = m_k_yz;
    }
    return m_full_data;
}