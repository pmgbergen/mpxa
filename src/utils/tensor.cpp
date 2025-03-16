#include "tensor.h"

#include <stdexcept>

SecondOrderTensor::SecondOrderTensor(const int dim, const double* k_xx)
    : m_dim(dim),
      m_k_xx(k_xx),
      m_k_yy(nullptr),
      m_k_xy(nullptr),
      m_k_zz(nullptr),
      m_k_xz(nullptr),
      m_k_yz(nullptr)
{
    m_is_isotropic = true;
    m_is_diagonal = true;
    m_k_yy = nullptr;
    m_k_xy = nullptr;
    m_k_zz = nullptr;
    m_k_xz = nullptr;
    m_k_yz = nullptr;
}

SecondOrderTensor::~SecondOrderTensor() {}

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
    m_k_yy = k_yy;
    // If the yy component is set, we assume it is not the same as the xx component. The
    // tensor is no longer isotropic (but it can still be diagonal).
    m_is_isotropic = false;
    return *this;
}

SecondOrderTensor& SecondOrderTensor::with_kxy(const double* k_xy)
{
    m_k_xy = k_xy;
    // If the xy component is set, the tensor is no longer diagonal.
    m_is_diagonal = false;

    // We will now treat this as a full tensor. If the yy component is not set, we set
    // it to be the same as the xx component. If the tensor is 3d, we also set the zz
    // component to be the same as the xx component.
    if (m_k_yy == nullptr)
    {
        m_k_yy = m_k_xx;
    }
    if (m_k_zz == nullptr && m_dim == 3)
    {
        m_k_zz = m_k_xx;
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
    m_k_zz = k_zz;
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
    m_k_xz = k_xz;
    // If the xz component is set, the tensor is no longer diagonal.
    m_is_diagonal = false;

    // We will now treat this as a full tensor. If the yy component is not set, we set
    // it to be the same as the xx component. If the zz component is not set, we set
    // it to be the same as the xx component.
    if (m_k_yy == nullptr)
    {
        m_k_yy = m_k_xx;
    }

    if (m_k_zz == nullptr)
    {
        m_k_zz = m_k_xx;
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
    m_k_yz = k_yz;
    // If the yz component is set, the tensor is no longer diagonal.
    m_is_diagonal = false;

    // We will now treat this as a full tensor. If the yy component is not set, we set
    // it to be the same as the xx component. If the zz component is not set, we set
    // it to be the same as the xx component.
    if (m_k_yy == nullptr)
    {
        m_k_yy = m_k_xx;
    }

    if (m_k_zz == nullptr)
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