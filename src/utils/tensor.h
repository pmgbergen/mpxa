#ifndef TENSOR_H
#define TENSOR_H

class SecondOrderTensor
{
   public:
    SecondOrderTensor(const int dim, const double* k_xx);
    ~SecondOrderTensor();

    // Setters for optional tensor components in 2d
    SecondOrderTensor& with_kyy(const double* k_yy);
    SecondOrderTensor& with_kxy(const double* k_xy);
    // Setters for optional tensor components in 3d
    SecondOrderTensor& with_kzz(const double* k_zz);
    SecondOrderTensor& with_kxz(const double* k_xz);
    SecondOrderTensor& with_kyz(const double* k_yz);

    // Get properties of the tensor
    bool is_isotropic() const;
    bool is_diagonal() const;

    const int dim() const;

    const double* isotropic_data() const;
    const double* const* diagonal_data() const;
    const double* const* full_data() const;

   private:
    int m_dim;

    bool m_is_isotropic;
    bool m_is_diagonal;

    const double* m_k_xx;
    const double* m_k_yy;
    const double* m_k_xy;
    const double* m_k_zz;
    const double* m_k_xz;
    const double* m_k_yz;

    mutable const double* m_diagonal_data[3];
    mutable const double* m_full_data[6];
};

#endif  // TENSOR_H