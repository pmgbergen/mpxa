#ifndef TENSOR_H
#define TENSOR_H

class SecondOrderTensor
{
   public:
    SecondOrderTensor(const int dim, const int num_cells, const double* k_xx);
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
    const int m_dim;
    const int m_num_cells;

    bool m_is_isotropic;
    bool m_is_diagonal;

    double* m_k_xx;
    double* m_k_yy;
    double* m_k_xy;
    double* m_k_zz;
    double* m_k_xz;
    double* m_k_yz;

    mutable const double* m_diagonal_data[3];
    mutable const double* m_full_data[6];
};

#endif  // TENSOR_H