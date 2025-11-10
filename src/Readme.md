# Overview
This folder contains the source file for the C++ implementation of mpfa. The main
content of the files are:
* compressed_storage.cpp provides storage for sparse data in compressed row (CSR) format.
  Both integer and double data is supported.
* tensor.cpp cotains a representation of second order symmetric tensors.
* grid.cpp provides a class for general polytopal/polyhedral grids in two and three
  dimensions. Furthermore, the file contains a factory function for Cartesian grids in
  two and three dimensions, and a function to compute grid the geometry (areas, volumes,
  and normal vectors) for a given grid. Grids of other types such as simplexes must be
  constructed alternatively.
* multipoint_common.cpp contains helper functions that are used for mpfa and a future
  (hoped for) extension to mpsa.
* tpfa.cpp provides an implementation of the two-point flux approximation method.
* mpfa.cpp provides an implementation of the multi-point flux approximation method.

Motivated by the bridge to PorePy, the tpfa and mpfa discretization constructs, in
addition to the standard flux discretization, various other fields that are used in the
multiphysics simulations targeted by PorePy.