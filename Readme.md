# Multipoint flux discretizations

This repository provides an implementation of the multipoint flux approximation method in 
2d and 3d.
The core implementation is in C++, with python bindings provided for a limited set of the
functionality.
The primary motivation is to provide accelerated discretizations for 
[PorePy](https://github.com/pmgbergen/porepy), with suitable wrappers under implementation.

The software provided here, both the C++ code and Python wrappers, should be considered
immature and be used with some caution.

# Getting started
Currently, the best starting point for the software is likely the PorePy wrappers, see
[here](https://github.com/keileg/mpxa/tree/main/tests/python/porepy), in particular
the file `test_tpfa.py`.

# Licence
See [Licence](https://github.com/keileg/mpxa/blob/main/LICENSE).
