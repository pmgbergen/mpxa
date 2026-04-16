# Orchestrator Plan — mpxa C++ Refactoring + Test Loop

## Codebase Overview

| File | Lines | Role |
|---|---|---|
| `include/tensor.h` + `src/tensor.cpp` | 62 + 130 | `SecondOrderTensor` — permeability tensor storage |
| `include/compressed_storage.h` + `src/compressed_storage.cpp` | 55 + 182 | `CompressedDataStorage` — CSR sparse-matrix storage |
| `include/grid.h` + `src/grid.cpp` | 87 + 827 | `Grid` — mesh topology and geometry |
| `include/multipoint_common.h` + `src/multipoint_common.cpp` | 85 + 257 | `BasisConstructor`, `InteractionRegion` — MPFA basis functions |
| `src/tpfa.cpp` | 293 | Two-point flux approximation discretisation |
| `src/mpfa.cpp` | 1213 | Multi-point flux approximation discretisation (largest file) |
| `include/discr.h` | 33 | Abstract discretisation interface |

## Coverage Baseline

- C++ unit tests: 1 test binary (`test_mpxa`) — minimal smoke tests only; no per-class unit tests.
- Python integration tests: 15 tests pass.
- Estimated line coverage of pure-computation code by unit tests: **< 10%**.

## Backlog

| # | Module(s) | Complexity | Status | Refactoring work | Notes |
|---|---|---|---|---|---|
| 1 | `tensor.h/.cpp` | small | **done** | Replace magic `6` with `DATA_PER_CELL`; extract private `validate_size()` helper; remove `<iostream>` | Session work already started |
| 2 | `compressed_storage.h/.cpp` | small | **done** | Verify const-correctness complete; add `operator[]` bounds check | `<stdexcept>` already added |
| 3 | `grid.h/.cpp` | medium | **done** | Split `compute_geometry()` (180-line monolith) into private helpers: `compute_face_normals()`, `compute_face_centers()`, `compute_cell_centers()`, `compute_cell_volumes()` | Primary OpenMP candidate: face loop and cell loop are independent |
| 4 | `multipoint_common.h/.cpp` | medium | **done** | Split `compute_basis_functions()` into `compute_basis_functions_2d()` and `compute_basis_functions_3d()` private helpers | `inv[4][4]` — only rows 1-3 used; signed/unsigned loop comparisons |
| 5 | `tpfa.cpp` | medium | **done** | Extract `compute_internal_face_flux()` and `compute_boundary_face_flux()` helpers from `tpfa()`; rename cryptic local variables | Face loop is OpenMP candidate |
| 6 | `mpfa.cpp` (geometry) | large | **done** | Deduplicate interaction-region geometry extraction (repeated per-node blocks); extract `extract_interaction_region_geometry()` helper | Existing TODO comments at lines 621, 891 |
| 7 | `mpfa.cpp` (names + helpers) | large | **done** | Rename cryptic functions (`count_nodes_of_faces`, `count_faces_of_cells` loops); fix signed/unsigned comparisons; extract sub-helpers | Node loop (line ~616) is primary OpenMP candidate |

## Iteration Schedule

Each iteration: RefactoringAgent applies changes to scoped files → TestWriterAgent writes tests and annotates remaining issues → OrchestratorAgent updates this plan.

### Iteration 1 — tensor + compressed_storage  (scope: ≤ 2 files, small complexity)
**Scope:** `include/tensor.h`, `src/tensor.cpp`, `include/compressed_storage.h`, `src/compressed_storage.cpp`
**Refactoring goals:** Complete backlog items #1 and #2.
**Test goals:** Unit tests for `SecondOrderTensor` (constructor, setters, getters, edge cases) and `CompressedDataStorage` (construction, accessors, bounds).
**Status:** `done`

### Iteration 2 — Grid geometry split  (scope: 2 files, medium complexity)
**Scope:** `include/grid.h`, `src/grid.cpp`
**Refactoring goals:** Backlog item #3 — split `compute_geometry()`.
**Test goals:** Unit tests for each extracted geometry helper; annotate any remaining large functions.
**Status:** `done`

### Iteration 3 — Basis functions split  (scope: 2 files, medium complexity)
**Scope:** `include/multipoint_common.h`, `src/multipoint_common.cpp`
**Refactoring goals:** Backlog item #4 — split `compute_basis_functions()`.
**Test goals:** Unit tests for `BasisConstructor`, 2D and 3D basis helpers, `InteractionRegion`.
**Status:** `done`

### Iteration 4 — TPFA helpers  (scope: 1 file, medium complexity)
**Scope:** `src/tpfa.cpp`
**Refactoring goals:** Backlog item #5 — extract face-flux helpers, improve names.
**Test goals:** Unit tests for internal and boundary face flux helpers.
**Status:** `done`

### Iteration 5 — MPFA geometry + names  (scope: 1 file, large complexity)
**Scope:** `src/mpfa.cpp`
**Refactoring goals:** Backlog items #6 and #7 — geometry deduplication, rename cryptic helpers, fix signed/unsigned.
**Test goals:** Unit tests for extracted helpers; annotate `mpfa()` itself if still too large for direct testing.
**Status:** `done`

## Completion Criteria

1. Test coverage for testable code ≥ **80%** (units annotated as needing further refactoring are excluded).
2. `refactor_annotations.md` is empty — no `TODO(test-writer)` annotations remain in the codebase.

## OpenMP Opportunities Identified

These loops must not be lost during refactoring and are targets for future parallelisation:

| Location | Loop | Notes |
|---|---|---|
| `src/grid.cpp` `compute_geometry()` | Face loop + cell loop | Both are data-independent; safe for `#pragma omp parallel for` |
| `src/mpfa.cpp` `mpfa()` ~line 616 | Node loop | Largest parallelisation opportunity in the codebase |
| `src/tpfa.cpp` `tpfa()` | Face loop | Parallelisable with pre-allocated output arrays |
