# Pic5_public

Companion repository for the paper [https://arxiv.org/abs/2606.29309].

This repository contains the Julia scripts used to enumerate mod $2$ toric-colorable weak pseudomanifolds and extract toric seeds for smooth complete toric varieties of Picard number $4$ and $5$.

---

## Requirements

**Julia** (tested with Julia 1.x). This repository includes a `Project.toml` and a `Manifest.toml` that pin all dependencies. To reproduce the exact environment:

```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

Then prefix every script invocation with `--project=.`, e.g.:

```bash
julia --project=. --threads auto RUN.jl
```

> `SparseArrays`, `LinearAlgebra`, and `Base.Threads` are part of the Julia standard library and are already available without installation.

---

## Repository structure

```
Pic5_public/
│
├── simplicial_complex_utilities.jl   # Core utilities for simplicial complexes
├── enumerate_kernel.jl               # Gray code kernel enumeration (Alg. 1 & 2)
│
├── Picard_4/                         # Scripts for Picard number 4
│   ├── RUN.jl                        # One-shot pipeline launcher
│   ├── create_mat_and_iso_DB_all_links.jl  # Build MatDB and isoDB (Alg. 3)
│   ├── create_mat_and_iso_DB.jl            # Auxiliary / alternative DB builder
│   ├── enumerate_pseudomanifolds.jl        # Enumerate weak pseudomanifolds (Alg. 4)
│   ├── reduce_autom_isom_seed_PLS.jl       # Post-processing (steps 1–3)
│   ├── resources/
│   │   ├── mat_DB.jls                # Serialized MatDB
│   │   ├── iso_DB.jls                # Serialized isoDB (restricted links)
│   │   └── iso_DB_all.jls            # Serialized isoDB (all links)
│   └── results/
│       ├── pseudo_manifolds.jls                               # Raw enumeration output
│       ├── pseudo_manifolds_autom_sorted_no_ghost.jls         # After step 1
│       └── TC_Seed_PLS.jls                                    # Final TCSeeds output
│
└── Picard_5/                         # Scripts for Picard number 5
    ├── create_mat_and_iso_DB_all_links.jl    # Build MatDB and isoDB (Alg. 3) [~30 min]
    ├── create_mat_and_iso_DB.jl              # Auxiliary / alternative DB builder
    ├── enumerate_pseudomanifolds_7-9.jl      # Enumerate for m ≤ 9 (Alg. 4)
    ├── enumerate_pseudomanifolds_10_each_l.jl # Enumerate for m = 10, one matroid at a time
    ├── reduce_autom_isom_PLS_7-9.jl          # Post-processing for m ≤ 9 [≥ 20 GB RAM]
    ├── step0_compile_10.jl                   # Merge m = 10 results into one file
    ├── step1_autom_reduction_10.jl           # Automorphism reduction for m = 10
    ├── step2_3_isom_seed_PLS_10.jl           # Isomorphism reduction + seed extraction
    ├── submit_10_enum_each_l.sh              # Slurm array job (indices 1–25, 27–46)
    ├── submit_10_enum_26.sh                  # Slurm job for matroid index 26
    ├── submit_step0.sh                       # Slurm job for step0_compile_10.jl
    ├── submit_step1.sh                       # Slurm job for step1_autom_reduction_10.jl
    ├── submit_step2_3.sh                     # Slurm job for step2_3_isom_seed_PLS_10.jl
    ├── resources/
    │   ├── mat_DB.jls                        # Serialized MatDB
    │   ├── iso_DB.jls                        # Serialized isoDB (restricted links)
    │   └── iso_DB_all.jls                    # Serialized isoDB (all links)
    └── results/
        ├── pseudo_manifolds_7-9.jls                            # Raw enumeration (m ≤ 9)
        ├── pseudo_manifolds_7-10.jls                           # Merged enumeration (m ≤ 10)
        ├── pseudo_manifolds_autom_sorted_no_ghost_7-9.jls      # After step 1 (m ≤ 9)
        ├── TC_seed_PLS_7-9.jls                                 # TCSeeds (m ≤ 9)
        └── TC_seed_PLS_7-10.jls                                # Final TCSeeds (m ≤ 10)
```

---

## File descriptions

### Root-level files

- **`simplicial_complex_utilities.jl`** — Utility functions for simplicial complexes encoded as `Vector{UInt}` bitmasks. Includes: boundary incidence matrices, mod $2$ homology / sphere tests, Euler characteristic, link computation, wedge-pair detection, seed extraction, Oscar interoperability, and an indexed isomorphism database.

- **`enumerate_kernel.jl`** — Parallel and sequential Gray code enumeration of the kernel of a Boolean matrix subject to row-sum constraints. This is the workhorse for enumerating pseudomanifolds; implements **Algorithms 1 and 2** of the paper.

---

### Folder `Picard_4`

All files here address the case of Picard number $4$ (cosimple binary matroids of corank $4$).

- **`RUN.jl`** — Runs the full pipeline by including the three scripts below in order. Launch with:
  ```bash
  julia --threads auto RUN.jl
  ```

- **`create_mat_and_iso_DB_all_links.jl`** — Constructs $\texttt{MatDB}$ (the database of cosimple binary matroids of corank $4$) and $\texttt{isoDB}$ (the isomorphism database of their links), implementing **Algorithm 3**. Saves `resources/mat_DB.jls` and `resources/iso_DB_all.jls`.

- **`enumerate_pseudomanifolds.jl`** — Dynamic programming enumeration of mod $2$ toric-colorable weak pseudomanifolds (**Algorithm 4**). Saves `results/pseudo_manifolds.jls`, corresponding to $\texttt{WkPsdMfd}$ in the paper.

- **`reduce_autom_isom_seed_PLS.jl`** — Post-processing pipeline (steps 1–3 of the paper):
  - **Step 1** (automorphism reduction): saves `results/pseudo_manifolds_autom_sorted_no_ghost.jls`.
  - **Steps 2–3** (isomorphism reduction + seed extraction): saves `results/TC_Seed_PLS.jls`, corresponding to $\texttt{TCSeeds}$ in the paper.

---

### Folder `Picard_5`

All files here address the case of Picard number $5$ (cosimple binary matroids of corank $5$). The larger search space requires HPC resources for the $m = 10$ case.

- **`create_mat_and_iso_DB_all_links.jl`** — Same role as its Picard_4 counterpart for corank $5$ matroids (**Algorithm 3**). Saves `resources/mat_DB.jls` and `resources/iso_DB_all.jls`.  
  > **Warning:** takes approximately 30 minutes.

- **`enumerate_pseudomanifolds_7-9.jl`** — **Algorithm 4** for $m_{\max} = 9$. Run from `Picard_5/` with:
  ```bash
  julia --threads auto enumerate_pseudomanifolds_7-9.jl
  ```
  Produces `results/pseudo_manifolds_7-9.jls`.

- **`reduce_autom_isom_PLS_7-9.jl`** — Post-processing for $m \le 9$. Saves `results/pseudo_manifolds_autom_sorted_no_ghost_7-9.jls` (step 1) and `results/TC_seed_PLS_7-9.jls` (steps 2–3).  
  > **Warning:** requires at least 20 GB of RAM.

- **`enumerate_pseudomanifolds_10_each_l.jl`** — **Algorithm 4** for $m = 10$, restricted to a single cosimple binary matroid of corank $5$ given by a command-line index $l \in \{1, \ldots, 46\}$. Requires `pseudo_manifolds_7-9.jls` and `TC_seed_PLS_7-9.jls` to be present. Produces `results/pseudo_manifolds_10_<l>.jls`.  
  > **Warning:** estimated to take **two months** on a single core. Intended to be run via Slurm (see below).

- **`step0_compile_10.jl`** — Merges `results/pseudo_manifolds_7-9.jls` with the 46 files `results/pseudo_manifolds_10_<l>.jls` into `results/pseudo_manifolds_7-10.jls`, corresponding to $\texttt{WkPsdMfd}$ with $m_{\max} = 10$.

- **`step1_autom_reduction_10.jl`** — Step 1 post-processing for $m \le 10$: reduces `pseudo_manifolds_7-10.jls` by the automorphism group of each matroid. Saves `results/pseudo_manifolds_auto_sorted_no_ghosts_7-10.jls`.

- **`step2_3_isom_seed_PLS_10.jl`** — Steps 2 and 3 of the post-processing for $m \le 10$. Saves the final `results/TC_seed_PLS_7-10.jls`, corresponding to $\texttt{TCSeeds}$ in the paper.

#### Slurm scripts (HPC)

The $m = 10$ enumeration is split across 46 independent jobs and must be run on a cluster. From `Picard_5/`:

| Script | Purpose |
|---|---|
| `submit_10_enum_each_l.sh` | Array job (indices 1–25, 27–46), 32 threads, 2-day wall time |
| `submit_10_enum_26.sh` | Dedicated job for the heaviest matroid (index 26) |
| `submit_step0.sh` | Runs `step0_compile_10.jl` |
| `submit_step1.sh` | Runs `step1_autom_reduction_10.jl` |
| `submit_step2_3.sh` | Runs `step2_3_isom_seed_PLS_10.jl` |

Launch order:
```bash
sbatch submit_10_enum_each_l.sh
sbatch submit_10_enum_26.sh
# wait for all array jobs to finish, then run one by one:
sbatch submit_step0.sh
sbatch submit_step1.sh
sbatch submit_step2_3.sh
```

---

## Loading serialized output files

All `.jls` files are written with Julia's `Serialization` standard library and are read back with the same pattern:

```julia
using Serialization
data = open("path/to/file.jls", "r") do io
    deserialize(io)
end
```

### Bitmask encoding convention

Simplicial complexes are stored as collections of unsigned integer bitmasks. Bit $i-1$ (0-indexed) being set means vertex $i$ (1-indexed) belongs to that simplex. For example, the bitmask `0b00010111 = UInt16(23)` encodes the face $\{1, 2, 3, 5\}$.

- **Picard_4** uses `UInt16` (up to 15 vertices).
- **Picard_5** uses `UInt32` (up to 18 vertices).

---

### `resources/mat_DB.jls`

**Type:** `Dict{Int, Vector{Vector{UInt16}}}` (Picard_4) / `Dict{Int, Vector{Vector{UInt32}}}` (Picard_5)

Maps each vertex count $m$ to the list of cosimple binary matroids of corank $r$ on $m$ elements. Each matroid is stored as the list of its bases, each basis encoded as a bitmask.

```julia
mat_DB = open("resources/mat_DB.jls", "r") do io deserialize(io) end

keys(mat_DB)          # available values of m, e.g. 5:15
length(mat_DB[8])     # number of cosimple binary matroids on 8 elements
mat_DB[8][1]          # bases of the first matroid on 8 elements: Vector{UInt16}

# Decode the first basis of the first matroid on 8 elements
basis_mask = mat_DB[8][1][1]
vertices_in_basis = [i for i in 1:16 if (basis_mask >> (i-1)) & 1 == 1]
```

---

### `resources/iso_DB_all.jls`

**Type:** `Dict{Int, Dict{Int, Vector{Tuple{Int, Any}}}}`

For each matroid $(m, k)$, stores one entry per vertex $v$ of the matroid: a pair `(index_contraction, perm)` where `index_contraction` is the index into `mat_DB[m-1]` of the deletion minor obtained by removing $v$, and `perm` is a `Dict{Int,Int}` relabeling the minor's vertices into matroid $k$'s vertex labels.

```julia
iso_DB = open("resources/iso_DB_all.jls", "r") do io deserialize(io) end

# For matroid k=1 at m=8: list of (index_contraction, perm) for each vertex
entries = iso_DB[8][1]   # Vector{Tuple{Int, Any}}
index_contraction, perm = entries[1]   # first vertex
# index_contraction: index in mat_DB[7] of the deletion minor
# perm: Dict{Int,Int} mapping mat_DB[7][index_contraction]'s labels → matroid 1's labels
```

---

### `results/pseudomanifolds.jls` (Picard_4) / `results/pseudomanifolds_7-9.jls`, `results/pseudomanifolds_7-10.jls` (Picard_5)

**Type:** `Dict{Int, Vector{Set{BitVector}}}`

Maps each $m$ to a list indexed by matroid index $l$. Each entry is the set of all toric-colorable weak pseudomanifolds found for matroid $l$ at size $m$, encoded as `BitVector` selectors over the list of cobases of that matroid.

```julia
pm_DB   = open("results/pseudomanifolds.jls", "r") do io deserialize(io) end
mat_DB  = open("resources/mat_DB.jls", "r") do io deserialize(io) end

m, l = 8, 1
bases   = mat_DB[m][l]
V       = reduce(|, bases)          # bitmask of all vertices
cobases = [b ⊻ V for b in bases]   # complementary cobases = potential facets

# Recover the facets of each pseudomanifold found for matroid (m, l)
for K_bit in pm_DB[m][l]
    facets = cobases[findall(K_bit)]  # Vector{UInt16}: bitmask of each facet
end

length(pm_DB[m][l])   # number of pseudomanifolds found for matroid (m=8, l=1)
```

---

### `results/pseudomanifolds_autom_sorted_no_ghost.jls` (Picard_4) / `…_7-9.jls`, `…_7-10.jls` (Picard_5)

**Type:** `Dict{Tuple{Int,Int}, Set{Vector{UInt16}}}` (Picard_4) / `…UInt32…` (Picard_5)

The automorphism-reduced database after Step 1 of the post-processing. Keys are `(d, nv)` pairs where `d` is the facet dimension and `nv` the number of vertices. Each element of the set is a sorted `Vector{UInt16}` (or `UInt32`) of facet bitmasks representing one pseudomanifold up to the matroid automorphism group.

```julia
db = open("results/pseudomanifolds_autom_sorted_no_ghost.jls", "r") do io deserialize(io) end

keys(db)                # all (d, nv) pairs present, e.g. (3, 8), (3, 9), ...
length(db[(3, 8)])      # number of pseudomanifolds of dimension 3 on 8 vertices

# Iterate over all complexes of dimension 3 on 8 vertices
for facets in db[(3, 8)]   # facets::Vector{UInt16}
    vertices = [i for i in 1:16 if any(f -> (f >> (i-1)) & 1 == 1, facets)]
end
```

---

### `results/TC_Seed_PLS.jls` (Picard_4) / `results/TC_seed_PLS_7-9.jls`, `results/TC_seed_PLS_7-10.jls` (Picard_5)

**Type:** `Dict{Tuple{Int,Int}, Set{Tuple{Vararg{UInt16}}}}` (Picard_4) / `…UInt32…` (Picard_5)

The final output: the set $\texttt{TCSeeds}$ of toric seeds. Keys are `(d, nv)` pairs. Each element of the set is a `Tuple` of facet bitmasks representing one TC seed (a mod $2$ PL sphere that is a seed of a smooth complete toric variety of the target Picard number).

```julia
tc = open("results/TC_Seed_PLS.jls", "r") do io deserialize(io) end

keys(tc)              # all (d, nv) pairs with at least one seed
length(tc[(3, 8)])    # number of TC seeds of dimension 3 on 8 vertices

# Inspect the facets of the first seed of dimension 3 on 8 vertices
seed = first(tc[(3, 8)])   # Tuple{Vararg{UInt16}}
for facet_mask in seed
    verts = [i for i in 1:16 if (facet_mask >> (i-1)) & 1 == 1]
    println(verts)
end

# Count total seeds across all sizes
total = sum(length(v) for v in values(tc))
```

---

## License

This work is released under [CC0 1.0 Universal](LICENSE) (public domain dedication).
