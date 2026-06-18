# Pic5_public

This is a companion reposotory containin the Julia scripts of <insert arxiv link here>.

## Julia packages required

`using Pkg; Pkg.add("Oscar", "Serialization", "Nemo", "Progressmeter")`

## Main folder

The folders and files are organized as follows:

- `simplicial_complex_utilities.jl` contains utilitary functions for simplicial complexes which are encoded as `Vector{UInt}`.

- `enumerate_kernel.jl` contains the Gray code enumeration algorithms (one parallel and one sequential, adapted for different kind of parallelization purpose)

## Folder `Picard_4`

- 