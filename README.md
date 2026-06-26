# Pic5_public

This is a companion reposotory containin the Julia scripts of <insert arxiv link here>.

## Julia packages required

`using Pkg; Pkg.add("Oscar", "Serialization", "Nemo", "Progressmeter")`

## Main folder

The folders and files are organized as follows:

- `simplicial_complex_utilities.jl` contains utilitary functions for simplicial complexes that are encoded as `Vector{UInt}`.

- `enumerate_kernel.jl` contains the Gray code enumeration algorithms (one parallel and one sequential, adapted for different kind of parallelization purpose), namely Algorithms 1 and 2 of the paper.

## Folder `Picard_4`
All the files in here are related to the case of Picard number $4$.

- `RUN.jl`: run it from the folder `Picard_4` with `julia --threads auto RUN.jl` for launching the whole computation.

- `create_mat_and_iso_DB_all_links.jl`: script for constructing $\texttt{MatDB}$ and $\texttt{isoDB}$, that is Algorithm 3 of the paper, both saved into the folder `resources/` as `mat_DB.jls` and `iso_DB_all.jls`, respectively.

- `enumerate_pseudomanifolds.jl`: script for the dynamic programming enumeration of the mod $2$ toric-colorable weak pseudomanifolds, as of Algorithm 4 of the paper, it saves the file `results/pseudomanifolds.jls` which corresponds to $\texttt{WkPsdMfd}$ output by Algorithm 4 for Picard number $4$ in the paper.

- `reduce_autom_isom_seed_PLS.jls`: script for the post-processing. It creates two files `results/pseudomanifolds_autom_sorted_no_ghost.jls` containing the weak pseudomanifolds obtained after Step 1 of the post-processing, and `results/TC_seed_PLS.jls` obtained at the final step of the post-processing.

## Folder `Picard_5`
All the files in here are related to the case of Picard number $5$.

- `create_mat_and_iso_DB_all_links.jl`: script for constructing $\texttt{MatDB}$ and $\texttt{isoDB}$, that is Algorithm 3 of the paper, both saved into the folder `resources/` as `mat_DB.jls` and `iso_DB_all.jls`, respectively.\
**WARNING:** this script takes around 30 minutes to finish.

- `enumerate_pseudomanifolds_7-9.jl`: script for the dynamic programming enumeration of the mod $2$ toric-colorable weak pseudomanifolds, as of Algorithm 4 of the paper, with $m_{\max}=9$. Run it from the folder `Picard_5` with `julia --threads auto enumerate_pseudomanifolds_7-9.jl` for better performances. This script creates the file `results/pseudomanifolds_7-9.jls`.

- `reduce_autom_isom_PLS_7-9.jls`: script for the post-processing. It creates two files `results/pseudomanifolds_autom_sorted_no_ghost_7-9.jls` containing the weak pseudomanifolds obtained after Step 1 of the post-processing, and `results/TC_seed_PLS_7-9.jls` obtained at the final step of the post-processing.\ 
**WARNING:** this script requires to have at least 20Go of memory.

- `enumerate_pseudomanifolds_10_each_l.jl`: script for running the case $m=10$ given an index among the $46$ cosimple binary matroid of corank $5$ on $10$ vertices, provided the scripts `enumerate_pseudomanifolds_7-9.jl` and `reduce_autom_isom_PLS_7-9.jls` have been run.\
This script is to be ran on a server that uses `slurm` by running `sbatch submit_10_enum_each_l.sh` and `sbatch submit_10_enum_26.sh` from the folder `Picard_5/`.
It produces the files `results/pseudomanifolds_18_<l>.jls` where $l=1,\ldots,46$.\
 **WARNING:** running this script on a single core computer is estimated to take **two months**.

- `step0_compile_10.jl`: compiles the file `results/pseudomanifolds_7-9.jls` with the $46$ files `results/pseudomanifolds_10_<l>.jls` into one `results/pseudomanifolds_7-10.jls` which corresponds to $\texttt{WkPsdMfd}$ output by Algorithm 4 with $m_{\max}=10$ and Picard number $5$ in the paper.\ 
This script is to be ran on a server that uses `slurm` by running `sbatch submit_step0.sh` from the folder `Picard_5/`.

- `step1_autom_reduction.jl`: reduces the database in `results/pseudomanifolds_7-10.jls` by the automorphism group of each cosimple binary matroid, that is step 1 of the post-processing in the paper.
It creates the file `results/pseudomanifolds_auto_sorted_no_ghosts_7-10.jls`\
This script is to be ran on a server that uses `slurm` by running `sbatch submit_step1.sh` from the folder `Picard_5/`.

- `step2_3_isom_seed_PLS.jl`: computes step 2 and 3 of the postprocessing in the paper.
It creates the final file `results/TC_seed_PLS_7-10.jls` which corresponds to `\texttt{TCSeeds}` in the paper.







