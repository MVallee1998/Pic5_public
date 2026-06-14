
include("../enumerate_kernel.jl")


function build_finalDB_single_v!(pseudo_manifolds_DB::Dict{Int,Vector{Set{BitVector}}},
                                  mat_DB::Dict{Int,Vector{Vector{UInt32}}},
                                  iso_DB::Dict{Int,Dict{Int,Vector{Tuple{Int,Any}}}},
                                  mmax; mstart=-1)
    mmin = minimum(collect(keys(mat_DB)))
    if mstart == -1
        mstart = mmin
    end

    for m = mstart:mmax
        pseudo_manifolds_DB[m] = Vector{Set{BitVector}}()
        n_matroids = length(mat_DB[m])

        prog_l = Progress(n_matroids, 0, "m=$m | matroid 0/$n_matroids | pm=0 ", 50, :cyan, stderr)

        for (l, bases_bin) in enumerate(mat_DB[m])
            V_bin = reduce(|, bases_bin)
            compl_bases_bin = [base ⊻ V_bin for base in bases_bin]
            ridges, A = boundary_incidence_facets_to_ridges(compl_bases_bin)
            kernel_basis = kernel_basis_mod2_sparse(A)
            push!(pseudo_manifolds_DB[m], Set{BitVector}())

            if m == mmin
                mandatory_facets_bit = falses(length(bases_bin))
                all_solutions_bit = enumerate_kernel_with_constraints_bitvector(A, kernel_basis, mandatory_facets_bit)
                for K_bit in all_solutions_bit
                    facets_bin = compl_bases_bin[findall(K_bit)]
                    if euler_sphere_test(facets_bin)
                        push!(pseudo_manifolds_DB[m][l], copy(K_bit))
                    end
                end
                n_pm = length(pseudo_manifolds_DB[m][l])
                next!(prog_l; desc="m=$m | matroid $l/$n_matroids | pm=$n_pm ")
            else
                iso_list = iso_DB[m][l]
                n_iso = length(iso_list)
                n_links_total = sum(length(pseudo_manifolds_DB[m-1][ic]) for (ic, _) in iso_list)

                n_links_done = 0
                for (iso_idx, (index_contraction, perm)) in enumerate(iso_list)
                    links = collect(pseudo_manifolds_DB[m-1][index_contraction])

                    for L_bit in links
                        mandatory_facets_bin = relabel(mat_DB[m-1][index_contraction][findall(L_bit)], perm)
                        mandatory_facets_bit = subset_bitvector(bases_bin, mandatory_facets_bin)

                        if count(mandatory_facets_bit) != length(mandatory_facets_bin)
                            @warn "Some mandatory facets not found in bases!" m l mandatory_facets_bin
                        end

                        all_solutions_bit = enumerate_kernel_with_constraints_bitvector(A, kernel_basis, mandatory_facets_bit)

                        for K_bit in all_solutions_bit
                            facets_bin = compl_bases_bin[findall(K_bit)]
                            if euler_sphere_test(facets_bin)
                                push!(pseudo_manifolds_DB[m][l], copy(K_bit))
                            end
                        end

                        n_links_done += 1
                        pct   = round(Int, 100 * n_links_done / max(1, n_links_total))
                        filled = round(Int, 50 * n_links_done / max(1, n_links_total))
                        bar   = "\e[33m" * "█"^filled * "\e[90m" * "░"^(50 - filled) * "\e[0m"
                        # move down 1, print inner bar, move back up 1
                        print(stderr, "\e[1B\e[2K\r  ↳ iso $iso_idx/$n_iso | link $n_links_done/$n_links_total ($pct%) |$bar|\e[1A")
                    end
                end
                # erase the inner bar line when this matroid is done
                print(stderr, "\e[1B\e[2K\e[1A")

                n_pm = length(pseudo_manifolds_DB[m][l])
                next!(prog_l; desc="m=$m | matroid $l/$n_matroids | pm=$n_pm ")
            end
        end
    end
end

mat_DB_bin = open("resources/mat_DB.jls", "r") do io
    deserialize(io)
end

iso_DB = open("resources/iso_DB_all.jls", "r") do io
    deserialize(io)
end

pseudo_manifolds_DB = Dict{Int,Vector{Set{BitVector}}}()

mmax=9


build_finalDB_single_v!(pseudo_manifolds_DB,mat_DB_bin,iso_DB,mmax)

# open("results/pseudo_manifolds_7-9_all.jls", "w") do io
#     serialize(io, pseudo_manifolds_DB)
# end