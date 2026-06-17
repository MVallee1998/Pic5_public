include("../enumerate_kernel.jl")

mmax = 15

mat_DB_bin = open("resources/mat_DB.jls", "r") do io deserialize(io) end
iso_DB     = open("resources/iso_DB_all.jls", "r") do io deserialize(io) end

function build_finalDB_single_v!(pseudo_manifolds_DB::Dict{Int,Vector{Set{BitVector}}},
                                  mat_DB::Dict{Int,Vector{Vector{UInt16}}},
                                  iso_DB::Dict{Int,Dict{Int,Vector{Tuple{Int,Any}}}},
                                  mmax; mstart=-1, last_single_link=false)
    mmin = minimum(collect(keys(mat_DB)))
    mstart == -1 && (mstart = mmin)

    for m = mstart:mmax
        println(m)
        pseudo_manifolds_DB[m] = Vector{Set{BitVector}}()

        for (l, bases_bin) in enumerate(mat_DB[m])
            V_bin           = reduce(|, bases_bin)
            compl_bases_bin = [base ⊻ V_bin for base in bases_bin]
            ridges, A       = boundary_incidence_facets_to_ridges(compl_bases_bin)
            kernel_basis    = kernel_basis_mod2_sparse(A)
            rows            = sparse_rows(A)    # precompute once per matroid
            push!(pseudo_manifolds_DB[m], Set{BitVector}())

            if m == mmin
                mandatory_facets_bit = falses(length(bases_bin))
                state = prepare_kernel_enumeration(A, kernel_basis, mandatory_facets_bit, rows)
                if state !== nothing
                    for K_bit in enumerate_from_prepared(state)
                        facets_bin = compl_bases_bin[findall(K_bit)]
                        euler_sphere_test(facets_bin) && push!(pseudo_manifolds_DB[m][l], copy(K_bit))
                    end
                end
            else
                # Precompute only 

                for (index_contraction, perm) in iso_DB[m][l]
                    @showprogress desc="links=$(length(pseudo_manifolds_DB[m-1][index_contraction]))" for L_bit in pseudo_manifolds_DB[m-1][index_contraction]
                        mandatory_facets_bin = relabel(mat_DB[m-1][index_contraction][findall(L_bit)], perm)
                        mandatory_facets_bit = subset_bitvector(bases_bin, mandatory_facets_bin)

                        if count(mandatory_facets_bit) != length(mandatory_facets_bin)
                            @warn "Some mandatory facets not found!" m l mandatory_facets_bin
                        end

                        state = prepare_kernel_enumeration(A, kernel_basis, mandatory_facets_bit, rows)
                        state === nothing && continue
                        for K_bit in enumerate_from_prepared_parallel(state)
                            if m <= mmax * 0.75
                                facets_bin = compl_bases_bin[findall(K_bit)]
                                euler_sphere_test(facets_bin) && push!(pseudo_manifolds_DB[m][l], copy(K_bit))
                            else
                                push!(pseudo_manifolds_DB[m][l], copy(K_bit))
                            end
                        end
                    end

                    m == mmax && last_single_link && break
                end
            end
        end
    end
end

pseudo_manifolds_DB = Dict{Int,Vector{Set{BitVector}}}()
build_finalDB_single_v!(pseudo_manifolds_DB, mat_DB_bin, iso_DB, mmax; last_single_link=true)

open("results/pseudo_manifolds_all.jls", "w") do io
    serialize(io, pseudo_manifolds_DB)
end