include("../enumerate_kernel.jl")

function build_finalDB_single_v_one_l!(pseudo_manifolds_DB::Dict{Int,Vector{Set{BitVector}}},
                                       mat_DB::Dict{Int,Vector{Vector{UInt32}}},
                                       iso_DB::Dict{Int,Dict{Int,Vector{Tuple{Int,Any}}}},
                                       m, l)
    println("Processing m=$(m), l=$(l)")

    V_bin = reduce(|, mat_DB[m][l])
    bases_bin = mat_DB[m][l]
    compl_bases_bin = [base ⊻ V_bin for base in bases_bin]
    ridges, A = boundary_incidence_facets_to_ridges(compl_bases_bin)
    kernel_basis = kernel_basis_mod2_sparse(A)

    result = Set{BitVector}()

    for (index_contraction, perm) in iso_DB[m][l]
        links = collect(pseudo_manifolds_DB[m-1][index_contraction])

        @showprogress desc="l=$(l), links=$(length(links))" for L_bit in links
            mandatory_facets_bin = relabel(mat_DB[m-1][index_contraction][findall(L_bit)], perm)
            mandatory_facets_bit = subset_bitvector(bases_bin, mandatory_facets_bin)

            if count(mandatory_facets_bit) != length(mandatory_facets_bin)
                @warn "Some mandatory facets not found!" m l mandatory_facets_bin
            end

            all_solutions_bit = enumerate_kernel_with_constraints_bitvector(A, kernel_basis, mandatory_facets_bit)

            for K_bit in all_solutions_bit
                facets_bin = compl_bases_bin[findall(K_bit)]
                if euler_sphere_test(facets_bin)
                    push!(result, copy(K_bit))
                end
            end
        end
    end

    open("results/pseudo_manifolds_10_part_$(l).jls", "w") do io
        serialize(io, result)
    end

    println("Done: l=$(l), found $(length(result)) pseudo-manifolds.")
end

# --- Parse task index from CLI ---
l = parse(Int, ARGS[1])
m = 10

mat_DB = open("resources/mat_DB.jls", "r") do io deserialize(io) end
iso_DB = open("resources/iso_DB.jls", "r") do io deserialize(io) end
pseudo_manifolds_DB = open("results/pseudo_manifolds_7-9_all.jls", "r") do io deserialize(io) end

build_finalDB_single_v_one_l!(pseudo_manifolds_DB, mat_DB, iso_DB, m, l)