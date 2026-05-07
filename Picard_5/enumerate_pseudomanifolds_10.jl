include("../enumerate_kernel.jl")

function build_finalDB_single_v_one_m!(pseudo_manifolds_DB::Dict{Int,Vector{Set{BitVector}}},
                                  mat_DB::Dict{Int,Vector{Vector{UInt32}}},
                                  iso_DB::Dict{Int,Dict{Int,Vector{Tuple{Int,Any}}}},
                                  m)
    println(m)

    pseudo_manifolds_DB[m] = Vector{Set{BitVector}}()
    println(length(mat_DB[m]), " mat_DB entries for m=$(m). ")

    for (l, bases_bin) in enumerate(mat_DB[m])
        V_bin = reduce(|, bases_bin)
        compl_bases_bin = [base ⊻ V_bin for base in bases_bin]
        ridges, A = boundary_incidence_facets_to_ridges(compl_bases_bin)
        kernel_basis = kernel_basis_mod2_sparse(A)
        push!(pseudo_manifolds_DB[m], Set{BitVector}())

        for (index_contraction, perm) in iso_DB[m][l]
            links = collect(pseudo_manifolds_DB[m-1][index_contraction])
            
            @showprogress desc="Number of links $(length(links))" for L_bit in links
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
            end

            open("results/pseudo_manifolds_10_part_$(l).jls", "w") do io
                serialize(io, pseudo_manifolds_DB[m][l])
            end
            pseudo_manifolds_DB[m][l] = Set{BitVector}()  # free memory
        end
    end
end

mat_DB_bin = open("resources/mat_DB.jls", "r") do io
    deserialize(io)
end

iso_DB = open("resources/iso_DB.jls", "r") do io
    deserialize(io)
end

pseudo_manifolds_DB = open("results/pseudo_manifolds_7-9.jls", "r") do io
    deserialize(io)
end

build_finalDB_single_v_one_m!(pseudo_manifolds_DB,mat_DB_bin,iso_DB,10)

open("results/pseudo_manifolds_7-10.jls", "w") do io
    serialize(io, pseudo_manifolds_DB)
end