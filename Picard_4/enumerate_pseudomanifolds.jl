include("../enumerate_kernel.jl")

mmax=15

mat_DB_bin = open("resources/mat_DB.jls", "r") do io
    deserialize(io)
end

iso_DB = open("resources/iso_DB.jls", "r") do io
    deserialize(io)
end

function build_finalDB_single_v!(pseudo_manifolds_DB::Dict{Int,Vector{Set{BitVector}}},mat_DB::Dict{Int,Vector{Vector{UInt16}}},iso_DB::Dict{Int,Dict{Int,Vector{Tuple{Int,Any}}}},mmax;mstart=-1)
    mmin = minimum(collect(keys(mat_DB)))
    if mstart == -1
        mstart = mmin
    end
    for m=mstart:mmax
        println(m)
        pseudo_manifolds_DB[m] = Vector{Set{BitVector}}()
        for (l,bases_bin) in enumerate(mat_DB[m])
            # display(bases)
            V_bin = reduce(|,bases_bin)
            compl_bases_bin = [base⊻V_bin for base in bases_bin] # we need to complement to get the correct boundary matrix
            ridges, A = boundary_incidence_facets_to_ridges(compl_bases_bin)  
            kernel_basis = kernel_basis_mod2_sparse(A)
            push!(pseudo_manifolds_DB[m], Set{BitVector}())
            if m==mmin # Base case
                mandatory_facets_bit=falses(length(bases_bin))
                all_solutions_bit = enumerate_kernel_with_constraints_bitvector(A,kernel_basis,mandatory_facets_bit)
                for K_bit in all_solutions_bit
                    facets_bin = [facet_bin for facet_bin in compl_bases_bin[findall(K_bit)]]
                    if euler_sphere_test(facets_bin)
                        push!(pseudo_manifolds_DB[m][l],copy(K_bit))
                    end
                end
            else
                for (index_contraction, perm) in iso_DB[m][l] # find by computing the links
                    @showprogress desc="Number of links $(length(pseudo_manifolds_DB[m-1][index_contraction]))" for L_bit in pseudo_manifolds_DB[m-1][index_contraction]
                        mandatory_facets_bin = relabel( mat_DB[m-1][index_contraction][findall(L_bit)], perm) # for every potential link
                        mandatory_facets_bit = subset_bitvector(bases_bin, mandatory_facets_bin)
                        if count(mandatory_facets_bit) != length(mandatory_facets_bin)
                            @warn "Some mandatory facets not found in bases!" m l mandatory_facets_bin
                        end
                        all_solutions_bit = enumerate_kernel_with_constraints_bitvector(A,kernel_basis,mandatory_facets_bit)
                        for K_bit in all_solutions_bit
                            if m<=mmax*0.75 # only do the euler test for smaller m, to save time
                                facets_bin = [facet_bin for facet_bin in compl_bases_bin[findall(K_bit)]]
                                if euler_sphere_test(facets_bin)
                                    push!(pseudo_manifolds_DB[m][l],copy(K_bit))
                                end
                            else
                                push!(pseudo_manifolds_DB[m][l],copy(K_bit))
                            end
                        end
                    end
                end
            end
        end
    end
end

pseudo_manifolds_DB = Dict{Int,Vector{Set{BitVector}}}()


build_finalDB_single_v!(pseudo_manifolds_DB,mat_DB_bin,iso_DB,mmax)

open("results/pseudo_manifolds.jls", "w") do io
    serialize(io, pseudo_manifolds_DB)
end


