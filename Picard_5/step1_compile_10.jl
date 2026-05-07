using Serialization
using ProgressMeter
using Oscar

mat_DB_bin = open("resources/mat_DB.jls", "r") do io
    deserialize(io)
end

iso_DB = open("resources/iso_DB.jls", "r") do io
    deserialize(io)
end

pseudo_manifolds_DB = open("results/pseudo_manifolds_7-9.jls", "r") do io
    deserialize(io)
end

pseudo_manifolds_DB[10] = Vector{Set{BitVector}}[]

@showprogress desc="compiling all separate files" for k=1:46
    pseudo_manifolds_DB_k = open("results/pseudo_manifolds_10_part_$(k).jls", "r") do io
        deserialize(io)
    end
    selected_pseudomanifolds = Set{BitVector}()
    V = reduce(|, mat_DB_bin[10][k])
    compl_basis_vecs = [b ⊻ V for b in mat_DB_bin[10][k]]  
    for facets_bit in pseudo_manifolds_DB_k
        facets_bin = compl_basis_vecs[findall(facets_bit)]
        nv_K = count_ones(reduce(|, facets_bin))
        nv_K == 10 && push!(selected_pseudomanifolds, facets_bit)
    end

    push!(pseudo_manifolds_DB[10], selected_pseudomanifolds)
end

open("resources/pseudo_manifolds_7-10.jls", "w") do io
    serialize(io, pseudo_manifolds_DB)
end

println("Step 1 done. results/pseudo_manifolds_7-10.jls saved.")