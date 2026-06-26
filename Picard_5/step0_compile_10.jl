using Serialization
using ProgressMeter
using Oscar

mat_DB_bin = open("resources/mat_DB.jls", "r") do io
    deserialize(io)
end

pseudomanifolds_DB = open("results/pseudomanifolds_7-9.jls", "r") do io
    deserialize(io)
end

pseudomanifolds_DB[10] = Vector{Set{BitVector}}[]

@showprogress desc="compiling all separate files" for k=1:46
    pseudomanifolds_DB_k = open("results/pseudomanifolds_10_part_$(k).jls", "r") do io
        deserialize(io)
    end
    selected_pseudomanifolds = Set{BitVector}()
    V = reduce(|, mat_DB_bin[10][k])
    compl_basis_vecs = [b ⊻ V for b in mat_DB_bin[10][k]]  
    for facets_bit in pseudomanifolds_DB_k
        facets_bin = compl_basis_vecs[findall(facets_bit)]
        nv_K = count_ones(reduce(|, facets_bin))
        push!(selected_pseudomanifolds, facets_bit)
    end

    push!(pseudomanifolds_DB[10], selected_pseudomanifolds)
end

open("results/pseudomanifolds_7-10.jls", "w") do io
    serialize(io, pseudomanifolds_DB)
end

number_before_automorphisms_each_m = [sum(length.(pseudomanifolds_DB[m])) for m in 7:10]
println("Number before automorphisms: ", number_before_automorphisms_each_m)

println("Step 1 done. results/pseudomanifolds_7-10.jls saved.")