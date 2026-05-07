using Oscar
using IterTools
using LinearAlgebra
using ProgressMeter
using Serialization
using SparseArrays

using Polymake
const topaz = Polymake.topaz

using Nemo
F = GF(2)


# -------------------------------------------------------
# mat_DB generation
# -------------------------------------------------------

function all_nonzero_binary_vectors(n)
    k = (2^n) - 1
    mat = Matrix{Int}(undef, k, n)
    for i in 1:k
        v = digits(i, base=2, pad=n)
        mat[i, :] = v
    end
    return mat
end

function find_lower_dim_matroids(list_bin_mat)
    local S_rep = Set{Matroid}()
    @showprogress for M in list_bin_mat
        for v in matroid_groundset(M)
            if v in coloops(M)
                continue
            end
            M1 = deletion(M, v)
            is_isom = false
            for M2 in S_rep
                if is_isomorphic(M1, M2)
                    is_isom = true
                    break
                end
            end
            if is_isom == false
                push!(S_rep, M1)
            end
        end
    end
    return S_rep
end

pic = 5

A  = matrix(GF(2), all_nonzero_binary_vectors(pic))
M0 = matroid_from_matrix_rows(A)

global S = Set{Matroid}()
push!(S, M0)

simple_bin_matroids     = Dict{Int, Vector{Vector{Vector{UInt8}}}}()
simple_bin_matroids_bin = Dict{Int, Vector{Vector{UInt32}}}()

function make_binary(L::Vector{Vector{UInt8}})
    result = Vector{UInt32}()
    for elt in L
        push!(result, reduce(|, [UInt32(1) << (i - 1) for i in elt]))
    end
    return sort(result)
end

global m = size(A)[1]   # = 2^5 - 1 = 31

while m > pic
    print("number of elements ", m, " number of matroids ", length(S), "\n")
    simple_bin_matroids[m]     = [bases(M) for M in S]
    simple_bin_matroids_bin[m] = [make_binary(Vector{Vector{UInt8}}(bases_M))
                                   for bases_M in simple_bin_matroids[m]]
    global S  = find_lower_dim_matroids(S)
    global m -= 1
end

open("resources/mat_DB.jls", "w") do io
    serialize(io, simple_bin_matroids_bin)
end



# -------------------------------------------------------
# Helper functions
# -------------------------------------------------------

function vertex_list_of_bases(bases::Vector{Vector{UInt8}})
    s = Set{UInt8}()
    for b in bases
        for v in b
            push!(s, v)
        end
    end
    return sort(collect(s))
end

function bases_to_topaz_complex(bases::Vector{Vector{UInt8}})
    verts = sort!(unique(vcat(bases...)))
    vmap  = Dict(v => i - 1 for (i, v) in enumerate(verts))
    facets = [[vmap[x] for x in B] for B in bases]
    return topaz.SimplicialComplex(FACETS = facets), verts
end

function topaz_isomorphism(basesA, basesB)
    SC_A, vertsA = bases_to_topaz_complex(basesA)
    SC_B, vertsB = bases_to_topaz_complex(basesB)
    length(vertsA) == length(vertsB) || return nothing
    T = topaz.find_facet_vertex_permutations(SC_B, SC_A)
    T === nothing && return nothing
    _, p = T
    return Dict(vertsA[i] => vertsB[p[i] + 1] for i in eachindex(vertsA))
end


# -------------------------------------------------------
# iso_DB generation
# -------------------------------------------------------

"""
build_iso_db!(Iso_DB, mat_DB; ms, verbose)

For each level m in ms and each matroid k in mat_DB[m], finds the first
non-coloop vertex v such that deletion(M, v) is isomorphic to some entry
mat_DB[m-1][l], and stores (l, perm) in Iso_DB[m][k].
"""
function build_iso_db!(
    Iso_DB::Dict{Int, Dict{Int, Vector{Tuple{Int, Any}}}},
    mat_DB::Dict{Int, Vector{Vector{UInt32}}};
    ms      = nothing,
    verbose = false
)
    for m in ms
        println("Processing m = ", m)
        if !haskey(mat_DB, m - 1)
            verbose && @info "Skipping m=$m: mat_DB[$(m-1)] not present"
            continue
        end

        Iso_DB[m] = Dict{Int, Vector{Tuple{Int, Any}}}()

        @showprogress for (k, Mbases_bin) in enumerate(mat_DB[m])

            # Decode binary representation (UInt32, bits 1..32)
            Mbases = [[UInt8(i) for i in 1:32 if ((b >> (i - 1)) & UInt32(1)) == 1]
                      for b in Mbases_bin]
            V = vertex_list_of_bases(Mbases)
            M = matroid_from_bases(Mbases, V)
            cl = coloops(M)

            Iso_DB[m][k] = Vector{Tuple{Int, Any}}()

            # Iterate non-coloop vertices until one resolves to a parent in mat_DB[m-1]
            for v in V
                v in cl && continue

                Mv             = deletion(M, v)
                deletion_bases = bases(Mv)
                found_index    = -1
                perm           = nothing

                for (l, target_bin) in enumerate(mat_DB[m - 1])
                    target_bases = [[UInt8(i) for i in 1:32 if ((b >> (i - 1)) & UInt32(1)) == 1]
                                    for b in target_bin]
                    M2 = matroid_from_bases(target_bases, vertex_list_of_bases(target_bases))
                    is_isomorphic(Mv, M2) || continue

                    perm = topaz_isomorphism(target_bases, deletion_bases)
                    perm === nothing && continue

                    # Verify permutation
                    relabeled  = Set([sort([perm[x] for x in b]) for b in target_bases])
                    del_sorted = Set([sort(collect(b)) for b in deletion_bases])
                    if relabeled != del_sorted
                        verbose && @warn "Perm verification failed m=$m k=$k v=$v — trying next"
                        perm = nothing
                        continue
                    end

                    found_index = l
                    break
                end

                if found_index != -1
                    push!(Iso_DB[m][k], (found_index, perm))
                    break   # one valid parent edge is sufficient
                end
            end

            if isempty(Iso_DB[m][k])
                @warn "m=$m, k=$k: no parent found — mat_DB may be incomplete"
            end
        end
    end
    return Iso_DB
end

iso_DB = Dict{Int, Dict{Int, Vector{Tuple{Int, Any}}}}()

build_iso_db!(iso_DB, simple_bin_matroids_bin, ms = 7:10, verbose = true)

open("resources/iso_DB.jls", "w") do io
    serialize(io, iso_DB)
end