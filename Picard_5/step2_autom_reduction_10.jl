include("../simplicial_complex_utilities.jl")


mmax = 10

# ── Load data ─────────────────────────────────────────────────────────────────

mat_DB_bin = open("resources/mat_DB.jls", "r") do io
    deserialize(io)
end

pseudo_manifolds_DB = open("results/pseudo_manifolds_7-10.jls", "r") do io
    deserialize(io)
end

# ── Automorphism reduction ────────────────────────────────────────────────────

function reduce_by_automorphisms(
    pseudo_manifolds_DB::Dict{Int, Vector{Set{BitVector}}},
    mat_DB_bin::Dict{Int, Vector{Vector{UInt32}}},
    ms::UnitRange{Int}
)::Dict{Int, Vector{Set{BitVector}}}

    result = Dict{Int, Vector{Set{BitVector}}}()

    for m in ms
        result[m] = Vector{Set{BitVector}}()

        for (l, bases) in enumerate(mat_DB_bin[m])
            push!(result[m], Set{BitVector}())

            V_bin       = reduce(|, bases)
            compl_bases = [base ⊻ V_bin for base in bases]

            facets_M = [
                [i for i in 1:(8 * sizeof(cobase)) if (cobase >> (i-1)) & 1 == 1]
                for cobase in compl_bases
            ]

            M               = simplicial_complex(facets_M)
            faces_list      = collect(facets(M))
            facets_internal = Vector{UInt32}(undef, length(faces_list))

            for j in eachindex(faces_list)
                mask = UInt32(0)
                for v in faces_list[j]
                    mask |= UInt32(1) << (v - 1)
                end
                facets_internal[j] = mask
            end

            index     = Dict(facets_internal[i] => i for i in eachindex(facets_internal))
            G         = automorphism_group(M)
            all_autos = collect(elements(G))

            @inline function permute_facet(mask::UInt32, g)
                h = UInt32(0)
                x = mask
                while x != 0
                    v  = trailing_zeros(x) + 1
                    h |= UInt32(1) << (g(v) - 1)
                    x &= x - 1
                end
                return h
            end

            sigmas = map(all_autos) do g
                map(eachindex(facets_internal)) do j
                    index[permute_facet(facets_internal[j], g)]
                end
            end

            function canonical_rep(χ::BitVector)
                best = χ
                for σ in sigmas
                    χ2 = falses(length(χ))
                    @inbounds for ii in eachindex(χ)
                        χ[ii] && (χ2[σ[ii]] = true)
                    end
                    χ2 < best && (best = χ2)
                end
                return best
            end

            @showprogress desc="Automorphisms (m=$m, l=$l): " for χ in pseudo_manifolds_DB[m][l]
                push!(result[m][l], canonical_rep(χ))
            end
        end
    end

    return result
end

database_reduce_autom = reduce_by_automorphisms(pseudo_manifolds_DB, mat_DB_bin, mmax:mmax)

# ── Build database_before_iso ─────────────────────────────────────────────────

database_before_iso = open("results/pseudo_manifolds_autom_sorted_no_ghost_7-9.jls", "r") do io
    deserialize(io)
end

for m in mmax:mmax
    for (l, bases) in enumerate(mat_DB_bin[m])
        V           = reduce(|, bases)
        compl_bases = [base ⊻ V for base in bases]

        @showprogress desc="Building DB (m=$m): " for facets_bit in database_reduce_autom[m][l]
            facets_bin = compl_bases[findall(facets_bit)]
            nv_K = count_ones(reduce(|, facets_bin))
            d_K  = count_ones(facets_bin[1]) - 1
            db = get!(database_before_iso, (d_K, nv_K), Set{Vector{UInt32}}())
            push!(db, copy(sort(facets_bin)))
        end
    end
end

open("results/pseudo_manifolds_autom_sorted_no_ghost_7-10.jls", "w") do io
    serialize(io, database_before_iso)
end

number_before_pre_filters_each_m = [length(database_before_iso[(m - 5 - 1, m)]) for m in 7:10]
println("Number before pre-filters: ", number_before_pre_filters_each_m)

println("Step 2 done. database_before_iso saved to results/pseudo_manifolds_autom_sorted_no_ghost_7-10.jls")