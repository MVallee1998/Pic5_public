include("../simplicial_complex_utilities.jl")
using Base.Threads
# ── Helpers ───────────────────────────────────────────────────────────────────

function convert_dict_uint16_to_uint32(
    d::Dict{Tuple{Int,Int}, Set{Tuple{Vararg{UInt16}}}}
)::Dict{Tuple{Int,Int}, Set{Tuple{Vararg{UInt32}}}}
    Dict(key => Set(Tuple(UInt32.(t)) for t in val) for (key, val) in d)
end

# ── Load data ─────────────────────────────────────────────────────────────────

mat_DB_bin = open("resources/mat_DB.jls", "r") do io
    deserialize(io)
end

pseudo_manifolds_DB = open("results/pseudo_manifolds_7-9.jls", "r") do io
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

            V_bin      = reduce(|, bases)
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

            index  = Dict(facets_internal[i] => i for i in eachindex(facets_internal))
            G      = automorphism_group(M)
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

number_before_automorphisms_each_m = [sum(length.(pseudo_manifolds_DB[m])) for m in 7:9]
println("Number before automorphisms: ", number_before_automorphisms_each_m)


database_reduce_autom = reduce_by_automorphisms(pseudo_manifolds_DB, mat_DB_bin, 7:9)

# ── Build database_before_iso ─────────────────────────────────────────────────

database_before_iso = Dict{Tuple{Int,Int}, Set{Vector{UInt32}}}()

for m in 7:9
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

open("results/pseudo_manifolds_autom_sorted_no_ghost_7-9.jls", "w") do io
    serialize(io, database_before_iso)
end

number_before_pre_filters_each_m = [length(database_before_iso[(m - 5 - 1, m)]) for m in 7:9]
println("Number before pre-filters: ", number_before_pre_filters_each_m)

# ── Seed database initialization ──────────────────────────────────────────────

database_tc_seed_PLS = Dict{Tuple{Int,Int}, Set{Tuple{Vararg{UInt32}}}}()

database_tc_seed_PLS[(0, 2)] = Set([(UInt32(1), UInt32(2))])
database_tc_seed_PLS[(3, 8)] = Set([index_to_bin(vec([[x...] for x in Iterators.product(1:2, 3:4, 5:6, 7:8)]),UInt32)])
database_tc_seed_PLS[(2, 6)] = Set([index_to_bin(vec([[x...] for x in Iterators.product(1:2, 3:4, 5:6)]),UInt32)])

const database_tc_seed_index = Dict{Tuple{Int,Int},
    Dict{Tuple{Vector{Int},Vector{Int}}, Vector{Tuple{Vararg{UInt16}}}}}()

for (k, v) in database_tc_seed_PLS
    database_tc_seed_index[k] = build_index(v, UInt16)
end


# ── Main loop ─────────────────────────────────────────────────────────────────

number_before_PL_sphere_checks_each_m = zeros(Int, 3)

number_seeds_each_m = zeros(Int, 3)

for m in 3:9
    for Pic in 1:5
        key_in = (m - Pic - 1, m)
        haskey(database_before_iso, key_in) || continue
        items   = collect(database_before_iso[key_in])
        db_seed = get!(database_tc_seed_PLS, key_in, Set{Tuple{Vararg{UInt32}}}())
        db_index = get!(database_tc_seed_index, key_in,
                        build_index(db_seed, UInt32))   # shared reference, mutated in-place
                        
        # Phase 1 : parallel filters
        prog = Progress(length(items); desc="Filters (m=$m, Pic=$Pic): ")

        if Pic == 5 && m>=7
            number_before_PL_sphere_checks_each_m[m-6] = length(items)
        end

        candidates = let
            local_cands = [Vector{Tuple{Vararg{UInt32}}}() for _ in 1:length(items)]
            @threads :dynamic for i in eachindex(items)
                facets_bin = items[i]
                next!(prog; showvalues = [(:seeds, length(db_seed)), (:Pic, Pic), (:m, m)])
                is_seed_bit(facets_bin)    || continue
                is_mod2_sphere(facets_bin) || continue
                push!(local_cands[i], Tuple(facets_bin))
            end
            reduce(vcat, local_cands)
        end

        # Phase 2 : Oscar verifications (sequential)
        db_index = build_index(db_seed,UInt32)
        prog2 = Progress(length(candidates); desc="Iso checks (m=$m, Pic=$Pic): ")

        for facets_bin in candidates
            verts = vertices_from_mask(vertex_mask(facets_bin))

            all_links_ok = all(verts) do v
                Lk    = find_seed_bit(link_facets(facets_bin, v))
                isempty(Lk) && return false
                key_L = (facet_dim(Lk[1]), count_ones(vertex_mask(Lk)))
                idx_L = get(database_tc_seed_index, key_L, nothing)
                isnothing(idx_L) && return false
                is_isomorphic_to_any_indexed(Lk, idx_L)
            end

            if all_links_ok && !is_isomorphic_to_any_indexed(facets_bin, db_index)
                push_indexed!(db_seed, db_index, facets_bin)
                # db_index IS database_tc_seed_index[key_in], so nothing else to update
            end
            next!(prog2; showvalues = [(:candidates, length(candidates)),
                                       (:seeds,      length(db_seed)),
                                       (:buckets,    length(db_index))])
        end

        if Pic == 5 && m>=7
            number_seeds_each_m[m-6] = length(db_seed)
        end
    end
end

# ── Save ──────────────────────────────────────────────────────────────────────

open("results/TC_seed_PLS_7-9.jls", "w") do io
    serialize(io, database_tc_seed_PLS)
end