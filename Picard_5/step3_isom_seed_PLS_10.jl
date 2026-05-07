include("../simplicial_complex_utilities.jl")

mmax = 10

# ── Load data ─────────────────────────────────────────────────────────────────

mat_DB_bin = open("resources/mat_DB.jls", "r") do io
    deserialize(io)
end

database_before_iso = open("results/pseudo_manifolds_autom_sorted_no_ghost_7-10.jls", "r") do io
    deserialize(io)
end

database_tc_seed_PLS = open("results/TC_seed_PLS_7-9.jls", "r") do io
    deserialize(io)
end

# ── Main loop ─────────────────────────────────────────────────────────────────

# Build once, outside the m loop
pls_indices = Dict{Any, Any}()
for (key, db) in database_tc_seed_PLS
    pls_indices[key] = build_index(db)
end

struct LinkKey
    facets::Tuple{Vararg{UInt32}}
    vertex::UInt32
end

const link_cache = Dict{LinkKey, Union{Nothing, Tuple{Vararg{UInt32}}}}()

function cached_link_seed(facets_bin::Facets, v::UInt32)
    key = LinkKey(Tuple(sort!(collect(facets_bin))), v)
    get!(link_cache, key) do
        Lk = find_seed_bit(link_facets(facets_bin, v))
        isempty(Lk) ? nothing : Tuple(Lk)
    end
end

candidates_all = Dict{Any, Vector{Tuple{Vararg{UInt32}}}}()

Pic=5
key_in = (m - Pic - 1, m)
haskey(database_before_iso, key_in)
items   = collect(database_before_iso[key_in])
db_seed = get!(database_tc_seed_PLS, key_in, Set{Tuple{Vararg{UInt32}}}())

# Phase 1 : filtres parallèles (pur Julia)
prog = Progress(length(items); desc="Filters (m=$m, Pic=$Pic): ")

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

candidates_all[key_in] = candidates

open("results/candidates_7-10.jls", "w") do io
    serialize(io, candidates_all)
end

haskey(candidates_all, key_in)
candidates = candidates_all[key_in]

db_seed  = get!(database_tc_seed_PLS, key_in, Set{Tuple{Vararg{UInt32}}}())  # ← re-fetch
db_index = build_index(db_seed)
prog2 = Progress(length(candidates); desc="Iso checks (m=$m, Pic=$Pic): ")
for facets_bin in candidates
    next!(prog2; showvalues = [(:candidates, length(candidates)),
                                (:seeds,      length(db_seed)),
                                (:buckets,    length(db_index))])

    verts = vertices_from_mask(vertex_mask(facets_bin))

    all_links_ok = all(verts) do v
        Lk = cached_link_seed(facets_bin, v)
        isnothing(Lk) && return false
        key_L = (facet_dim(Lk[1]), count_ones(vertex_mask(Lk)))
        haskey(pls_indices, key_L) &&
            is_isomorphic_to_any_indexed(Lk, pls_indices[key_L])
    end

    if all_links_ok && !is_isomorphic_to_any_indexed(facets_bin, db_index)
        push_indexed!(db_seed, db_index, facets_bin)
        pls_indices[key_in] = db_index  # keep in sync
    end
end

println("Seed count Pic=$Pic m=$m: ", length(db_seed))

# ── Save ──────────────────────────────────────────────────────────────────────

open("results/TC_seed_PLS_7-10.jls", "w") do io
    serialize(io, database_tc_seed_PLS)
end

println("Step 3 done. Seeds saved to results/Pic_5_tc_seed_PLS_7-10.jls")