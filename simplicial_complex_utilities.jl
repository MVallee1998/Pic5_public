using Serialization
using Oscar
using ProgressMeter
using SparseArrays
using Serialization
using LinearAlgebra
using Nemo

const F2 = GF(2)
# ─────────────────────────────────────────────────────────────────────────────
# § Bit utilities
# ─────────────────────────────────────────────────────────────────────────────

"""Iterator over 0-based positions of set bits in `x::Unsigned`."""
set_bit_positions(x::U) where {U<:Unsigned} =
    (i for i in 0:(8 * sizeof(x) - 1) if (x >> i) & one(x) == one(x))

    
function boundary_incidence_facets_to_ridges(facets::Vector{U})  where {U<:Unsigned}
    ridge_dict = Dict{U,Int}()
    ridges     = U[]

    for f in facets, i in set_bit_positions(f)
        r = f ⊻ (one(U) << i)
        if !haskey(ridge_dict, r)
            push!(ridges, r)
            ridge_dict[r] = length(ridges)
        end
    end

    m, n = length(ridges), length(facets)
    I, J = Int[], Int[]

    for (j, f) in pairs(facets), i in set_bit_positions(f)
        push!(I, ridge_dict[f ⊻ (one(U) << i)])
        push!(J, j)
    end

    return ridges, sparse(I, J, true, m, n)
end



function relabel(facets_bin::Vector{U},perm) where {U<:Unsigned}
    rel_facets_bin = Vector{U}()
    for facet_bin in facets_bin
        rel_facet_bin = zero(U)
        for (i,j) in perm
            if (facet_bin>>(i-1))&1==1
                rel_facet_bin |= one(U)<<(j-1)
            end
        end
        push!(rel_facets_bin,copy(rel_facet_bin))
    end
    return rel_facets_bin
end

function subset_bitvector(superset::Vector{U}, subset::Vector{U}) where {U<:Unsigned}
    S = Set(subset)
    return BitVector(x in S for x in superset)
end

function euler_characteristic_sphere(top_facets::Vector{U}) where {U<:Unsigned}
    isempty(top_facets) && return 0

    d = count_ones(top_facets[1]) - 1

    # faces_by_dim[i] will store i-faces
    faces_by_dim = Vector{Set{U}}(undef, d+1)

    # Top dimension
    faces_by_dim[d+1] = Set(top_facets)

    # Generate all lower-dimensional faces
    for dim = d:-1:1
        current_faces = faces_by_dim[dim+1]
        lower_faces = Set{U}()

        for f in current_faces
            x = f
            while x != 0
                i = trailing_zeros(x)
                push!(lower_faces, f ⊻ (one(U) << i))
                x &= x - 1
            end
        end

        faces_by_dim[dim] = lower_faces
    end

    # Compute χ = Σ (-1)^i f_i
    χ = 0
    for i in 0:d
        χ += (-1)^i * length(faces_by_dim[i+1])
    end

    return χ
end

function euler_sphere_test(top_facets::Vector{U}) where {U<:Unsigned}
    isempty(top_facets) && return false
    d = count_ones(top_facets[1]) - 1
    χ = euler_characteristic_sphere(top_facets)
    return χ == 1 + (-1)^d
end

function mod2_rank_nemo(A::SparseMatrixCSC)
    m, n = size(A)
    if m == 0 || n == 0
        return 0
    end

    M = zero_matrix(F2, m, n)

    # Fill matrix directly from sparse structure
    for col in 1:n
        for idx in A.colptr[col]:(A.colptr[col+1]-1)
            row = A.rowval[idx]
            M[row, col] = F2(1)
        end
    end

    return rank(M)
end


function is_mod2_sphere(top_facets::Vector{U}) where {U<:Unsigned}

    isempty(top_facets) && return true

    d = count_ones(top_facets[1]) - 1

    current_faces = top_facets

    # ---- Step 1: Top homology β_d ----
    # β_d = (#d-faces - rank ∂_d)
    ridges, B = boundary_incidence_facets_to_ridges(current_faces)
    rank_prev = mod2_rank_nemo(B)

    n_d = length(current_faces)
    β_d = n_d - rank_prev
    β_d == 1 || return false

    current_faces = ridges

    # ---- Step 2: Middle dimensions ----
    # For i = d-1 down to 1:
    for dim = d-1:-1:1

        ridges, B = boundary_incidence_facets_to_ridges(current_faces)
        rank_curr = mod2_rank_nemo(B)

        n_i = length(current_faces)

        # β_i = (#i-faces - rank ∂_i) - rank ∂_{i+1}
        β_i = (n_i - rank_curr) - rank_prev
        β_i == 0 || return false

        rank_prev = rank_curr
        current_faces = ridges
    end

    # ---- Step 3: β_0 ----
    # β_0 = (#vertices) - rank ∂_1
    n_0 = length(current_faces)
    β_0 = n_0 - rank_prev
    β_0 == 1 || return false

    return true
end

function find_wedge_pair(K::SimplicialComplex)
    m = nv(K)
    MNF_sets = Set(minimal_nonfaces(K))

    # iterate over all pairs of vertices
    for v in 1:m-1
        for w in v+1:m

            pair = Set((v, w))
            is_pair = true

            for F in MNF_sets
                # exactly one present → break condition
                if (v in F) ⊻ (w in F)
                    is_pair = false
                    break
                end
            end

            # if always together and pair not itself MNF → not seed
            if is_pair && !(pair in MNF_sets)
                return v
            end
        end
    end
    return -1
end

function find_seed(K::SimplicialComplex)
    v = find_wedge_pair(K)
    seed_K = K
    while v != -1
        seed_K = link_subcomplex(seed_K, [v])
        v = find_wedge_pair(seed_K)
    end
    return seed_K
end


@inline function link_facets(facets::Union{Vector{U},Tuple{Vararg{U}}}, v::U) where {U<:Unsigned}
    mask = one(U) << v
    out = U[]
    for f in facets
        if (f & mask) != 0
            push!(out, f ⊻ mask)
        end
    end
    return out
end

@inline function link_without_vertex(facets::Union{Vector{U},Tuple{Vararg{U}}}, vmask::U) where {U<:Unsigned}
    lk = U[]
    for f in facets
        if (f & vmask) != 0
            push!(lk, f & ~vmask)
        end
    end
    sort!(lk)
    return lk
end

@inline function vertex_mask(facets::Union{Vector{U},Tuple{Vararg{U}}}) where {U<:Unsigned}
    m = zero(U)
    for f in facets
        m |= f
    end
    return m
end

@inline function vertices_from_mask(mask::U) where {U<:Unsigned}
    out = U[]
    for i in 0:(8*sizeof(mask)-1)
        if (mask >> i) & 1 == 1
            push!(out, U(i))
        end
    end
    return out
end

@inline facet_dim(f::U) where {U<:Unsigned} = count_ones(f) - 1 

function is_seed_bit(facets::Union{Vector{U},Tuple{Vararg{U}}}) where {U<:Unsigned}
    verts_mask = vertex_mask(facets)
    verts = vertices_from_mask(verts_mask)

    # Compute all links once
    links = Dict{U,Vector{U}}()
    for v in verts
        links[v] = link_facets(facets, v)
    end

    # Check all vertex pairs
    for i in 1:length(verts)-1
        v = verts[i]
        lk_v = links[v]
        mask_v = one(U) << v

        for j in i+1:length(verts)
            w = verts[j]
            mask_w = one(U) << w

            # Check if w appears in lk(v) - means (v,w) is a face
            appears = any(f -> (f & mask_w) != 0, lk_v)
            appears || continue

            # Compute lk(w) with v and w swapped
            lk_w_relabel = U[]
            for f in facets
                if (f & mask_w) != 0
                    g = f & ~mask_w  # remove w
                    if (g & mask_v) != 0  # if has v
                        g = (g & ~mask_v) | mask_w  # swap v↔w
                    end
                    push!(lk_w_relabel, g)
                end
            end
            sort!(lk_w_relabel)

            if lk_v == lk_w_relabel
                return false
            end
        end
    end

    return true
end

function find_wedge_vertex(facets::Union{Vector{U},Tuple{Vararg{U}}}) where {U<:Unsigned}
    verts_mask = vertex_mask(facets)
    verts = vertices_from_mask(verts_mask)

    for i in 1:length(verts)-1
        v = verts[i]
        vmask = one(U) << v
        lk_v = link_without_vertex(facets, vmask)

        for j in i+1:length(verts)
            w = verts[j]
            wmask = one(U) << w

            # Check if (v,w) is a face
            is_face = any(f -> (f & vmask != 0) && (f & wmask != 0), facets)
            is_face || continue

            lk_w = link_without_vertex(facets, wmask)

            # Remove the other vertex from both links
            lk_v2 = sort!([f & ~wmask for f in lk_v])
            lk_w2 = sort!([f & ~vmask for f in lk_w])

            if lk_v2 == lk_w2
                return v
            end
        end
    end

    return UInt32(0xffffffff)
end

function find_seed_bit(facets::Union{Vector{U},Tuple{Vararg{U}}}) where {U<:Unsigned}
    current = collect(facets)  # Convert to Vector for mutation

    while true
        v = find_wedge_vertex(current)
        v == UInt32(0xffffffff) && return current
        current = link_without_vertex(current, one(U) << v)
    end
end

# Oscar conversion function
function to_oscar_complex(facets::Union{Vector{U},Tuple{Vararg{U}}}) where {U<:Unsigned}
    Vmask = vertex_mask(facets)
    verts = vertices_from_mask(Vmask)

    # Convert facets from bitmasks to vertex sets (1-indexed for Oscar)
    facets_as_sets = Vector{Int}[]
    for f in facets
        facet_verts = Int[]
        for v in verts
            if (f >> v) & 1 == 1
                push!(facet_verts, Int(v) + 1)  # Oscar uses 1-indexed vertices
            end
        end
        push!(facets_as_sets, facet_verts)
    end

    return simplicial_complex(facets_as_sets)
end

# Check if complex is isomorphic to any in the database
function is_isomorphic_to_any(facets_bin::Union{Vector{U},Tuple{Vararg{U}}}, db) where {U<:Unsigned}
    K_oscar = to_oscar_complex(facets_bin)

    for existing_bin in db
        existing_oscar = to_oscar_complex(existing_bin)
        if Oscar.is_isomorphic(K_oscar, existing_oscar)
            return true
        end
    end

    return false
end

# ── Cheap isomorphism invariant (all computed from bitmasks, no Oscar) ────────
function complex_invariant(facets::Union{Vector{U},Tuple{Vararg{U}}}) where {U<:Unsigned}
    # Facet size multiset: e.g. [3,3,3,4] — fast and discriminating
    facet_sizes = sort!([count_ones(f) for f in facets])
    # Vertex degree sequence: sorted number of facets each vertex appears in
    Vmask = vertex_mask(facets)
    verts = vertices_from_mask(Vmask)
    degrees = sort!([count(f -> (f >> v) & 1 == 1, facets) for v in verts])

    return (facet_sizes, degrees)
end

# ── Indexed database: invariant → list of complexes ───────────────────────────
# Replaces a flat Set with a Dict of small buckets.
# Oscar is only called within the same bucket (typically size 1).

function build_index(db_seed,type::Type{U}=UInt32) where {U<:Unsigned}
    idx = Dict{Tuple{Vector{Int},Vector{Int}}, Vector{Tuple{Vararg{U}}}}()
    for facets_bin in db_seed
        inv = complex_invariant(facets_bin)
        push!(get!(idx, inv, []), facets_bin)
    end
    return idx
end

function is_isomorphic_to_any_indexed(facets_bin::Union{Vector{U},Tuple{Vararg{U}}}, idx) where {U<:Unsigned}
    inv = complex_invariant(facets_bin)
    bucket = get(idx, inv, nothing)
    isnothing(bucket) && return false          # invariant not seen → definitely new
    K = to_oscar_complex(facets_bin)
    for existing in bucket                     # bucket is usually size 0 or 1
        Oscar.is_isomorphic(K, to_oscar_complex(existing)) && return true
    end
    return false
end

function push_indexed!(db_seed, idx, facets_bin)
    tup = Tuple(facets_bin)
    push!(db_seed, tup)
    inv = complex_invariant(facets_bin)
    push!(get!(idx, inv, []), tup)
end

function index_to_bin(facets::Vector{Vector{Int}},type::Type{U}=UInt32) where {U<:Unsigned}
    @assert max([max(f...) for f in facets]...) <= 32
    return Tuple(sort([reduce(|, [one(U) << (i - 1) for i in facet]) for facet in facets]))
end

