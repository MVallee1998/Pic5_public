# Parallel Gray code enumeration of the kernel of a Boolean matrix with constraints, for enumerating pseudo-manifolds with given facets. See `enumerate_kernel_with_constraints_bitvector` for the main API function.

using Base.Threads

include("simplicial_complex_utilities.jl")

# ─────────────────────────────────────────────────────────────────────────────
# § Sparse adjacency utilities
# ─────────────────────────────────────────────────────────────────────────────

"""Row → column-index adjacency list for a sparse Boolean CSC matrix."""
function sparse_rows(A::SparseMatrixCSC{Bool,Int})
    adj = [Int[] for _ in 1:size(A, 1)]
    @inbounds for col in 1:size(A, 2), ptr in A.colptr[col]:(A.colptr[col+1] - 1)
        push!(adj[A.rowval[ptr]], col)
    end
    return adj
end

"""Column → row-index adjacency list for a sparse Boolean CSC matrix."""
function sparse_cols(A::SparseMatrixCSC{Bool,Int})
    adj = [Int[] for _ in 1:size(A, 2)]
    @inbounds for col in 1:size(A, 2), ptr in A.colptr[col]:(A.colptr[col+1] - 1)
        push!(adj[col], A.rowval[ptr])
    end
    return adj
end

# ─────────────────────────────────────────────────────────────────────────────
# § Row-sum helpers
# ─────────────────────────────────────────────────────────────────────────────

"""Compute row sums of `y` w.r.t. a row adjacency list."""
function compute_row_sums(y::BitVector, rows::Vector{Vector{Int}})
    rs = zeros(Int, length(rows))
    @inbounds for (r, js) in enumerate(rows), j in js
        rs[r] += y[j]
    end
    return rs
end

"""Return `true` iff every entry of `row_sums` is 0 or 2."""
@inline function valid_row_sums(row_sums::Vector{Int})
    @inbounds for s in row_sums
        s == 0 || s == 2 || return false
    end
    return true
end

"""
Update `row_sums` in-place after `y` has been XOR-flipped along `support`.
`y[j]` already holds its new value; the delta is +1 if true, -1 if false.
"""
@inline function update_row_sums!(row_sums::Vector{Int}, y::BitVector,
                                   support::Vector{Tuple{Int,Vector{Int}}})
    @inbounds for (r, cols) in support, j in cols
        row_sums[r] += y[j] ? 1 : -1
    end
end

"""
1-based index of the bit that changes between Gray codes g(i-1) and g(i), i ≥ 1.
Uses the identity: the differing bit of consecutive Gray codes at step i is
trailing_zeros(i), so this equals trailing_zeros(i) + 1.
"""
@inline gray_flip_index(i::U) where {U<:Unsigned} = trailing_zeros(i) + 1

# ─────────────────────────────────────────────────────────────────────────────
# § Data structure
# ─────────────────────────────────────────────────────────────────────────────

"""Precomputed state for kernel enumeration; produced by `prepare_kernel_enumeration`."""
struct KernelEnumState
    B_ech            ::Vector{BitVector}
    free_indices     ::Vector{Int}
    y_forced         ::BitVector
    rows             ::Vector{Vector{Int}}
    free_row_support ::Vector{Vector{Tuple{Int,Vector{Int}}}}
    num_free         ::Int
end

# ─────────────────────────────────────────────────────────────────────────────
# § Core algorithms
# ─────────────────────────────────────────────────────────────────────────────


"""
Compute a basis for the kernel of `A` over GF(2) via sparse RREF.
`A` may be any sparse matrix with integer or Bool entries; only the parity of
each entry is used.
"""
function kernel_basis_mod2_sparse(A::SparseMatrixCSC{Bool,Int})
    m, n   = size(A)
    col_nz = sparse_cols(A)                      # col → [row indices]
    rows   = [Set{Int}() for _ in 1:m]           # GF(2) row representation

    @inbounds for (j, rs) in enumerate(col_nz), i in rs
        if j in rows[i]; delete!(rows[i], j); else push!(rows[i], j); end
    end

    pivcol = Int[]
    pivrow = Int[]
    row    = 1

    @inbounds for col in 1:n
        row > m && break

        # Find pivot
        piv = 0
        for r in row:m
            col in rows[r] && (piv = r; break)
        end
        piv == 0 && continue

        piv != row && ((rows[row], rows[piv]) = (rows[piv], rows[row]))
        push!(pivcol, col); push!(pivrow, row)

        pivot_set = rows[row]
        for r in 1:m
            r != row && col in rows[r] && symdiff!(rows[r], pivot_set)
        end

        row += 1
    end

    pivot_set = Set(pivcol)
    free_cols = [j for j in 1:n if !(j in pivot_set)]
    isempty(free_cols) && return BitVector[]

    basis = BitVector[]
    @inbounds for f in free_cols
        x_set = Set{Int}([f])
        for t in length(pivcol):-1:1
            c, r = pivcol[t], pivrow[t]
            parity = isodd(count(j -> j != c && j in x_set, rows[r]))
            parity ? push!(x_set, c) : delete!(x_set, c)
        end
        x = falses(n)
        for j in x_set; x[j] = true; end
        push!(basis, x)
    end

    return basis
end


"""
Echelon form of `B` over GF(2), prioritising columns in `S` (forced 1)
then `T` (forced 0). Returns `(B_ech, pivots)`.
"""
function kernel_basis_echelon_prioritize_with_constraints(B ::Vector{BitVector},
                                                          S ::BitVector,
                                                          T ::BitVector)
    isempty(B) && return (BitVector[], Int[])

    n, k   = length(B[1]), length(B)
    B_ech  = [copy(b) for b in B]
    pivots = Int[]

    col_order = vcat(findall(S), findall(T), findall(.!(S .| T)))

    current_row = 1
    for col in col_order
        current_row > k && break

        piv = 0
        for r in current_row:k
            B_ech[r][col] && (piv = r; break)
        end
        piv == 0 && continue

        piv != current_row &&
            ((B_ech[current_row], B_ech[piv]) = (B_ech[piv], B_ech[current_row]))
        push!(pivots, col)

        for r in 1:k
            r != current_row && B_ech[r][col] && (B_ech[r] .⊻= B_ech[current_row])
        end

        current_row += 1
    end

    rank = length(pivots)
    return rank == 0 ? (BitVector[], Int[]) : (B_ech[1:rank], pivots)
end


"""
Preprocess `A`, `B`, `S` for enumeration: propagate unit constraints,
compute echelon form, and build per-free-variable row support.
Returns a `KernelEnumState`, or `nothing` if the system is infeasible.
"""
function prepare_kernel_enumeration(A::SparseMatrixCSC{Bool,Int},
                                    B::Vector{BitVector},
                                    S::BitVector)
    n    = size(A, 2)
    rows = sparse_rows(A)

    if isempty(B)
        all(.!S) || return nothing
        return KernelEnumState(BitVector[], Int[], falses(n), rows,
                               Vector{Vector{Tuple{Int,Vector{Int}}}}[], 0)
    end

    S = copy(S)
    T = falses(n)

    # ── unit propagation ──────────────────────────────────────────────────────
    changed = true
    while changed
        changed = false
        for js in rows
            s         = count(j ->  S[j], js)
            free_cols = [j for j in js if !S[j] && !T[j]]
            s > 2 && return nothing
            if s == 2
                for j in free_cols
                    T[j] || (T[j] = true; changed = true)
                end
            elseif s == 1
                isempty(free_cols) && return nothing
                if length(free_cols) == 1
                    j = free_cols[1]
                    S[j] || (S[j] = true; changed = true)
                end
            elseif s == 0 && length(free_cols) == 1
                j = free_cols[1]
                T[j] || (T[j] = true; changed = true)
            end
        end
    end
    # ─────────────────────────────────────────────────────────────────────────

    B_ech, pivots = kernel_basis_echelon_prioritize_with_constraints(B, S, T)
    k = length(B_ech)
    @assert k ≤ 64

    forced_one   = falses(k)
    free_indices = Int[]
    for i in 1:k
        if     S[pivots[i]]; forced_one[i] = true
        elseif !T[pivots[i]]; push!(free_indices, i)
        end
    end

    y = falses(n)
    for i in 1:k; forced_one[i] && (y .⊻= B_ech[i]); end

    for j in 1:n
        ((S[j] && !y[j]) || (T[j] && y[j])) && return nothing
    end

    num_free = length(free_indices)
    num_free ≥ 40 && @warn "num_free = $num_free; Gray-code enumeration may be very slow."

    free_row_support = map(1:num_free) do fi
        bv = B_ech[free_indices[fi]]
        [(r, [j for j in rows[r] if bv[j]])
         for r in eachindex(rows) if any(j -> bv[j], rows[r])]
    end

    return KernelEnumState(B_ech, free_indices, y, rows, free_row_support, num_free)
end


"""
Enumerate all valid solutions in the Gray-code interval [`start`, `stop`].
`y` and `row_sums` must already be initialised to the state at index `start`.
Valid solutions (all row sums 0 or 2) are appended to `results`.
"""
function enumerate_block!(results  ::Vector{BitVector},
                          y        ::BitVector,
                          row_sums ::Vector{Int},
                          state    ::KernelEnumState,
                          start    ::UInt64,
                          stop     ::UInt64)
    valid_row_sums(row_sums) && push!(results, copy(y))
    @inbounds for i in (start + UInt64(1)):stop
        fi  = gray_flip_index(i)
        idx = state.free_indices[fi]
        y .⊻= state.B_ech[idx]
        update_row_sums!(row_sums, y, state.free_row_support[fi])
        valid_row_sums(row_sums) && push!(results, copy(y))
    end
end


function enumerate_from_prepared(state::KernelEnumState)
    y  = copy(state.y_forced)
    rs = compute_row_sums(y, state.rows)

    results = BitVector[]
    if state.num_free == 0
        valid_row_sums(rs) && push!(results, copy(y))
        return results
    end

    total = UInt64(1) << state.num_free
    sizehint!(results, min(total, 1000))
    enumerate_block!(results, y, rs, state, UInt64(0), total - UInt64(1))
    return results
end


function enumerate_from_prepared_parallel(state::KernelEnumState)
    state.num_free == 0 && return enumerate_from_prepared(state)

    num_threads = Threads.nthreads()
    prefix_bits = min(ceil(Int, log2(max(num_threads, 2))), state.num_free - 1)
    prefix_bits == 0 && return enumerate_from_prepared(state)

    num_blocks = 1 << prefix_bits
    block_size = UInt64(1) << (state.num_free - prefix_bits)

    thread_results = [BitVector[] for _ in 1:num_blocks]

    Threads.@threads for b in 0:(num_blocks - 1)
        let b = b, local_results = thread_results[b + 1]
            start      = UInt64(b) * block_size
            gray_start = start ⊻ (start >> 1)

            y = copy(state.y_forced)
            for fi in 1:state.num_free
                (gray_start >> (fi - 1)) & UInt64(1) == UInt64(1) &&
                    (y .⊻= state.B_ech[state.free_indices[fi]])
            end

            rs = compute_row_sums(y, state.rows)
            enumerate_block!(local_results, y, rs, state,
                             start, start + block_size - UInt64(1))
        end
    end

    return reduce(vcat, thread_results)
end


# ── Public API ────────────────────────────────────────────────────────────────

function enumerate_kernel_with_constraints_bitvector(A::SparseMatrixCSC{Bool,Int},
                                                      B::Vector{BitVector},
                                                      S::BitVector)
    state = prepare_kernel_enumeration(A, B, S)
    state === nothing && return BitVector[]
    return enumerate_from_prepared_parallel(state)
end