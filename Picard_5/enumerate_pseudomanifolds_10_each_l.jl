include("../enumerate_kernel.jl")

using Base.Threads
using Profile

const HEAVY_THRESHOLD = 35

function build_finalDB_single_v_one_l!(pseudo_manifolds_DB::Dict{Int,Vector{Set{BitVector}}},
                                       mat_DB::Dict{Int,Vector{Vector{UInt32}}},
                                       iso_DB::Dict{Int,Dict{Int,Vector{Tuple{Int,Any}}}},
                                       m, l)
    println("Processing m=$(m), l=$(l)")

    V_bin = reduce(|, mat_DB[m][l])
    bases_bin = mat_DB[m][l]
    compl_bases_bin = [base ⊻ V_bin for base in bases_bin]
    ridges, A = boundary_incidence_facets_to_ridges(compl_bases_bin)
    kernel_basis = kernel_basis_mod2_sparse(A)

    result          = Set{BitVector}()
    thread_results  = [Set{BitVector}() for _ in 1:Threads.maxthreadid()]
    rows = sparse_rows(A)

    for (index_contraction, perm) in iso_DB[m][l]
        links = collect(pseudo_manifolds_DB[m-1][index_contraction])

        prepared = Vector{BitVector}(undef, length(links))
        for (i, L_bit) in pairs(links)
            mandatory_facets_bin = relabel(mat_DB[m-1][index_contraction][findall(L_bit)], perm)
            prepared[i] = subset_bitvector(bases_bin, mandatory_facets_bin)
        end

        states = [prepare_kernel_enumeration(A, kernel_basis, prepared[i], rows) for i in eachindex(prepared)]

        @showprogress for state in states
            (state === nothing || state.num_free < HEAVY_THRESHOLD) && continue
            for K_bit in enumerate_from_prepared_parallel(state)
                facets_bin = compl_bases_bin[findall(K_bit)]
                euler_sphere_test(facets_bin) && push!(result, copy(K_bit))
            end
        end

        foreach(empty!, thread_results)
        prog = Progress(count(s -> s === nothing || s.num_free < HEAVY_THRESHOLD, states),
                        dt=0.5, desc="l=$(l), links=$(length(links))")
        Threads.@threads :static for i in eachindex(states)
            state = states[i]
            (state === nothing || state.num_free >= HEAVY_THRESHOLD) && continue
            tid = Threads.threadid()
            for K_bit in enumerate_from_prepared(state)
                facets_bin = compl_bases_bin[findall(K_bit)]
                euler_sphere_test(facets_bin) && push!(thread_results[tid], copy(K_bit))
            end
            next!(prog)
        end
        for lr in thread_results; union!(result, lr); end
    end

    open("results/pseudo_manifolds_10_part_$(l).jls", "w") do io
        serialize(io, result)
    end

    println("Done: l=$(l), found $(length(result)) pseudo-manifolds.")
end

# --- Parse task index from CLI ---
l = parse(Int, ARGS[1])
m = 10

mat_DB = open("resources/mat_DB.jls", "r") do io deserialize(io) end
iso_DB = open("resources/iso_DB.jls", "r") do io deserialize(io) end
pseudo_manifolds_DB = open("results/pseudo_manifolds_7-9_all.jls", "r") do io deserialize(io) end

Profile.init(n = 10^7, delay = 0.01)
@profile build_finalDB_single_v_one_l!(pseudo_manifolds_DB, mat_DB, iso_DB, m, l)
Profile.print(mincount=50)
