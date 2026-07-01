using Printf
using ProgressMeter

function orbit_reduction_table!(
        mat_DB::Dict{Int,Vector{Vector{UInt16}}},
        iso_DB::Dict{Int,Dict{Int,Vector{Tuple{Int,Any}}}},
        mmin::Int, mmax::Int)

    total_vertices_all = 0
    actual_calls_all   = 0
    results            = Tuple{Int,Int,Int}[]   # (m, total_vertices, actual_calls)

    @showprogress for m in mmin+1:mmax          # base level mmin uses Algo 1 directly
        total_vertices_m = 0
        actual_calls_m   = 0

        for (l, bases_bin) in enumerate(mat_DB[m])
            V_bin           = reduce(|, bases_bin)
            compl_bases_bin = [base ⊻ V_bin for base in bases_bin]
            support_set     = Set(Int.(vertices_from_mask(V_bin)) .+ 1)
            compl_set       = Set(compl_bases_bin)

            # mirrors dict_one_per_isom from build_finalDB_single_v!
            # stores only the FIRST representative per index_contraction
            dict_seen = Dict{Int, Tuple{Int, Any}}()

            for (v, (index_contraction, perm)) in enumerate(iso_DB[m][l])
                total_vertices_m += 1
                reuse = false

                if haskey(dict_seen, index_contraction)
                    v_first, perm_first = dict_seen[index_contraction]

                    target_of        = Dict(i => j for (i, j) in perm)
                    image_perm       = Set(values(target_of))
                    image_perm_first = Set(j for (i, j) in perm_first)

                    v_missing        = setdiff(support_set, image_perm)
                    v_first_missing  = setdiff(support_set, image_perm_first)

                    if !isempty(v_missing) && !isempty(v_first_missing)
                        v_label       = first(v_missing)
                        v_first_label = first(v_first_missing)

                        final_perm = Tuple{Int,Int}[(v_first_label, v_label)]
                        for (i, j_first) in perm_first
                            push!(final_perm, (j_first, target_of[i]))
                        end

                        # automorphism check: same condition as in build_finalDB_single_v!
                        if Set(relabel(compl_bases_bin, final_perm)) == compl_set
                            reuse = true
                        end
                    end
                end

                if !reuse
                    actual_calls_m += 1
                    # mirror: only store the first representative, never overwrite
                    if !haskey(dict_seen, index_contraction)
                        dict_seen[index_contraction] = (v, perm)
                    end
                end
            end
        end

        push!(results, (m, total_vertices_m, actual_calls_m))
        total_vertices_all += total_vertices_m
        actual_calls_all   += actual_calls_m
    end

    # ----------------------------------------------------------------
    # Print table
    # ----------------------------------------------------------------
    sep = "=" ^ 70
    println(sep)
    println("Orbit reduction — Picard number 4  (m = $(mmin+1) to $mmax)")
    println(sep)
    @printf "  %3s | %18s | %22s | %s\n" "m" "non-loop vertices" "orbit representatives" "reduction factor"

    println("  ----+--------------------+------------------------+----------------")
    for (m, tv, ac) in results
        @printf "  %3d | %18d | %22d | %.2f×\n" m tv ac tv/ac
    end
    println("  ----+--------------------+------------------------+----------------")
    @printf "  %3s | %18d | %22d | %.2f×\n" "tot" total_vertices_all actual_calls_all (total_vertices_all/actual_calls_all)


    return results
end


include("../../enumerate_kernel.jl")


mat_DB_bin = open("../resources/mat_DB.jls", "r") do io deserialize(io) end
iso_DB     = open("../resources/iso_DB_all.jls", "r") do io deserialize(io) end

orbit_reduction_table!(mat_DB_bin, iso_DB, 6,15)

