include("../enumerate_kernel.jl")

using Base.Threads
using Profile
using Statistics

function free_gens_distribution!(
        pseudomanifolds_DB::Dict{Int,Vector{Set{BitVector}}},
        mat_DB::Dict{Int,Vector{Vector{UInt32}}},
        iso_DB::Dict{Int,Dict{Int,Vector{Tuple{Int,Any}}}},
        m)

    # --- kernel dimension d: 46 values, store them all ---
    kernel_sizes = Int[]

    # --- |𝒮°| and reduction d - |𝒮°|: frequency arrays ---
    max_val       = 200
    freq          = zeros(Int, max_val + 1)   # freq[k+1]     counts num_free == k
    red_freq      = zeros(Int, max_val + 1)   # red_freq[k+1] counts reduction == k
    red_sum       = 0
    total_calls   = 0
    skipped_calls = 0

    @showprogress for (l, bases_bin) in enumerate(mat_DB[m])
        V_bin           = reduce(|, bases_bin)
        compl_bases_bin = [base ⊻ V_bin for base in bases_bin]
        ridges, A       = boundary_incidence_facets_to_ridges(compl_bases_bin)
        kernel_basis    = kernel_basis_mod2_sparse(A)
        d               = length(kernel_basis)
        push!(kernel_sizes, d)
        rows = sparse_rows(A)

        for (index_contraction, perm) in iso_DB[m][l]
            links = collect(pseudomanifolds_DB[m-1][index_contraction])

            prepared = Vector{BitVector}(undef, length(links))
            for (i, L_bit) in pairs(links)
                mandatory_facets_bin = relabel(
                    mat_DB[m-1][index_contraction][findall(L_bit)], perm)
                prepared[i] = subset_bitvector(bases_bin, mandatory_facets_bin)
            end

            for i in eachindex(prepared)
                state = prepare_kernel_enumeration(
                    A, kernel_basis, prepared[i], rows)
                if state === nothing
                    skipped_calls += 1
                    continue
                end
                nf        = state.num_free
                reduction = d - nf
                freq[nf + 1]        += 1
                red_freq[reduction + 1] += 1
                red_sum              += reduction
                total_calls          += 1
            end

            break  # single contraction per matroid: correct at m = m_max
        end
    end

    sep = "=" ^ 60

    # ----------------------------------------------------------------
    # Kernel dimension d
    # ----------------------------------------------------------------
    println(sep)
    println("Kernel dimension d  ($(length(kernel_sizes)) matroids, m=$m)")
    println(sep)
    println("  min    = $(minimum(kernel_sizes))")
    println("  max    = $(maximum(kernel_sizes))")
    println("  mean   = $(round(mean(kernel_sizes), digits=2))")
    println("  median = $(Int(median(kernel_sizes)))")
    println()
    println("  d  | count | histogram")
    println("  ---+-------+" * "─"^20)
    for d in minimum(kernel_sizes):maximum(kernel_sizes)
        c = count(==(d), kernel_sizes)
        c == 0 && continue
        println("  $(lpad(d,2)) | $(lpad(c,5)) | $("█"^c)")
    end

    # ----------------------------------------------------------------
    # Free generators |𝒮°|
    # ----------------------------------------------------------------
    active        = findall(>(0), freq)
    lo, hi        = active[1] - 1, active[end] - 1
    freq_trim     = freq[lo+1:hi+1]
    weighted_mean = sum((lo + k - 1) * freq_trim[k]
                        for k in eachindex(freq_trim)) / total_calls
    cum, half, med = 0, total_calls / 2, lo
    for k in eachindex(freq_trim)
        cum += freq_trim[k]
        if cum >= half; med = lo + k - 1; break; end
    end

    println()
    println(sep)
    println("|𝒮°| distribution  ($total_calls calls, $skipped_calls pruned by unit propagation)")
    println("(one contraction per matroid, reflecting actual computation)")
    println(sep)
    println("  min    = $lo")
    println("  max    = $hi")
    println("  mean   = $(round(weighted_mean, digits=2))")
    println("  median = $med")
    println()
    bin_w = 5
    println("  |𝒮°| range  |      count | % of calls | bar")
    println("  -----------+------------+------------+" * "─"^25)
    for b in lo:bin_w:hi
        c   = sum(freq[k+1] for k in b:min(b+bin_w-1, hi))
        c == 0 && continue
        pct = round(100 * c / total_calls, digits=1)
        bar = "█" ^ round(Int, 25 * c / total_calls)
        println("  [$(lpad(b,2)), $(lpad(b+bin_w,2)))    |" *
                " $(lpad(c,10)) | $(lpad(pct,9))% | $bar")
    end

    # ----------------------------------------------------------------
    # Reduction d - |𝒮°|
    # ----------------------------------------------------------------
    red_active    = findall(>(0), red_freq)
    red_lo, red_hi = red_active[1] - 1, red_active[end] - 1
    red_trim      = red_freq[red_lo+1:red_hi+1]
    red_mean      = red_sum / total_calls
    cum, half, red_med = 0, total_calls / 2, red_lo
    for k in eachindex(red_trim)
        cum += red_trim[k]
        if cum >= half; red_med = red_lo + k - 1; break; end
    end

    println()
    println(sep)
    println("Reduction d - |𝒮°|  (same $total_calls calls)")
    println(sep)
    println("  min    = $red_lo")
    println("  max    = $red_hi")
    println("  mean   = $(round(red_mean, digits=2))")
    println("  median = $red_med")
    println()
    println("  reduction range  |      count | % of calls | bar")
    println("  ----------------+------------+------------+" * "─"^25)
    for b in red_lo:bin_w:red_hi
        c   = sum(red_freq[k+1] for k in b:min(b+bin_w-1, red_hi))
        c == 0 && continue
        pct = round(100 * c / total_calls, digits=1)
        bar = "█" ^ round(Int, 25 * c / total_calls)
        println("  [$(lpad(b,2)), $(lpad(b+bin_w,2)))           |" *
                " $(lpad(c,10)) | $(lpad(pct,9))% | $bar")
    end

    return kernel_sizes, freq[lo+1:hi+1], red_freq[red_lo+1:red_hi+1]
end

function kernel_size_distribution!(mat_DB::Dict{Int,Vector{Vector{UInt32}}}, m)
    kernel_sizes = Int[]

    @showprogress for (l, _) in enumerate(mat_DB[m])
        V_bin = reduce(|, mat_DB[m][l])
        compl_bases_bin = [base ⊻ V_bin for base in mat_DB[m][l]]
        ridges, A = boundary_incidence_facets_to_ridges(compl_bases_bin)
        kernel_basis = kernel_basis_mod2_sparse(A)
        push!(kernel_sizes, length(kernel_basis))
    end

    # Summary statistics
    println("n matroids : $(length(kernel_sizes))")
    println("Min        : $(minimum(kernel_sizes))")
    println("Max        : $(maximum(kernel_sizes))")
    println("Mean       : $(round(mean(kernel_sizes), digits=2))")
    println("Median     : $(median(kernel_sizes))")

    # Exact frequency table
    d_min, d_max = minimum(kernel_sizes), maximum(kernel_sizes)
    freq = Dict(d => count(==(d), kernel_sizes) for d in d_min:d_max)

    println("\n d  | count | bar")
    println("----+-------+" * "-"^20)
    for d in d_min:d_max
        c = get(freq, d, 0)
        c == 0 && continue
        bar = "█" ^ c
        println("$(lpad(d,3)) | $(lpad(c,5)) | $bar")
    end

    return kernel_sizes
end

# --- Parse task index from CLI ---
m = 10

mat_DB = open("../resources/mat_DB.jls", "r") do io deserialize(io) end
iso_DB = open("../resources/iso_DB.jls", "r") do io deserialize(io) end
pseudomanifolds_DB = open("../results/pseudomanifolds_7-9.jls", "r") do io deserialize(io) end


free_gens_distribution!(pseudomanifolds_DB,mat_DB,iso_DB, m)