# export/to_plaintext.jl
# Exports all TC seeds from both Picard 4 and Picard 5 databases
# to human-readable plain text, one seed per line.

using Serialization

function decode_mask(mask::T, nbits::Int=32) where T <: Unsigned
    sort([i for i in 1:nbits if (mask >> (i-1)) & 1 == 1])
end

function export_seeds(tc::Dict, filename::String, nbits::Int, picard_number::Int)
    total = 0
    open(filename, "w") do f
        for key in sort(collect(keys(tc)), by = k -> (k[2], k[1]))
            d, nv = key
            nv - d - 1 == picard_number || continue
            seeds = tc[key]
            for seed in seeds
                # decode original bitmask labels (each facet is a sorted Vector{Int})
                facets_decoded = [decode_mask(mask, nbits) for mask in seed]
                sort!(facets_decoded)

                # relabel to consecutive 1..nv
                all_verts = sort(unique([v for f in facets_decoded for v in f]))
                @assert length(all_verts) == nv "Expected $nv vertices, got $(length(all_verts))"
                relabel = Dict(v => i for (i, v) in enumerate(all_verts))

                # apply relabeling and re-sort each facet
                facets_relabeled = [sort([relabel[v] for v in f]) for f in facets_decoded]
                sort!(facets_relabeled)

                println(f, join(
                    ["{" * join(verts, ",") * "}" for verts in facets_relabeled],
                    " "))
                total += 1
            end
            println(stderr, "  (dim=$(d), nv=$(nv)): $(length(seeds)) seeds")
        end
    end
    println(stderr, "Total exported: $total seeds")
    return total
end

# --- Picard 4 ---
tc4 = open("../Picard_4/results/TC_seed_PLS.jls", "r") do io
    deserialize(io)
end
n4 = export_seeds(tc4, "../objects/TC_seeds_pic4.txt", 16, 4)
@assert n4 == 3153  "Expected 3153 Picard-4 seeds, got $n4"

# --- Picard 5 ---
tc5 = open("../Picard_5/results/TC_seed_PLS_7-10.jls", "r") do io
    deserialize(io)
end
n5 = export_seeds(tc5, "../objects/TC_seeds_pic5.txt", 32, 5)
@assert n5 == 200030  "Expected 200030 Picard-5 seeds, got $n5"