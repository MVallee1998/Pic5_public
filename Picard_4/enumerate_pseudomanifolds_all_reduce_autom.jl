include("../enumerate_kernel.jl")

mmax = 15

mat_DB_bin = open("resources/mat_DB.jls", "r") do io deserialize(io) end
iso_DB     = open("resources/iso_DB_all.jls", "r") do io deserialize(io) end

# ── Automorphisms of mat_DB[m][l], as `relabel`-compatible generator lists ──
# OSCAR's matroid_from_bases/automorphism_group work on a ground set 1:n, so
# we relabel l's real (possibly scattered) labels to 1:n, compute Aut there,
# then translate generators back to real labels for use with `relabel`.
function matroid_automorphism_generators(bases_bin::Vector{U}, support_set::Set{Int}) where {U<:Unsigned}
    sorted_support = sort(collect(support_set))
    n        = length(sorted_support)
    local_of = Dict(lbl => i for (i, lbl) in enumerate(sorted_support))   # real -> local 1:n
    real_of  = sorted_support                                            # local 1:n -> real

    local_bases = [sort([local_of[lbl] for lbl in Int.(vertices_from_mask(b)) .+ 1]) for b in bases_bin]

    M = matroid_from_bases(local_bases, n)
    G = automorphism_group(M)

    return [ [(real_of[i], real_of[i^g]) for i in 1:n] for g in gens(G) ]
end

# ── Reduce a Set{BitVector} of pseudomanifolds to one representative per
#    orbit under a list of generating automorphisms (given as `relabel`
#    pairs). Orbit computed by BFS over the generators -- never enumerates
#    the full automorphism group, which can be huge for symmetric matroids.
function reduce_orbits!(S::Set{BitVector}, compl_bases_bin::Vector{U},
                         gen_perms::Vector{Vector{Tuple{Int,Int}}}) where {U<:Unsigned}
    isempty(gen_perms) && return S

    visited         = Set{BitVector}()
    representatives = Set{BitVector}()

    for K in S
        K in visited && continue

        orbit = Set{BitVector}([K])
        queue = BitVector[K]
        while !isempty(queue)
            cur        = pop!(queue)
            cur_facets = compl_bases_bin[findall(cur)]
            for g in gen_perms
                relabeled_facets = relabel(cur_facets, g)
                new_K            = subset_bitvector(compl_bases_bin, relabeled_facets)
                if count(new_K) != length(relabeled_facets)
                    @warn "Automorphism generator produced an unrecognized facet"
                    continue
                end
                if new_K ∉ orbit
                    push!(orbit, new_K)
                    push!(queue, new_K)
                end
            end
        end

        union!(visited, orbit)
        push!(representatives, argmin(K2 -> Tuple(findall(K2)), orbit))
    end

    empty!(S)
    union!(S, representatives)
    return S
end

function build_finalDB_single_v!(pseudo_manifolds_DB::Dict{Int,Vector{Set{BitVector}}},
                                  mat_DB::Dict{Int,Vector{Vector{UInt16}}},
                                  iso_DB::Dict{Int,Dict{Int,Vector{Tuple{Int,Any}}}},
                                  mmax; mstart=-1, last_single_link=false)
    mmin = minimum(collect(keys(mat_DB)))
    mstart == -1 && (mstart = mmin)

    for m = mstart:mmax
        println(m)
        pseudo_manifolds_DB[m] = Vector{Set{BitVector}}()

        for (l, bases_bin) in enumerate(mat_DB[m])
            V_bin           = reduce(|, bases_bin)
            compl_bases_bin = [base ⊻ V_bin for base in bases_bin]
            ridges, A       = boundary_incidence_facets_to_ridges(compl_bases_bin)
            kernel_basis    = kernel_basis_mod2_sparse(A)
            rows            = sparse_rows(A)    # precompute once per matroid
            push!(pseudo_manifolds_DB[m], Set{BitVector}())

            # l's actual vertex labels (not necessarily 1:m).
            support_set = Set(Int.(vertices_from_mask(V_bin)) .+ 1)

            if m == mmin
                mandatory_facets_bit = falses(length(bases_bin))
                state = prepare_kernel_enumeration(A, kernel_basis, mandatory_facets_bit, rows)
                if state !== nothing
                    for K_bit in enumerate_from_prepared(state)
                        facets_bin = compl_bases_bin[findall(K_bit)]
                        euler_sphere_test(facets_bin) && push!(pseudo_manifolds_DB[m][l], copy(K_bit))
                    end
                end
            else
                dict_one_per_isom = Dict{Int,Tuple{Int,Any,Set{BitVector}}}()
                compl_set = Set(compl_bases_bin)

                for (v, (index_contraction, perm)) in enumerate(iso_DB[m][l])
                    reused = false

                    if haskey(dict_one_per_isom, index_contraction)
                        v_first, perm_first, set_pseudomanifolds = dict_one_per_isom[index_contraction]

                        target_of        = Dict(i => j for (i, j) in perm)
                        image_perm       = Set(values(target_of))
                        image_perm_first = Set(j for (i, j) in perm_first)

                        v_missing       = setdiff(support_set, image_perm)
                        v_first_missing = setdiff(support_set, image_perm_first)

                        if length(v_missing) == 1 && length(v_first_missing) == 1
                            v_label       = first(v_missing)
                            v_first_label = first(v_first_missing)

                            final_perm = Tuple{Int,Int}[(v_first_label, v_label)]
                            for (i, j_first) in perm_first
                                push!(final_perm, (j_first, target_of[i]))
                            end

                            if Set(relabel(compl_bases_bin, final_perm)) == compl_set
                                @showprogress desc="reusing previous results" for K_bit in set_pseudomanifolds
                                    relabeled_facets = relabel(compl_bases_bin[findall(K_bit)], final_perm)
                                    new_K_bit        = subset_bitvector(compl_bases_bin, relabeled_facets)
                                    push!(pseudo_manifolds_DB[m][l], new_K_bit)
                                end
                                reused = true
                            end
                        else
                            @warn "Could not uniquely determine deleted-vertex label" m l index_contraction v v_first
                        end
                    end

                    if !reused
                        set_pseudomanifolds = Set{BitVector}()
                        @showprogress desc="links=$(length(pseudo_manifolds_DB[m-1][index_contraction]))" for L_bit in pseudo_manifolds_DB[m-1][index_contraction]
                            mandatory_facets_bin = relabel(mat_DB[m-1][index_contraction][findall(L_bit)], perm)
                            mandatory_facets_bit = subset_bitvector(bases_bin, mandatory_facets_bin)

                            if count(mandatory_facets_bit) != length(mandatory_facets_bin)
                                @warn "Some mandatory facets not found!" m l mandatory_facets_bin
                            end

                            state = prepare_kernel_enumeration(A, kernel_basis, mandatory_facets_bit, rows)
                            state === nothing && continue
                            for K_bit in enumerate_from_prepared_parallel(state)
                                if m <= mmax * 0.75
                                    facets_bin = compl_bases_bin[findall(K_bit)]
                                    euler_sphere_test(facets_bin) && push!(set_pseudomanifolds, copy(K_bit))
                                else
                                    push!(set_pseudomanifolds, copy(K_bit))
                                end
                            end
                        end

                        if !haskey(dict_one_per_isom, index_contraction)
                            dict_one_per_isom[index_contraction] = (v, perm, copy(set_pseudomanifolds))
                        end
                        union!(pseudo_manifolds_DB[m][l], set_pseudomanifolds)
                    end

                    m == mmax && last_single_link && break
                end
            end

            # Reduce to one representative per Aut(mat_DB[m][l])-orbit before
            # moving on to the next l / level -- see the completeness caveat
            # discussed alongside this code.
            gen_perms = matroid_automorphism_generators(bases_bin, support_set)
            reduce_orbits!(pseudo_manifolds_DB[m][l], compl_bases_bin, gen_perms)
        end
    end
end

pseudo_manifolds_DB = Dict{Int,Vector{Set{BitVector}}}()
build_finalDB_single_v!(pseudo_manifolds_DB, mat_DB_bin, iso_DB, mmax; last_single_link=true)

open("results/pseudo_manifolds_all_reduce_autom.jls", "w") do io
    serialize(io, pseudo_manifolds_DB)
end