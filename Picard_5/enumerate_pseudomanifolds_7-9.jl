include("../enumerate_kernel.jl")

function build_finalDB_single_v!(pseudo_manifolds_DB, mat_DB, iso_DB, mmax; mstart=-1)
    mmin = minimum(collect(keys(mat_DB)))
    mstart == -1 && (mstart = mmin)

    for m = mstart:mmax
        pseudo_manifolds_DB[m] = Vector{Set{BitVector}}()
        n_matroids = length(mat_DB[m])

        for (l, bases_bin) in enumerate(mat_DB[m])
            V_bin            = reduce(|, bases_bin)
            compl_bases_bin  = [base ⊻ V_bin for base in bases_bin]
            ridges, A        = boundary_incidence_facets_to_ridges(compl_bases_bin)
            kernel_basis     = kernel_basis_mod2_sparse(A)
            rows             = sparse_rows(A)          # ← precompute once per matroid
            push!(pseudo_manifolds_DB[m], Set{BitVector}())

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

                # n_pm = length(pseudo_manifolds_DB[m][l])
                # println(stderr, "m=$m | matroid $l/$n_matroids | pm=$n_pm")
            else
                iso_list = iso_DB[m][l]
                n_iso    = length(iso_list)

                # Caches, per distinct index_contraction, the first vertex's
                # embedding and the full set of pseudomanifolds it produced --
                # so a later vertex with the SAME contraction can try to reuse
                # it via relabeling instead of redoing the kernel enumeration.
                dict_one_per_isom = Dict{Int,Tuple{Int,Any,Set{BitVector}}}()
                compl_set = Set(compl_bases_bin)

                # Progress is measured against the real workload: each distinct
                # contraction pays for kernel enumeration once, not once per vertex.
                unique_ics    = unique(ic for (ic, _) in iso_list)
                n_links_total = sum(length(pseudo_manifolds_DB[m-1][ic]) for ic in unique_ics)

                prog = Progress(n_links_total; desc="m=$m | links: ")

                for (iso_idx, (index_contraction, perm)) in enumerate(iso_list)
                    if haskey(dict_one_per_isom, index_contraction) # If the deletion was already enumerated, there is no need to call the whole enumeration
                        v_first, perm_first, set_pseudomanifolds = dict_one_per_isom[index_contraction]

                        target_of        = Dict(i => j for (i, j) in perm)
                        image_perm       = Set(values(target_of))
                        image_perm_first = Set(j for (i, j) in perm_first)

                        # The deleted vertex's real label is whichever element
                        # of l's support is missing from the embedding's image.
                        v_missing       = setdiff(support_set, image_perm)
                        v_first_missing = setdiff(support_set, image_perm_first)

                        v_label       = first(v_missing)
                        v_first_label = first(v_first_missing)

                        final_perm = Tuple{Int,Int}[(v_first_label, v_label)]
                        for (i, j_first) in perm_first
                            push!(final_perm, (j_first, target_of[i]))
                        end

                        # Only safe to reuse if final_perm is a genuine
                        # automorphism of l (maps l's own facets onto themselves).
                        if Set(relabel(compl_bases_bin, final_perm)) == compl_set
                            for K_bit in set_pseudomanifolds
                                relabeled_facets = relabel(compl_bases_bin[findall(K_bit)], final_perm)
                                new_K_bit        = subset_bitvector(compl_bases_bin, relabeled_facets)
                                push!(pseudo_manifolds_DB[m][l], new_K_bit)
                            end
                        else
                            @warn "Problem with the relabeling"
                        end
                    else # Otherwise, we need to enumerate
                        set_pseudomanifolds = Set{BitVector}()
                        links = collect(pseudo_manifolds_DB[m-1][index_contraction])

                        for L_bit in links
                            mandatory_facets_bin = relabel(mat_DB[m-1][index_contraction][findall(L_bit)], perm)
                            mandatory_facets_bit = subset_bitvector(bases_bin, mandatory_facets_bin)

                            if count(mandatory_facets_bit) != length(mandatory_facets_bin)
                                @warn "Some mandatory facets not found!" m l mandatory_facets_bin
                            end

                            state = prepare_kernel_enumeration(A, kernel_basis, mandatory_facets_bit, rows)
                            if state !== nothing
                                for K_bit in enumerate_from_prepared_parallel(state)
                                    facets_bin = compl_bases_bin[findall(K_bit)]
                                    euler_sphere_test(facets_bin) && push!(set_pseudomanifolds, copy(K_bit))
                                end
                            end

                            next!(prog; showvalues = [(:matroid, "$l/$n_matroids"), (:iso, "$iso_idx/$n_iso")])
                        end

                        # Keep the first representative for this index_contraction;
                        # don't overwrite it just because a later vertex's check failed.
                        if !haskey(dict_one_per_isom, index_contraction)
                            dict_one_per_isom[index_contraction] = (iso_idx, perm, copy(set_pseudomanifolds))
                        end
                        union!(pseudo_manifolds_DB[m][l], set_pseudomanifolds)
                    end
                end

                finish!(prog)
                # n_pm = length(pseudo_manifolds_DB[m][l])
                # println(stderr, "m=$m | matroid $l/$n_matroids | pm=$n_pm")
            end
        end
    end
end

mat_DB_bin = open("resources/mat_DB.jls", "r") do io
    deserialize(io)
end

iso_DB = open("resources/iso_DB_all.jls", "r") do io
    deserialize(io)
end

pseudo_manifolds_DB = Dict{Int,Vector{Set{BitVector}}}()

mmax = 9

build_finalDB_single_v!(pseudo_manifolds_DB, mat_DB_bin, iso_DB, mmax)

open("results/pseudo_manifolds_7-9.jls", "w") do io
    serialize(io, pseudo_manifolds_DB)
end