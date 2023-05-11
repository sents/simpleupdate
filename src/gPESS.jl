# [[file:../../notes.org::*gPESS][gPESS:1]]
module gPESS
using ..OptimalContraction
using ..Operators
using ..Util
export Simplex, PESSSite, PESSUnitCell, PESSModel,
    nsites, nvirt, virtualsiteinds, nsimps, psize,
    show, normalized_ops,
    pess_unitcell_from_ordered_structurematrix,
    register!, static_pess_su_info
# gPESS:1 ends here

# [[file:../../notes.org::*gPESS][gPESS:2]]
using LinearAlgebra: full!
using Base: ReverseOrdering
using StaticArrays
using LinearAlgebra
using TensorOperations
import Combinatorics: combinations


"""
    Simplex{M,N,T,A<:AbstractArray{T}}
A PESS simplex connecting multiple sites.

For practical purposes all sites that only connect to one
simplex are absorbed into the simplex.

The first `N` dimensions of `tensor` are connected to `siteind[i]` of `site[i]`
The next `M` dimensions are meant for the absorbed, 'virtual' sites.

# Fields
- `tensor :: A`: The simplex tensor
- `sites :: NTuple{N,Int}`: A tuple of the indices of the connected sites in the site list
- `siteinds :: NTuple{N,Int}`: A tuple of the indices of the site tensor the simplex is connected to
"""
mutable struct Simplex{M,N,T,A<:AbstractArray{T}}
    tensor::A
    sites::NTuple{N,Int}
    vsites::NTuple{M,Int}
    siteinds::NTuple{N,Int}
    function Simplex{M}(
        tensor::A,
        sites::NTuple{N,Int},
        vsites::NTuple{M,Int},
        siteinds::NTuple{N,Int}) where {N,M,T,A<:AbstractArray{T}}
        @assert ndims(tensor) == N + M """
            tensor with dim $(ndims(tensor)) should to have $(N) simplex and $(M)
            virtual dimensions!
            """
        new{M,N,T,A}(tensor, sites, vsites, siteinds)
    end
end

"""
    PESSSite{N, T1, T2, M<:AbstractArray{T1}}
A site tensor in a PESS tensor network state.
The first N Dimensions of `tensor` are connected to Simplices,
the last Dimension represents the physical index.
# Fields
- `tensor :: M`: The tensor containing N dimensions connecting to simplices and one physical
- `envVectors :: SizedArray{N,Vector{T2}}`: Vectors containing entanglement mean field weights in the direction of the connected simplices
"""
mutable struct PESSSite{N,T1,T2,M<:AbstractArray{T1}}
    tensor::M
    envVectors::SizedVector{N,Vector{T2}}
    function PESSSite(tensor::M,
        envVectors::NTuple{N,Vector{T2}}) where {N, T1, T2, M<:AbstractArray{T1}}
        @assert ndims(tensor)==N+1 """
        Dimension of `tensor` has to be N+1=$(N+1)!
        """
        new{N,T1,T2,M}(tensor, SizedVector{N,Vector{T2}}(envVectors))
    end
end

"""
    PESSUnitCell{T1,T2}
A unitcell consisting of a Vector of sites and a vector of simplices
"""
struct PESSUnitCell{T1,T2}
    sites::Vector{PESSSite{<:Any, T1, T2}}
    simplices::Vector{Simplex{<:Any, <:Any, T1}}
end

"""
    PESSModel{T1, T2}
Describes a `PESSUnitCell` together with observables that can be
calculated on the UnitCell. To model different kind of interactions
it contains a list `sitetypes` that maps the sites to an integer.

# Fields
- `unitcell :: PESSUnitCell{T1, T2}`: The PESSUnitCell containing sites and simplices
- `sitetypes :: Vector{Int}`: A list of length(unitcell.sites) sites, giving each site
an integer to specify its `type`. Defaults to ones if all interactions are equal.
- `observables :: Dict{Symbol, Vector{Operator{T1}}}`: A dict containing additional
observables in or model. Always contains atleast :hamiltonian.
- `m_connect :: Array{Int}`: The connection matrix describing how a primitive unit cell
connects to positively adjacent unit cells
- `tile_pattern :: Array{Int}`: The tile pattern with which the connection matrix is tiled
"""
struct PESSModel{T1, T2}
    unitcell :: PESSUnitCell{T1, T2}
    m_connect :: Array{Int}
    tile_pattern :: Array{Int}
    sitetypes :: Vector{Int}
    observables :: Dict{Symbol, Vector{Operator{T1}}}
    function PESSModel(
        unitcell::PESSUnitCell{T1, T2},
        m_connect::AbstractMatrix{Int},
        tile_pattern::AbstractArray{Int},
        sitetypes::Vector{Int} = [1 for _ in 1:length(unitcell.sites)],
        observables = Dict()) where {T1, T2}
        new{T1, T2}(
            unitcell,
            m_connect,
            tile_pattern |> collect,
            sitetypes,
            convert(Dict{Symbol, Vector{Operator{T1}}}, observables))
    end
end

Base.show(io::IO, S::Simplex{M,N,T,A}) where {M,N,T,A} = print(io,
    "Simplex{$(M),$(N),$(T)}: $(size(S.tensor))", "\n",
    "Connections: \n",
    ("$i -> $si\n" for (i,si) in zip(S.sites,S.siteinds))...,
    "Virtual sites: ", join(string.(S.vsites), ", ")
)

Base.show(io::IO, s::PESSSite{N,T1,T2}) where {N,T1,T2} = print(io,
    "PESSSite{$N,$T1,$T2}: $(size(s.tensor))")

Base.show(io::IO, u::PESSUnitCell{T1,T2}) where {T1,T2} = print(io,
    "PESSUnitCell{$T1,$T2}: $(length(u.sites)) sites, $(length(u.simplices)) simplices"
)

Base.show(io::IO, m::PESSModel{T1,T2}) where {T1,T2} = print(io,
    "PESSModel{$T1,$T2}: ", length(m.unitcell.sites), " sites, ",
    "$(length(m.unitcell.simplices)) simplices\n",
    "Number of sitetypes: ", length(m.sitetypes |> unique), "\n",
    "Defined observables: ",
    join([string.(keys(m.observables))], ", ")
)


nsites(::Simplex{M,N}) where {M,N} = N
nvirt(::Simplex{M,N}) where {M,N} = M
virtualsiteinds(s::Simplex) = range(nsites(s)+1, nvirt(s))
nsimps(::PESSSite{N}) where N = N
psize(site::PESSSite) = size(site.tensor)[end]

"""
    simpleupdate_step(sites, S, op, info, max_bond_rank, sv_cutoff)
A single PESS simple update step on a single simplex consisting of:
- Contracting the environment vectors of sites to adjacent simplices
- QR factoring the sites
- Contracting the operator-simplex-sites network to a tensor T
- Calculating U unitaries via a eigenvalue HOSVD
- Discarding EVs smaller than sv_cutoff and truncating the Us and eigenvalues to max_bond_rank
- Storing the truncated EVs as new environment vectors
- Retrieving S from the \$U^{†}s\$ and T
- Reversing the QR factorisation of the sites
- Reemitting the environment contracted in the first step
"""
function simpleupdate_step(sites::Vector{PESSSite{T1,T2}},
    S::Simplex{T1,N}, op, info,
    max_bond_rank, sv_cutoff) where {T1,T2,N}
    qs = Array{T1}[]
    rs = Array{T1}[]
    for (i, site) in enumerate(sites)
        contract_env!(site, info.env_inds[i])
        q, r = qr_site(site, info.qr_perms[i])
        push!(qs, q)
        push!(rs, r)
    end

    T = contract_op(op.tensor, S.tensor, tuple(rs...), info.contract_op)

    Us = Array{T1}[]

    step_diff = 0.0

    for (i, site) in enumerate(sites)
        U, Σ = eigsvd_trunc(T, info.svd_perms[i], max_bond_rank, sv_cutoff)
        step_diff += padded_inner_product(site.envVectors[info.sinds[i]], Σ)
        site.envVectors[info.sinds[i]] = Σ
        push!(Us, U)
    end

    recalc_S!(S, T, Us, info.S)

    for (i, (site, U, q)) in enumerate(zip(sites, Us, qs))
        deqr_site!(site, q, U, info.qr_perms[i])
        emit_env!(site, info.env_inds[i])
    end
    return step_diff
end

function qr_site(site, perm)
    Asize_permuted = size(site.tensor)[perm]
    sA_r = Asize_permuted[end-1:end]
    sA_q = Asize_permuted[1:end-2]
    sA_qr = min(prod(sA_q), prod(sA_r))
    A_reshaped = reshape(permutedims(site.tensor, perm),
        (prod(sA_q), prod(sA_r)))
    q, r = qr(A_reshaped)
    r_reshaped = reshape(r, (sA_qr, sA_r...))
    return q, r_reshaped
end

function contract_env!(site::PESSSite, inds)
    site.tensor .= .*(site.tensor,
        (reshape(site.envVectors[i], n)
         for (i, n) in inds)...)
end

function emit_env!(site::PESSSite, inds)
    site.tensor .= .*(site.tensor,
        (reshape(1 ./ site.envVectors[i], n)
         for (i, n) in inds)...)
end

@generated function contract_op(op, S, rs::NTuple{N,T_Site},
    info::Tuple{Val{i_op},Val{i_S},Val{i_rs}}) where {
    N,T_Site,
    i_op,i_S,i_rs}
    rightside = Expr(:call, :*,
        :(op[$(i_op...)]),
        :(S[$(i_S...)]),
        (:(rs[$i][$(i_rs[i]...)]) for i in 1:N)...
    )
    return :(@tensor out[:] = $rightside)
end

function eigsvd_trunc(T, info, max_bond_rank, sv_cutoff)
    Tsize_perm = size(T)[info.perm]
    sT_open = Tsize_perm[1:2]
    sT_closed = Tsize_perm[3:end]
    T_reshaped = reshape(permutedims(T, info.perm), (prod(sT_open), prod(sT_closed)))
    λ, U_r = eigen(T_reshaped * T_reshaped')
    sp = sortperm(λ, order=ReverseOrdering)
    Σ = sqrt.(λ)[sp]
    U = U_r[:, sp]
    Σ ./= norm(Σ)
    svs_over_cutoff = (count >= (sv_cutoff), Σ)
    new_dim = min(svs_over_cutoff, max_bond_rank)
    U_trunc = U[:, 1:new_dim]
    Σ_trunc = Σ[1:new_dim]
    return reshape(U_trunc, (sT_open..., new_dim)), Σ_trunc
end

@generated function recalc_S!(S, T, Us::NTuple{N,T_U},
    info::Tuple{Val{i_T},Val{i_Us}}) where {N,T_U,i_T,i_Us}
    rightside = Expr(:call, :*,
        :(T[$(i_T...)]),
        (:(conj(Us[$i])[$(i_Us[i]...)]) for i in 1:N)...
    )
    return :(S.tensor = @tensor _[:] := $rightside)
end

function deqr_site!(site, q, U, perm)
    Asize_permuted = size(site.tensor)[perm]
    s_q = Asize_permuted[1:end-2]
    s_physical = size(site.tensor)[end]
    s_new_bond = size(U)[2]
    @tensor A_new_rp[:] := q[-1,1] * U[1, -3, -2]
    A_new_p = reshape(A_new_rp, (s_q..., s_new_bond, s_physical))
    site.tensor = permutedims(A_new_p, sortperm(perm))
end

function rsize(site::PESSSite, D)
    auxN = nsimps(site) - 1
    qsize = D * auxN
    rightsize = (D,size(site.tensor)[end])
    leftsize = min(qsize, prod(rightsize))
    return (leftsize, rightsize...)
end

env_inds(S::Simplex) = Tuple(
        Tuple(map(i -> (i, Val(i)), filter(!=(sind), 1:nsimps(site))))
        for (site, sind) in zip(S.sites, S.siteinds))

function static_pess_su_info(u::PESSUnitCell, i_S, max_bond_rank,
    cache::ContractionCache)
    S = u.simplices[i_S]
    sites = u.sites[collect(S.sites)]
    sinds = S.siteinds
    env_inds = env_inds(S)
    qrperms = Tuple(
        let N = nsimps(site)
            moveind!(collect(1:N+1), sind, N)
        end
        for (site, sind) in zip(sites, sinds))

    virtualsizes = size(S.tensor)[collect(virtualsiteinds(S))]
    psizes = [[size(site.tensor)[end] for site in sites]; virtualsizes...]
    opsizes = Tuple([psizes; psizes])
    Ssizes = Tuple(fill(max_bond_rank, nsites(S)); virtualsizes...)
    rsizes = [rsize(site, max_bond_rank) for site in sites]
    op_num, S_num, r_nums... = 1:(nsites(S)+2)
    conts_nonvirt = [((op_num, i), (r_num, 3))
                     for (i, r_num) in enumerate(r_nums)]
    conts_virt = [((op_num, i + length(conts_nonvirt)), (S_num, vind))
                  for (i, vind) in enumerate(virtualsiteinds(S))]
    conts_Srs = [((S_num, i), (r_num, 2))
                 for (i, r_num) in enumerate(r_nums)]
    open_op = (op_num, Tuple((length(psizes)+1):length(opsizes)))
    open_rs = [(r_num, (1,)) for r_num in r_nums]
    i_c_op, i_c_S, i_c_rs... = ncon_indices(
        [opsizes, Ssizes, rsizes...],
        vcat(conts_nonvirt, conts_virt, conts_Srs,),
        vcat(open_rs, [open_op]),
        cache) .|> Tuple

    n_virt = nvirt(S)
    n_sites = nsites(S)
    T_dim = 2 * n_sites + n_virt #site bonds, site physical, virtual physical
    Tperms = Tuple(Tuple(moveind!(moveind!(collect(1:T_dim), i, 1), n_sites + i, 2))
             for (i, _) in enumerate(sites))

    T_size = ([rsize[1] for rsize in rsizes]...,
        psizes...)
    Usizes = [(rsize[1], rsize[3], max_bond_rank) for rsize in rsizes]
    T_num, U_nums... = 1:(n_sites+1)
    conts_reS_qr = [((T_num, i), (U_num, 1)) for (i, U_num) in enumerate(U_nums)]
    conts_reS_phys = [((T_num, i + n_sites), (U_num, 2))
                       for (i, U_num) in enumerate(U_nums)]
    open_reS_Us = [(U_num, (3,)) for U_num in U_nums]
    i_reS_T, i_reS_Us... = ncon_indices(vcat(T_size, Usizes),
        vcat(conts_reS_qr, conts_reS_phys), open_reS_Us, cache) .|> Tuple
    return (
        env_inds = env_inds,
        qr_perms = qrperms,
        contract_op = (Val(i_c_op), Val(i_c_S), Val(i_c_rs)),
        svd_perms = Tperms,
        S = (Val(i_reS_T), Val(i_reS_Us)),
        sinds = sinds
        )
end

function calc_simplex_ev(u::PESSUnitCell, op, n_simplex, cache::ContractionCache)
    S = u.simplices[n_simplex]
    N = nsites(S)
    M = nvirt(S)
    NM = N+M
    braket = calc_simplex_braket(u, n_simplex, cache)
    ev = ncon((braket, op), (collect(1:(2*NM)), collect(1:(2*NM))))
    norm = ncon((braket,), ([1:NM;1:NM]))
    return ev ./ norm
end

function calc_simplex_braket(u::PESSUnitCell, n_simplex, cache::ContractionCache)
    S = u.simplices[n_simplex]
    N = nsites(S)
    sites = u.sites[collect(S.sites)]
    site_tensors = [s.tensor for s in sites]
    tensors = [S.tensor, S.tensor, site_tensors..., site_tensors...]
    conjlist = vcat([false, true], repeat([false], N), repeat([true], N))
    nS_a, nS_b = (1, 2)
    ns_as = Tuple(3:(2+N))
    ns_bs = Tuple((3+N):(2+2N))

    simplex_site_contractions = [
        ((nS, i), (ns, sind))
        for (nS, nss) in zip((nS_a, nS_b), (ns_as, ns_bs))
        for (i, (ns, sind)) in enumerate(zip(nss, S.siteinds))
    ]
    site_braket_contractions = [
        ((ns_a, i), (ns_b, i))
        for (ns_a, ns_b, site, sind)
            in
            zip(ns_as, ns_bs, sites, S.siteinds)
        for i in filter(i -> i != sind, 1:nsimps(site))
    ]
    open_a = vcat(
        [(ns_a, (nsimps(site) + 1,))
         for (ns_a, site) in zip(ns_as, sites)],
        (nS_a, Tuple(virtualsiteinds(S)))
    )
    open_b = vcat(
        [(ns_b, (nsimps(site) + 1,))
         for (ns_b, site) in zip(ns_bs, sites)],
        (nS_b, Tuple(virtualsiteinds(S)))
    )
    ninds = ncon_indices(size.(tensors),
        vcat(simplex_site_contractions, site_braket_contractions),
        vcat(open_a, open_b),
        cache
    )

    braket = ncon(tensors, ninds, conjlist)
    return braket
end

function unitcell_from_simplices(Ss::Vector{Simplex{<:Any,<:Any,T}},
    psize=2,
    initf=rand) where {T}
    T2 = real(T)
    site_sizes = [(site, sind, size(S.tensor)[i])
                  for S in Ss
                  for (i, (site, sind)) in enumerate(zip(S.sites, S.siteinds))]
    sites = PESSSite{<:Any, T, T2}[
        let
            bsizes = Tuple(map(x -> (x[3]),
                sort(filter(i -> i[1] == snum, site_sizes), by=x -> x[2])))
            tensor = initf(T, (bsizes..., psize))
            envVectors = Tuple([initf(T2, bsize) for bsize in bsizes])
            PESSSite(tensor, envVectors)
        end
        for snum in unique(first.(site_sizes))
    ]
    PESSUnitCell(sites, Ss)
end

function pess_unitcell_from_structurematrix(m::AbstractMatrix{Int},
    simplex_dims,
    pdims,
    initt,
    initv)
    pess_unitcell_from_ordered_structurematrix(make_ordered_structurematrix(m),
        simplex_dims, pdims, initt, initv)
end

function pess_unitcell_from_ordered_structurematrix(m::AbstractMatrix{Int},
    simplex_dims,
    pdims,
    initt, initv)
    T1 = eltype(initt(1))
    T2 = eltype(initv(1))
    sitedims = Vector{Int}[]
    simplex_site_map = [
        Dict(sitenum => d for (d, sitenum) in zip(ds, findall(c .!= 0)))
        for (ds, c) in zip(simplex_dims, eachcol(m))
    ]
    virtual_sites_for_simplex = [Tuple{Int,Int}[] for _ in simplex_dims]
    for (i, siterow) in enumerate(eachrow(m))
        i_simplex_for_site = findall(siterow .!= 0)
        if count(siterow .!= 0) == 1 # virtual site
            i_simplex = only(i_simplex_for_site)
            siteind = siterow[i_simplex]
            virtual_sites = virtual_sites_for_simplex[i_simplex]
            push!(virtual_sites, (siteind, i))
            sort!(virtual_sites, by=first)
        else
            site_simplex_inds = siterow[i_simplex_for_site]
            sdims = [
                smap[i] for smap in simplex_site_map[i_simplex_for_site]
            ][sortperm(site_simplex_inds)]
            push!(sitedims, vcat(sdims, pdims[i]))
        end
    end

    sites = (PESSSite{N,T1,T2} where {N})[
        PESSSite(initt(Tuple(dims)),
        Tuple(initv.(dims[1:end-1]))) for dims in sitedims]
    simplices = (Simplex{M,N,T1} where {M,N})[
        let sdims = simplex_dims[i]
            i_site_for_simplex = findall(scol .!= 0)
            simplex_site_inds = scol[i_site_for_simplex]
            virtual_dims = [pdims[vsite] for (_, vsite) in virtual_inds]
            nvirtual = length(virtual_dims)
            sdims_full = Tuple(vcat(sdims, virtual_dims))
            Simplex{nvirtual}(
                initt(sdims_full),
                Tuple(i_site_for_simplex),
                Tuple([virt[2] for virt in virtual_inds]),
                Tuple(simplex_site_inds))
        end
        for (i, (virtual_inds, scol))
        in enumerate(zip(virtual_sites_for_simplex, eachcol(m)))]
    PESSUnitCell(sites, simplices)
end


function normalized_1site_ops(ops::Dict{Tuple{Int}, Operator{T1, 1, A}} where {T1,A},
                              u::PESSUnitCell{T}, sitetypes) where {T}
    [
        let simplex_sites = (simplex.sites..., simplex.vsites...)
            sum([
            let
                occurrences = i <= nsites(simplex) ? nsimps(u.sites[i]) : 1
                ⊗([i == j ?
                   1/occurrences * ops[(sitetypes[snum],)].tensor :
                   collect((one(T) * I)(psize(u.sites[osnum])))
                   for (j, osnum) in enumerate(simplex_sites)]...) |> Operator
            end
            for (i, snum) in enumerate(simplex_sites)])
        end
        for simplex in u.simplices
    ]
end

function normalize_edge_ind(inds, n)
    inds |> collect |> sort
    o1 = inds[1][2]
    Tuple([(siten, orig_cell - o1 + 1) for (siten, orig_cell) in inds])
end

function twosite_normalization_dict(m_connect, tile_pattern)
    n = 2^ndims(tile_pattern)
    m_origin = tile_structurematrix_with_origin(m_connect, tile_pattern)
    m_normal = tile_structurematrix(m_connect, tile_pattern)
    pair_dict = Dict{NTuple{2, Tuple{Int,Int}}, Int}()
    for (i_S, scol) in enumerate(eachcol(m_normal))
        sites = findall(scol .!= 0)
        for (i,s_i) in enumerate(sites)
            for s_j in sites[i+1:end]
                inds = normalize_edge_ind(
                       ((s_i, m_origin[s_i, i_S][2]),
                        (s_j, m_origin[s_j, i_S][2])),
                    n)
                pair_dict[inds] = get(pair_dict, inds, 0) + 1
            end
        end
    end
    normal_dict = Dict{Tuple{Int,Int}, Tuple{Int,Int}}()
    for (k,v) in pair_dict
        site_pair = k[1][1],k[2][1]
        n_physical, n_nominal = get(normal_dict, site_pair, (0,0))
        normal_dict[site_pair] = n_physical + 1, n_nominal + v
    end
    Dict(k=>v[1]/v[2] for (k,v) in normal_dict)
end

function normalized_ops(ops::Dict{Tuple{Int, Int}, Operator{T,2,A}} where {T,A},
    u::PESSUnitCell, m_connect, tile_pattern, sitetypes)
    normalization_dict = twosite_normalization_dict(m_connect, tile_pattern)
    [
    let simplex_sites = (simplex.sites..., simplex.vsites...)
        site_dims = vcat(
            [size(site.tensor)[end] for site in u.sites[simplex.sites |> collect]],
                         size(simplex.tensor)[virtualsiteinds(simplex)] |> collect)
       sum([
            let
                pair_id = (s_i, s_j) |> collect |> sort |> Tuple
                op_id = (sitetypes[s_i], sitetypes[s_j]) |> collect |> sort |> Tuple
                normalization_dict[pair_id] * nsite_op(ops[op_id],
                                                       (i, j+i), site_dims)
            end
            for (i, s_i) in enumerate(simplex_sites)
            for (j, s_j) in enumerate(simplex_sites[i+1:end])
        ])
     end
        for simplex in u.simplices
    ]
end

normalized_2site_ops(
    ops::Dict{Int,Operator},
    model::PESSModel) = normalized_2site_ops(
        ops,
        model.unitcell,
        model.m_connect,
        model.tile_pattern,
        model.sitetypes)

normalized_2site_ops(
    op::Operator,
    u::PESSUnitCell,
    m_connect,
    tile_pattern) = normalized_2site_ops(
        Dict(1 => op),
        u,
        m_connect,
        tile_pattern,
        ones(Int, length(u.sites)))

normalized_2site_ops(op::Operator,
    model::PESSModel) = normalized_2site_ops(
        op,
        model.unitcell,
        model.m_connect,
        model.tile_pattern)
# gPESS:2 ends here

# [[file:../../notes.org::*gPESS][gPESS:3]]
end
# gPESS:3 ends here
