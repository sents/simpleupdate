# [[file:../SimpleUpdate.org::*gPESS][gPESS:1]]
module gPESS
import ..Definitions: s_x, s_y, s_z
import ..Interface: register!, simple_update, per_site_energy
using ..OptimalContraction
using ..Operators
using ..Util
import ..Util: cached_similar_ordered_inds, ordered_inds
export Simplex,
    PESSSite,
    PESSUnitCell,
    PESSModel,
    nsites,
    nvirt,
    virtualsiteinds,
    nsimps,
    psize,
    pess_unitcell_from_structurematrix,
    register!,
    static_pess_su_info,
    per_site_energy,
    simple_update
# gPESS:1 ends here

# [[file:../SimpleUpdate.org::*gPESS][gPESS:2]]
using LinearAlgebra: full!
using StaticArrays
using LinearAlgebra
using TensorOperations
import TensorOperations: cached_similar_from_indices, similar_from_structure
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
        siteinds::NTuple{N,Int},
    ) where {N,M,T,A<:AbstractArray{T}}
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
mutable struct PESSSite{N,T1,T2,M<:AbstractArray{T1},V<:AbstractVector{T2}}
    tensor::M
    envVectors::SizedVector{N,V,Vector{V}}
    function PESSSite(
        tensor::M,
        envVectors::NTuple{N,V},
    ) where {N,T1,T2,M<:AbstractArray{T1}, V<:AbstractVector{T2}}
        @assert ndims(tensor) == N + 1 """
            Dimension of `tensor` has to be N+1=$(N+1)!
            """
        new{N,T1,T2,M,V}(tensor, SizedVector{N,V}(envVectors))
    end
end

"""
    PESSUnitCell{T1,T2}
A unitcell consisting of a Vector of sites and a vector of simplices
"""
struct PESSUnitCell{T1,T2}
    sites::Vector{PESSSite{<:Any,T1,T2}}
    simplices::Vector{Simplex{<:Any,<:Any,T1}}
end

"""
    PESSModel{T1, T2, N}
Describes a `PESSUnitCell` together with observables that can be
calculated on the UnitCell. To model different kind of interactions
it contains a list `sitetypes` that maps the sites to an integer.
N is the number of tile directions.

# Fields
- `unitcell :: PESSUnitCell{T1, T2}`: The PESSUnitCell containing sites and simplices
- `sitetypes :: Vector{Int}`: A list of length(unitcell.sites) sites, giving each site
an integer to specify its `type`. Defaults to ones if all interactions are equal.
- `observables :: Dict{Symbol, Vector{Operator{T1}}}`: A dict containing additional
observables in or model. Always contains atleast :hamiltonian.
- `m_connect :: Array{Int, 2}`: The connection matrix describing how a primitive unit cell
connects to positively adjacent unit cells
- `tile_pattern :: Array{Int}`: The tile pattern with which the connection matrix is tiled
- `interactions :: Vector{Tuple{Tuple{Int,Int}, NTuple{N, Int}}}`:
"""
struct PESSModel{T1,T2,N}
    unitcell::PESSUnitCell{T1,T2}
    m_connect::Array{Int,2}
    tile_pattern::Array{Int,N}
    interactions::Vector{Tuple{NTuple{2,Int},NTuple{N,Int}}}
    sitetypes::Vector{Int}
    observables::Dict{Symbol,Vector{Operator{T1}}}
    cache::ContractionCache
    function PESSModel(
        unitcell::PESSUnitCell{T1,T2},
        m_connect::AbstractMatrix{Int},
        tile_pattern::AbstractArray{Int,N},
        interactions::Vector{Tuple{NTuple{2,Int},NTuple{N,Int}}},
        sitetypes::Vector{Int}=[1 for _ = 1:length(unitcell.sites)],
        observables=Dict(),
        cache::ContractionCache=ContractionCache(),
    ) where {T1,T2,N}
        new{T1,T2,N}(
            unitcell,
            m_connect,
            tile_pattern |> collect,
            interactions,
            sitetypes,
            convert(Dict{Symbol,Vector{Operator{T1}}}, observables),
            cache
        )
    end
end

function PESSModel(
    unitcell::PESSUnitCell{T1,T2},
    m_connect::AbstractMatrix{Int},
    tile_pattern::AbstractArray{Int},
    m_interactions::AbstractMatrix{Int},
    sitetypes::Vector{Int}=[1 for _ = 1:length(unitcell.sites)],
    observables=Dict(),
    cache::ContractionCache=ContractionCache()
) where {T1,T2}
    PESSModel(
        unitcell,
        m_connect,
        tile_pattern |> collect,
        interactions_from_tiling(m_interactions, tile_pattern),
        sitetypes,
        convert(Dict{Symbol,Vector{Operator{T1}}}, observables),
        cache
    )
end

Base.show(io::IO, S::Simplex{M,N,T,A}) where {M,N,T,A} = print(
    io,
    "Simplex{$(M),$(N),$(T)}: $(size(S.tensor))",
    "\n",
    "Connections: \n",
    ("$i -> $si\n" for (i, si) in zip(S.sites, S.siteinds))...,
    "Virtual sites: ",
    join(string.(S.vsites), ", "),
)

Base.show(io::IO, s::PESSSite{N,T1,T2}) where {N,T1,T2} =
    print(io, "PESSSite{$N,$T1,$T2}: $(size(s.tensor))")

Base.show(io::IO, u::PESSUnitCell{T1,T2}) where {T1,T2} = print(
    io,
    "PESSUnitCell{$T1,$T2}: $(length(u.sites)) sites, $(length(u.simplices)) simplices",
)

Base.show(io::IO, m::PESSModel{T1,T2}) where {T1,T2} = print(
    io,
    "PESSModel{$T1,$T2}: ",
    length(m.unitcell.sites),
    " sites, ",
    "$(length(m.unitcell.simplices)) simplices\n",
    "Number of interactions: ",
    length(m.interactions),
    "\n",
    "Number of sitetypes: ",
    length(m.sitetypes |> unique),
    "\n",
    "Defined observables: ",
    join(string.(keys(m.observables)), ", "),
)


nsites(::Simplex{M,N}) where {M,N} = N
nvirt(::Simplex{M,N}) where {M,N} = M
virtualsiteinds(s::Simplex) = range(nsites(s) + 1, nvirt(s))
nsimps(::PESSSite{N}) where {N} = N
psize(site::PESSSite) = size(site.tensor)[end]
nsites(u::PESSUnitCell) = length(u.sites) + sum(nvirt, u.simplices)
site_simplices(u::PESSUnitCell, snum) = [
    (i,S) for (i,S) in enumerate(u.simplices) if snum in [S.sites..., S.vsites...]
]


"""
`simple_update(m::PESSModel; τ₀, max_bond_rank, convergence, maxit, logger)`
Iterated simple update of unit cell with one operator per simplex
"""
function simple_update(m::PESSModel; kwargs...)
    simple_update(m.unitcell, m.observables[:H]; cache=m.cache, kwargs...)
end

const LogStep = @NamedTuple begin
    diff::Float64
    Δs_trunc::Matrix{Float64}
end

"""
`simple_update(u::PESSUnitCell, ops, max_bond_dim, convergence, maxit, logger)`
Iterated simple update of unit cell with one operator per bond
"""
function simple_update(
    u::PESSUnitCell,
    ops;
    τ₀=1.0,
    max_bond_rank=10,
    min_τ=1e-5,
    convergence=1e-8,
    sv_cutoff=1e-8,
    maxit=-1,
    logger=Logger{LogStep}(; printit=50),
    cache::ContractionCache=ContractionCache(),
)
    bondinfo =
        [static_pess_su_info(u, i, max_bond_rank, cache) for (i, _) in enumerate(ops)]
    it = 0
    l0 = length(logger.log)
    τ = τ₀
    while τ >= min_τ && (it < maxit || maxit < 0)
        println("τ: ", τ)
        simple_update(
            u,
            ops,
            bondinfo,
            τ;
            max_bond_rank=max_bond_rank,
            convergence=convergence,
            sv_cutoff=sv_cutoff,
            maxit=maxit - it,
            logger=logger,
        )
        it = length(logger.log) - l0
        τ /= 10
    end
    return logger
end

function simple_update(
    u::PESSUnitCell,
    ops,
    bond_infos,
    τ;
    max_bond_rank=10,
    convergence=1e-8,
    sv_cutoff=1e-8,
    maxit=-1,
    logger=nothing,
)
    eops = [exp(-τ * op) for op in ops]
    it = 0
    while it < maxit || maxit < 0
        diff = 0.0
        simplex_Δs_trunc = Vector{Float64}[]
        for (op, info, simplex) in zip(eops, bond_infos, u.simplices)
            d, Δs_trunc = simple_update_step!(
                Tuple(u.sites[collect(simplex.sites)]),
                simplex,
                op,
                info,
                max_bond_rank,
                sv_cutoff,
            )
            diff += d
            push!(simplex_Δs_trunc, Δs_trunc)
        end
        !isnothing(logger) && record!(logger, (diff, stack(simplex_Δs_trunc)))
        if diff < convergence
            return logger
        end
        it += 1
    end
    return logger
end


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
function simple_update_step!(
    sites::NTuple{N, PESSSite{<:Any,T1,T2}},
    S::Simplex{M,N,T1,A},
    op,
    info,
    max_bond_rank,
    sv_cutoff,
) where {T1,T2,N,M,A}
    qs, rs = zip(ntuple(i->
    begin
        contract_env!(sites[i], info.env_inds[i])
        qr_site(sites[i], info.qr_perms[i])
    end, length(sites))...)

    T = contract_op(op.tensor, S.tensor, tuple(rs...), info.contract_op)

    Us = n_array_type(A,Val(3))[]

    step_diff = 0.0
    Δs_trunc = T2[]

    for (i, site) in enumerate(sites)
        U, Σ, Δ_trunc = eigsvd_trunc(
            T,
            info.svd_inds[i],
            max_bond_rank,
            sv_cutoff,
        )::Tuple{similar_atype(A,3),similar_atype(A,1,T2), T2}
        step_diff += padded_inner_product(site.envVectors[info.sinds[i]], Σ)
        site.envVectors[info.sinds[i]] = Σ
        push!(Us, U)
        push!(Δs_trunc, Δ_trunc)
    end

    recalc_S!(S, T, Tuple(Us), info.S)

    for (i, (site, U, q)) in enumerate(zip(sites, Us, qs))
        deqr_site!(site, q, U, info.qr_perms[i])
        emit_env!(site, info.env_inds[i])
    end
    return step_diff, Δs_trunc
end

function qr_site(site::PESSSite{N, T1, T2, M}, perm) where {N, T1, T2, M}
    Asize_permuted = size(site.tensor)[perm]
    sA_r = Asize_permuted[end-1:end]
    sA_q = Asize_permuted[1:end-2]
    sA_qr = min(prod(sA_q), prod(sA_r))
    A_reshaped = reshape(permutedims(site.tensor, perm), (prod(sA_q), prod(sA_r)))
    q, r = qr(A_reshaped)
    r_reshaped = reshape(r, (sA_qr, sA_r...))
    return n_array_type(M, Val(2))(q), r_reshaped
end

function contract_env!(site::PESSSite, inds)
    site.tensor .= .*(site.tensor, (reshape(site.envVectors[i], n) for (i, n) in inds)...)
end

function emit_env!(site::PESSSite, inds)
    site.tensor .=
        .*(site.tensor, (reshape(1 ./ site.envVectors[i], n) for (i, n) in inds)...)
end

@generated function contract_op(
    op,
    S,
    rs::NTuple{N,T_Site},
    info::Tuple{Val{i_op},Val{i_S},Val{i_rs}},
) where {N,T_Site,i_op,i_S,i_rs}
    rightside = Expr(
        :call,
        :*,
        :(op[$(i_op...)]),
        :(S[$(i_S...)]),
        (:(rs[$i][$(i_rs[i]...)]) for i = 1:N)...,
    )
    return quote
        @ctensor out[:] := $rightside
    end
end


function eigsvd_trunc(T, inds, max_bond_rank, sv_cutoff)
    T_contr = eig_contraction(T, inds)
    out_size = size(T)[inds[3]]
    λ, U_r = eigf(
        Hermitian(reshape(T_contr, (prod(out_size), prod(out_size)))),
    )
    λ ./= sum(abs, λ)
    svs_over_cutoff = count(>=(sv_cutoff^2), λ)
    new_dim = min(svs_over_cutoff, max_bond_rank)
    Σ_trunc = sqrt.(λ[end-new_dim+1:end])
    U_trunc = U_r[:, end-new_dim+1:end]
    Δ_trunc = sum(abs, λ[1:end-new_dim])
    return reshape(U_trunc, (out_size..., new_dim)), Σ_trunc, Δ_trunc
end

@generated function eig_contraction(
    T::AbstractArray{T1,N},
    (inds_open, inds_closed, inds_out)::Tuple{NTuple{2,Int},NTuple{M,Int},SVector{2,Int}},
) where {T1,N,M}
    Ttype = eltype(T)
    syms = (gensym(), gensym(), gensym(), gensym())
    quote
        out = cached_similar_from_indices($syms[1], $Ttype, inds_open, inds_open, T, :N)
        TensorOperations.contract!(
            true,
            T,
            :N,
            T,
            :C,
            false,
            out,
            inds_open,
            inds_closed,
            inds_open,
            inds_closed,
            (1, 2, 3, 4),
            $syms[2:end],
        )
    end
end

_map_sizes(ainds, sizes) = map(((tnum, tdim),)->sizes[tnum][tdim], ainds)

@generated function recalc_S!(
    S,
    T,
    Us::NTuple{N,T_U},
    info::Tuple{Val{i_T},Val{i_Us}},
) where {N,T_U,i_T,i_Us}
    Ttype = eltype(T)
    ainds = ordered_inds((i_T, i_Us...))
    rightside =
        Expr(:call, :*, :(T[$(i_T...)]), (:(conj(Us[$i])[$(i_Us[i]...)]) for i = 1:N)...)
    return quote
        sizes = size.([T, Us...])
        outsize = _map_sizes($ainds, sizes)
        if outsize == size(S.tensor)
            out = S.tensor
        else
            out = similar_from_structure(S.tensor, $Ttype, outsize)
            S.tensor = out
        end

        @tensor S.tensor[:] = $rightside
        S.tensor /= norm(S.tensor)
    end
end

function deqr_site!(site::PESSSite{N}, q, U, perm) where {N}
    Asize_permuted = size(site.tensor)[perm]
    s_q = Asize_permuted[1:end-2]
    s_physical = size(site.tensor)[end]
    s_new_bond = size(U)[3]
    @tensor A_new_rp[:] := q[-1, 1] * U[1, -3, -2]
    A_new_p = reshape(A_new_rp, (s_q..., s_new_bond, s_physical))
    site.tensor = permutedims(A_new_p, sortperm(perm))
end

function rsize(site::PESSSite, D)
    auxN = nsimps(site) - 1
    qsize = D * auxN
    rightsize = (D, size(site.tensor)[end])
    leftsize = min(qsize, prod(rightsize))
    return (leftsize, rightsize...)
end

site_env_inds(u::PESSUnitCell, S::Simplex) = Tuple(
    Tuple([
        (i, ntuple(x -> x == i ? Colon() : 1, i)) for
        i = 1:nsimps(u.sites[site]) if i != sind
    ]) for (site, sind) in zip(S.sites, S.siteinds)
)

function static_pess_su_info(u::PESSUnitCell, i_S, max_bond_rank, cache::ContractionCache)
    S = u.simplices[i_S]
    sites = u.sites[collect(S.sites)]
    sinds = S.siteinds
    env_inds = site_env_inds(u, S)
    qrperms = Tuple(
        let N = nsimps(site)
            SVector{N + 1}(moveind!(collect(1:N+1), sind, N))
        end for (site, sind) in zip(sites, sinds)
    )

    virtualsizes = size(S.tensor)[collect(virtualsiteinds(S))]
    psizes = [[size(site.tensor)[end] for site in sites]; virtualsizes...]
    opsizes = Tuple([psizes; psizes])
    Ssizes = Tuple(fill(max_bond_rank, nsites(S)); virtualsizes...)
    rsizes = [rsize(site, max_bond_rank) for site in sites]
    op_num, S_num, r_nums... = 1:(nsites(S)+2)
    conts_nonvirt = [((op_num, i), (r_num, 3)) for (i, r_num) in enumerate(r_nums)]
    conts_virt = [
        ((op_num, i + length(conts_nonvirt)), (S_num, vind)) for
        (i, vind) in enumerate(virtualsiteinds(S))
    ]
    conts_Srs = [((S_num, i), (r_num, 2)) for (i, r_num) in enumerate(r_nums)]
    open_op = (op_num, Tuple((length(psizes)+1):length(opsizes)))
    open_rs = [(r_num, (1,)) for r_num in r_nums]
    i_c_op, i_c_S, i_c_rs... =
        ncon_indices(
            [opsizes, Ssizes, rsizes...],
            vcat(conts_nonvirt, conts_virt, conts_Srs),
            vcat(open_rs, [open_op]),
            cache,
        ) .|> Tuple

    n_virt = nvirt(S)
    n_sites = nsites(S)
    T_dim = 2 * n_sites + n_virt #site bonds, site physical, virtual physical
    Tperms = Tuple(
        Tuple(moveind!(moveind!(collect(1:T_dim), i, 1), n_sites + i, 2)) for
        (i, _) in enumerate(sites)
    )
    svd_inds = Tuple(
        let
            open_inds = (i, n_sites + i)
            closed_inds = Tuple([n for n = 1:T_dim if n ∉ open_inds])
            out_inds = SVector{2}(open_inds)
            (open_inds, closed_inds, out_inds)
        end for i = 1:n_sites
    )

    T_size = ([rsize[1] for rsize in rsizes]..., psizes...)
    Usizes = [(rsize[1], rsize[3], max_bond_rank) for rsize in rsizes]
    T_num, U_nums... = 1:(n_sites+1)
    conts_reS_qr = [((T_num, i), (U_num, 1)) for (i, U_num) in enumerate(U_nums)]
    conts_reS_phys =
        [((T_num, i + n_sites), (U_num, 2)) for (i, U_num) in enumerate(U_nums)]
    open_reS_Us = [(U_num, (3,)) for U_num in U_nums]
    i_reS_T, i_reS_Us... =
        ncon_indices(
            vcat(T_size, Usizes),
            vcat(conts_reS_qr, conts_reS_phys),
            open_reS_Us,
            cache,
        ) .|> Tuple
    return (
        env_inds=env_inds,
        qr_perms=qrperms,
        contract_op=(Val(i_c_op), Val(i_c_S), Val(i_c_rs)),
        svd_inds=svd_inds,
        S=(Val(i_reS_T), Val(i_reS_Us)),
        sinds=sinds,
    )
end

calc_simplex_ev(model::PESSModel, op, n_simplex) = calc_simplex_ev(
    model.unitcell,
    op,
    n_simplex,
    model.cache
)

function calc_simplex_ev(u::PESSUnitCell, op, n_simplex, cache::ContractionCache)
    S = u.simplices[n_simplex]
    N = nsites(S)
    M = nvirt(S)
    NM = N + M
    braket = calc_simplex_braket(u, n_simplex, cache)
    ev = ncon((braket, op.tensor), (collect(1:(2*NM)), collect(1:(2*NM))))
    norm = ncon((braket,), ([1:NM; 1:NM],))
    return ev ./ norm
end

function calc_simplex_braket(u::PESSUnitCell, n_simplex, cache::ContractionCache)
    S = u.simplices[n_simplex]
    N = nsites(S)
    sites = u.sites[collect(S.sites)]
    env_inds = site_env_inds(u, S)
    for (site, env_ind) in zip(sites, env_inds)
        contract_env!(site, env_ind)
    end
    site_tensors = [s.tensor for s in sites]
    tensors = [S.tensor, S.tensor, site_tensors..., site_tensors...]
    conjlist = vcat([false, true], repeat([false], N), repeat([true], N))
    nS_a, nS_b = (1, 2)
    ns_as = Tuple(3:(2+N))
    ns_bs = Tuple((3+N):(2+2N))

    simplex_site_contractions = [
        ((nS, i), (ns, sind)) for (nS, nss) in zip((nS_a, nS_b), (ns_as, ns_bs)) for
        (i, (ns, sind)) in enumerate(zip(nss, S.siteinds))
    ]
    site_braket_contractions = [
        ((ns_a, i), (ns_b, i)) for
        (ns_a, ns_b, site, sind) in zip(ns_as, ns_bs, sites, S.siteinds) for
        i in filter(!=(sind), 1:nsimps(site))
    ]
    open_a = vcat(
        [(ns_a, (nsimps(site) + 1,)) for (ns_a, site) in zip(ns_as, sites)],
        (nS_a, Tuple(virtualsiteinds(S))),
    )
    open_b = vcat(
        [(ns_b, (nsimps(site) + 1,)) for (ns_b, site) in zip(ns_bs, sites)],
        (nS_b, Tuple(virtualsiteinds(S))),
    )
    ninds = ncon_indices(
        size.(tensors),
        vcat(simplex_site_contractions, site_braket_contractions),
        vcat(open_a, open_b),
        cache,
    )

    braket = ncon(tensors, collect.(ninds), conjlist)

    for (site, env_ind) in zip(sites, env_inds)
        emit_env!(site, env_ind)
    end
    return braket
end

function unitcell_from_simplices(
    Ss::Vector{Simplex{<:Any,<:Any,T}},
    psize=2,
    initf=rand,
) where {T}
    T2 = real(T)
    site_sizes = [
        (site, sind, size(S.tensor)[i]) for S in Ss for
        (i, (site, sind)) in enumerate(zip(S.sites, S.siteinds))
    ]
    sites = PESSSite{<:Any,T,T2}[
        let
            bsizes = Tuple(
                map(x -> (x[3]), sort(filter(i -> i[1] == snum, site_sizes), by=x -> x[2])),
            )
            tensor = initf(T, (bsizes..., psize))
            envVectors = Tuple([initf(T2, bsize) for bsize in bsizes])
            PESSSite(tensor, envVectors)
        end for snum in unique(first.(site_sizes))
    ]
    PESSUnitCell(sites, Ss)
end

function pess_unitcell_from_unordered_structurematrix(
    m::AbstractMatrix{Int},
    simplex_dims,
    pdims,
    initt,
    initv,
)
    pess_unitcell_from_structurematrix(
        make_ordered_structurematrix(m),
        simplex_dims,
        pdims,
        initt,
        initv,
    )
end

function pess_unitcell_from_structurematrix(
    m::AbstractMatrix{Int},
    simplex_dim::Int,
    pdim::Int,
    initt,
    initv,
)
    simplex_dims = [ntuple(i->simplex_dim, count(!=(0), scol))
        for scol in eachcol(m)]
    pdims = [pdim for _ in 1:size(m)[1]]
    return pess_unitcell_from_structurematrix(
        m, simplex_dims, pdims, initt, initv
    )
end

function pess_unitcell_from_structurematrix(
    m::AbstractMatrix{Int},
    simplex_dims,
    pdims,
    initt,
    initv,
)
    T1 = eltype(initt(1))
    T2 = eltype(initv(1))
    sitedims = Vector{Int}[]
    simplex_site_map = [
        Dict(sitenum => d for (d, sitenum) in zip(ds, findall(c .!= 0))) for
        (ds, c) in zip(simplex_dims, eachcol(m))
    ]
    virtual_sites_for_simplex = [Tuple{Int,Int}[] for _ in simplex_dims]
    for (i, siterow) in enumerate(eachrow(m))
        i_simplex_for_site = findall(siterow .!= 0)
        if length(i_simplex_for_site) == 1 # virtual site
            i_simplex = only(i_simplex_for_site)
            siteind = siterow[i_simplex]
            virtual_sites = virtual_sites_for_simplex[i_simplex]
            push!(virtual_sites, (siteind, i))
            sort!(virtual_sites, by=first)
        else
            site_simplex_inds = siterow[i_simplex_for_site]
            sdims =
                [smap[i] for smap in simplex_site_map[i_simplex_for_site]][sortperm(site_simplex_inds)]
            push!(sitedims, vcat(sdims, pdims[i]))
        end
    end

    sites = (PESSSite{N,T1,T2} where {N})[
        PESSSite(initt(Tuple(dims)), Tuple(initv.(dims[1:end-1]))) for dims in sitedims
    ]
    simplices = (Simplex{M,N,T1} where {M,N})[
        let sdims = simplex_dims[i] |> collect
            i_site_for_simplex =
                filter(i_s -> count(!=(0), m[i_s, :]) > 1, findall(scol .!= 0))
            simplex_site_inds = scol[i_site_for_simplex]
            virtual_dims = [pdims[vsite] for (_, vsite) in virtual_inds]
            nvirtual = length(virtual_dims)
            sdims_full = Tuple(vcat(sdims, virtual_dims))
            Simplex{nvirtual}(
                initt(sdims_full),
                Tuple(i_site_for_simplex),
                Tuple([virt[2] for virt in virtual_inds]),
                Tuple(simplex_site_inds),
            )
        end for (i, (virtual_inds, scol)) in
        enumerate(zip(virtual_sites_for_simplex, eachcol(m)))
    ]
    PESSUnitCell(sites, simplices)
end


function normalized_1site_ops(
    ops::Dict{Tuple{Int},Operator{T1,1,A}} where {T1,A},
    u::PESSUnitCell{T},
    sitetypes,
) where {T}
    [
        let simplex_sites = (simplex.sites..., simplex.vsites...)
            sum([
                let
                    isvirt = i > nsites(simplex)
                    occurrences = isvirt ? 1 : nsimps(u.sites[i])
                    ⊗(
                        [
                        let
                            oisvirt = j>nsites(simplex)
                            opsize = oisvirt ? size(simplex.tensor)[j] :
                                               psize(u.sites[osnum])
                            i == j ? 1 / occurrences * ops[(sitetypes[snum],)].tensor :
                            collect((one(T) * I)(opsize))
                        end
                        for (j, osnum) in enumerate(simplex_sites)
                        ]...,
                    ) |> Operator
                end for (i, snum) in enumerate(simplex_sites)
            ])
        end for simplex in u.simplices
    ]
end

function calc_onesite_ops(ops, model)
    u = model.unitcell
    sitetypes = model.sitetypes
    sites_ops = [
        [
            (S_num, onesite_to_simplex_ops(ops, u, S_num, s_num, sitetypes)) for
            (S_num, _) in site_simplices(u, s_num)
        ] for s_num = 1:nsites(u)
    ]
    return sites_ops
end

function onesite_to_simplex_ops(
    ops::Dict{Tuple{Int},O},
    u::PESSUnitCell,
    S_num,
    s_num,
    sitetypes) where {O<:Operator{<:Any,1,<:Any}}
    S = u.simplices[S_num]
    isvirt = s_num ∈ S.vsites
    occurrences = isvirt ? 1 : nsimps(u.sites[s_num])
    simp_site_i = findfirst(==(s_num), isvirt ? S.vsites : S.sites)
    simplex_site_pdims =
        [psize.(u.sites[[S.sites...]])..., size(S.tensor)[virtualsiteinds(S)]...]
    op = ops[(sitetypes[s_num],)]
    simplex_op = nsite_op(1/occurrences * op, (simp_site_i,), simplex_site_pdims)
    return simplex_op
end

function edge_signature(((s1, o1), (s2, o2)))
    sinds = ((s1, o1), (s2, o2)) |> collect |> sort
    diff = o1 - o2
    Tuple([siten for (siten, origin) in sinds]), diff.I
end

function interactions_from_tiling(m_connect, tile_pattern)
    D = ndims(tile_pattern)
    interactions =
        [
            combinations(
                [(i, c1[i][2]) for i in findall(c1 .!= Ref((0, zero(CartesianIndex{D}))))],
                2,
            ) for c1 in eachcol(tile_structurematrix_with_origin(m_connect, tile_pattern))
        ] |>
        Iterators.flatten .|>
        edge_signature |>
        unique |>
        sort
    return interactions
end

function twosite_normalization_dict(
    m_connect,
    tile_pattern,
    interactions=interactions_from_tiling(m_connect, tile_pattern),
)
    D = ndims(tile_pattern)
    m_origin = tile_structurematrix_with_origin(m_connect, tile_pattern)
    m_normal = tile_structurematrix(m_connect, tile_pattern)
    pair_dict = Dict{Tuple{NTuple{2,Int},NTuple{D,Int}},Int}()
    for (i_S, scol) in enumerate(eachcol(m_normal))
        sites = findall(scol .!= 0)
        for (i, s_i) in enumerate(sites)
            for s_j in sites[i+1:end]
                inds = edge_signature((
                    (s_i, m_origin[s_i, i_S][2]),
                    (s_j, m_origin[s_j, i_S][2]),
                ))

                (inds in interactions) || continue
                pair_dict[inds] = get(pair_dict, inds, 0) + 1
            end
        end
    end
    Dict(k => 1 / v for (k, v) in pair_dict)
end

function register!(model::PESSModel, ops, name)
    normalized_ops = normalized_ops(ops, model)
    model.observables[name] = normalized_ops
end

function register!(model::PESSModel, ops::Vector{Operator}, name)
    model.observables[name] = ops
end

function Operators.normalized_ops(
    ops::Dict{Tuple{Int,Int},Operator{T,2,A}} where {T,A},
    u::PESSUnitCell,
    m_connect,
    tile_pattern,
    sitetypes,
    interactions,
)
    normalization_dict = twosite_normalization_dict(m_connect, tile_pattern, interactions)
    m_origin = tile_structurematrix_with_origin(m_connect, tile_pattern)
    [
        let simplex_sites = (simplex.sites..., simplex.vsites...)
            site_dims = vcat(
                [size(site.tensor)[end] for site in u.sites[simplex.sites|>collect]],
                size(simplex.tensor)[virtualsiteinds(simplex)] |> collect,
            )
            sum([
                let
                    op_id = (sitetypes[s_i], sitetypes[s_j]) |> collect |> sort |> Tuple
                    normalization_dict[edge_id] *
                    nsite_op(ops[op_id], (i, j + i), site_dims)
                end for (i, s_i) in enumerate(simplex_sites) for
                (j, (s_j, edge_id)) in enumerate(
                    map(simplex_sites[i+1:end]) do s_j
                        (
                            s_j,
                            edge_signature((
                                (s_i, m_origin[s_i, i_S][2]),
                                (s_j, m_origin[s_j, i_S][2]),
                            )),
                        )
                    end,
                ) if edge_id in interactions
            ])
        end for (i_S, simplex) in enumerate(u.simplices)
    ]
end

Operators.normalized_ops(
    ops::Dict{Tuple{Int},Operator{T,1,A}} where {T,A},
    model::PESSModel) = normalized_1site_ops(ops, model.unitcell, model.sitetypes)
Operators.normalized_ops(
    ops::Dict{Tuple{Int,Int},Operator{T,2,A}} where {T,A},
    model::PESSModel) =
    normalized_ops(
        ops,
        model.unitcell,
        model.m_connect,
        model.tile_pattern,
        model.sitetypes,
        model.interactions,
    )

function per_site_energy(model::PESSModel)
    nsites = length(model.unitcell.sites)
    simplex_energies = [
        calc_simplex_ev(model.unitcell, op, i, model.cache) for
        (i, op) in enumerate(model.observables[:H])
    ]
    real(sum(simplex_energies) / nsites)
end


function magnetizations(model::PESSModel)
    spin_ops = Operator.((s_x, s_y, s_z))
    spin_ops_dict = [Dict((i,) => s_i for i in model.sitetypes) for s_i in spin_ops]
    spins_sites_ops = [calc_onesite_ops(s_i, model) for s_i in spin_ops_dict]
    site_magnetizations = [
        [
            sum(
                calc_simplex_ev(model, op, n_S) for
                (n_S, op) in spins_sites_ops[spin_dir][site_num]
            ) for spin_dir = 1:3
        ] for site_num = 1:nsites(model.unitcell)
    ]
    return site_magnetizations
end
# gPESS:2 ends here

# [[file:../SimpleUpdate.org::*gPESS][gPESS:3]]
end
# gPESS:3 ends here
