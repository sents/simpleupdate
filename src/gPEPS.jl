# [[file:../SimpleUpdate.org::*gPEPS][gPEPS:1]]
module gPEPS
using ..Util
using ..Operators
export Site,
    Site2Operator,
    Bond,
    UnitCell,
    simple_update,
    calc_1site_ev,
    calc_2site_ev,
    normalized_1site_ops,
    unitcell_from_structurematrix,
    PEPS_SU_LogStep,
    PEPSModel,
    register!,
    per_site_energy

# gPEPS:1 ends here

# [[file:../SimpleUpdate.org::*gPEPS][gPEPS:2]]
using LinearAlgebra
using TensorOperations

"""
    Site{T <: AbstractArray}
Represents a site in a PEPS state.

Holds the site `tensor` a `N` dimensional array. The first `N-1` dimensions describe
virtual (or bond) dimensions. The `N`th dimension is the physical dimension.
"""
mutable struct Site{T<:AbstractArray}
    tensor::T
end


"""
    Site2Operator{T<:AbstractArray{t,4} where t}

Describes operator acting on the physical indices of 2 sites.

Dimensions:
- 1: Site A in
- 2: Site B in
- 3: Site A out
- 4: Site B out
"""
const Site2Operator = Operator{T,2} where {T}

"""
    Bond(vector, A, B, Aind, Bind)

Holds information about bonds between two PEPS sites in the context of a `UnitCell`.
`tensor` is the simple update bond tensor.  `A` is the index of the first `Site` of the
Bond as ordered in the `UnitCell`, `B` is the second. `Aind` and `Bind` are the indices
of Site A or B that the Bond binds to.

`tensor` has two indices. The first binding to `Aind` of `A` and the second to `Bind` of
`B`.  """
mutable struct Bond{T<:AbstractVector}
    vector::T
    A::Int
    B::Int
    Aind::Int
    Bind::Int
end

Base.show(io::IO, b::Bond{T}) where {T} =
    print(io, "Bond{$(repr(T))}[$(length(b.vector))]($(b.A),$(b.B),$(b.Aind),$(b.Bind))")
"""
    UnitCell(sites, bonds)

A unit cell of a iPEPS tensor network.
sites is a `Vector` of `Site`s describing the sites in the `UnitCell`. The order of
sites in the Vector is important. `bonds` is a Vector of `Bond`s describing all the
bonds necessary to cover the lattice.
"""
struct UnitCell
    sites::Vector{Site}
    bonds::Vector{Bond}
end

struct PEPSModel
    unitcell :: UnitCell
    sitetypes::Vector{Int}
    observables::Dict{Symbol,Vector{Site2Operator}}
    function PEPSModel(
        unitcell::UnitCell,
        sitetypes::Vector{Int}=[1 for _ = 1:length(unitcell.sites)],
        observables=Dict()
    )
        new(
            unitcell,
            sitetypes,
            convert(Dict{Symbol,Vector{Site2Operator}}, observables)
        )
    end
end

function register!(model::PEPSModel, ops, name)
    model.observables[name] = convert(Vector{Site2Operator}, ops)
end

function simple_update(m::PEPSModel; kwargs...)
    simple_update(m.unitcell, m.observables[:H]; kwargs...)
end

function per_site_energy(model::PEPSModel)
    nsites = length(model.unitcell.sites)
    bond_energies = [
        calc_2site_ev(model.unitcell, op, i) for
        (i, op) in enumerate(model.observables[:H])
    ]
    real(sum(bond_energies) / nsites)
end

involved(sitenum, bond) = bond.A == sitenum || bond.B == sitenum
function auxbonds(bonds, sitenum, bondnum)
    return bonds[involved.(sitenum, bonds).&&eachindex(bonds).!=bondnum]
end

nbonds(u::UnitCell, sitenum) = count(involved.(sitenum, u.bonds))

"""    ind(b::Bond, i)
Helper function to get the Bond indice of a bond b.
i is the id of a Site.
"""
ind(b::Bond, i) = i == b.A ? b.Aind : b.Bind

shape_to_last(n) = ntuple(x -> x == n ? Colon() : 1, n)

"""
    static_simpleupdate_info(A, B, bond, auxbonds_A, auxbonds_B; cache=nothing)
Calculate nessecary information about a simple update step for a certain `bond`.
The information is returned as a NamedTuple. Some of the values contain value types to
    dispatch generated functions.
"""
function static_simpleupdate_info(
    A::Site{S1},
    B::Site{S2},
    bond,
    auxbonds_A,
    auxbonds_B,
) where {T,N1,N2,S1<:AbstractArray{T,N1},S2<:AbstractArray{T,N2}}

    indsauxa = [[0]; [ind(b, bond.A) for b in auxbonds_A]]
    indsauxb = [[0]; [ind(b, bond.B) for b in auxbonds_B]]
    indsrea = copy(indsauxa)
    indsreb = copy(indsauxb)

    if prod(size(B.tensor)) > prod(size(A.tensor))
        indsauxb[1] = bond.Bind
    else
        indsauxa[1] = bond.Aind
    end

    shapeauxa = shape_to_last.(indsauxa)
    shapeauxb = shape_to_last.(indsauxb)

    shaperea = shape_to_last.(indsrea)
    shapereb = shape_to_last.(indsreb)

    qrpermA = moveind!(collect(1:N1), bond.Aind, N1)
    qrpermB = moveind!(collect(1:N2), bond.Bind, N2)

    permA = moveind!(collect(1:N1), N1, bond.Aind)
    permB = moveind!(collect(1:N2), N2, bond.Bind)

    return (
        auxA=shapeauxa |> Val ∘ Tuple,
        auxB=shapeauxb |> Val ∘ Tuple,
        reA=shaperea |> Val ∘ Tuple,
        reB=shapereb |> Val ∘ Tuple,
        qrA_perm=qrpermA,
        qrB_perm=qrpermB,
        permA=permA,
        permB=permB,
    )
end

"""
    simple_update_information(u::UnitCell, bondnum)

Calculate auxillary bonds and contraction information for a simple update step on bond
of `bondnumber`"""
function simple_update_information(u::UnitCell, bondnum)
    bond = u.bonds[bondnum]
    iA = bond.A
    iB = bond.B
    A = u.sites[iA]
    B = u.sites[iB]
    auxbonds_A = auxbonds(u.bonds, iA, bondnum)
    auxbonds_B = auxbonds(u.bonds, iB, bondnum)
    info = static_simpleupdate_info(A, B, bond, auxbonds_A, auxbonds_B)
    return (info, auxbonds_A, auxbonds_B)
end

@generated function contract_bonds!(
    A::AbstractArray{T,N},
    bond_tensors,
    ::Val{order},
) where {T,N,order}
    bonds = [:(reshape(bond_tensors[$n], $(order[n]))) for n = 1:(N-1) if order[n] ≠ ()]
    return :(A .= .*(A, $(bonds...)))
end

@generated function contract_bonds(
    A::AbstractArray{T,N},
    bond_tensors,
    ::Val{order},
) where {T,N,order}
    bonds = [:(reshape(bond_tensors[$n], $(order[n]))) for n = 1:(N-1) if order[n] ≠ ()]
    return :(.*(A, $(bonds...)))
end

@generated function contract_2siteoperator(
    A::AbstractArray{T,N},
    B::AbstractArray{T,M},
    op::AbstractArray{T,O},
    order_A::Val{K},
) where {T,N,M,O,K}
    leftside = Expr(:call, :*, :(A[$(K.A...)]), :(B[$(K.B...)]), :(op[$(K.op...)]))
    return :(@tensor S[:] := $leftside)
end

"""
`simple_update_step!(A, B, op, bond, info, max_bond_rank, sv_cutoff)`

Simple update step for a single 2-site operator.
"""
function simple_update_step!(
    A::Site{S1},
    B::Site{S2},
    op::Site2Operator,
    bond,
    contraction_info,
    max_bond_rank,
    sv_cutoff=0.0,
) where {T,N1,N2,S1<:AbstractArray{T,N1},S2<:AbstractArray{T,N2}}
    info, auxbonds_A, auxbonds_B = contraction_info
    auxtensors_A = [bond.vector for bond in auxbonds_A]
    auxtensors_B = [bond.vector for bond in auxbonds_B]
    p_A = size(A.tensor)[end]
    p_B = size(B.tensor)[end]

    old_bond_dim = length(bond.vector)

    # Only one of them is going to contract bond.vector
    contract_bonds!(A.tensor, [[bond.vector]; auxtensors_A], info.auxA)
    contract_bonds!(B.tensor, [[bond.vector]; auxtensors_B], info.auxB)

    Asizep = size(A.tensor)[info.qrA_perm]
    sA_bond = Asizep[end-1:end]
    sA_rest = Asizep[1:end-2]
    sA_qr = min(prod(sA_bond), prod(sA_rest))
    Ar = reshape(permutedims(A.tensor, info.qrA_perm), (prod(sA_rest), prod(sA_bond)))
    Q_A, R_A = qr(Ar)
    R_Ar = reshape(R_A, (sA_qr, sA_bond...))

    Bsizep = size(B.tensor)[info.qrB_perm]
    sB_bond = Bsizep[end-1:end]
    sB_rest = Bsizep[1:end-2]
    sB_qr = min(prod(sB_bond), prod(sB_rest))
    Br = reshape(permutedims(B.tensor, info.qrB_perm), (prod(sB_rest), prod(sB_bond)))
    Q_B, R_B = qr(Br)
    R_Br = reshape(R_B, (sA_qr, sB_bond...))

    # Optimal for D>=d²
    @tensor S[:] := R_Ar[-1, 2, 1] * R_Br[-3, 3, 1] * op.tensor[2, 3, -2, -4]

    S_r = reshape(S, (sA_qr * p_A, sB_qr * p_B))
    F = svd!(S_r) # maybe Lancos TSVD?
    U, E, Vt = F.U, F.S, F.Vt

    E ./= norm(E)
    svs_over_cutoff = count(>=(sv_cutoff), E)
    new_bond_dim = min(svs_over_cutoff, max_bond_rank)
    new_bond = E[1:new_bond_dim]

    R_A_new = reshape(U[:, 1:new_bond_dim], (sA_qr, p_A, new_bond_dim))
    R_B_new = reshape(Vt[1:new_bond_dim, :], (new_bond_dim, sB_qr, p_B))

    @tensor A_new[l, p, b] := Matrix(Q_A)[l, x] * R_A_new[x, p, b]
    @tensor B_new[l, p, b] := R_B_new[b, x, p] * Matrix(Q_B)[l, x]

    A_new_r = reshape(A_new, (Asizep[1:end-1]..., new_bond_dim))
    B_new_r = reshape(B_new, (Bsizep[1:end-1]..., new_bond_dim))

    if new_bond_dim == old_bond_dim
        permutedims!(A.tensor, A_new_r, info.permA)
        permutedims!(B.tensor, B_new_r, info.permB)
    else
        A.tensor = permutedims(A_new_r, info.permA)
        B.tensor = permutedims(B_new_r, info.permB)
    end

    step_diff = padded_inner_product(bond.vector, new_bond)
    bond.vector = new_bond

    # Re-Emit bond tensors
    contract_bonds!(A.tensor, [[bond.vector]; [inv.(t) for t in auxtensors_A]], info.reA)
    contract_bonds!(B.tensor, [[bond.vector]; [inv.(t) for t in auxtensors_B]], info.reB)

    return step_diff
end

function calc_1site_BraKet(u::UnitCell, sitenum)
    bonds = filter(bond -> involved(sitenum, bond), u.bonds)
    A = contract_bonds(
        u.sites[sitenum].tensor,
        [bond.vector for bond in bonds],
        Val(Tuple(shape_to_last(ind(bond, sitenum)) for bond in bonds)),
    )
    N = ndims(A)
    inds = ncon_indices(size.([A, A]), [((1, i), (2, i)) for i = 1:(N-1)], [(1, N), (2, N)])
    AA = ncon([A, A], inds, [false, true])
    return AA
end

function calc_2site_BraKet(u::UnitCell, alongbond)
    bond = u.bonds[alongbond]
    A = u.sites[bond.A].tensor
    B = u.sites[bond.B].tensor
    auxbonds_A = auxbonds(u.bonds, bond.A, alongbond)
    auxbonds_B = auxbonds(u.bonds, bond.B, alongbond)
    auxtensors_A = [bond.vector for bond in auxbonds_A]
    auxtensors_B = [bond.vector for bond in auxbonds_B]

    if prod(size(A)) <= prod(size(B))
        ainds = Tuple(shape_to_last.(
            (bond.Aind, (ind(b, bond.A) for b in auxbonds_A)...)
        ))
        binds = Tuple(shape_to_last.(
            (0,(ind(b, bond.B) for b in auxbonds_B)...)
        ))
    else
        binds = Tuple(shape_to_last.(
            (bond.Bind, (ind(b, bond.B) for b in auxbonds_B)...)
        ))
        ainds = Tuple(shape_to_last.(
            (0, (ind(b, bond.A) for b in auxbonds_A)...)
        ))
    end

    Ab = contract_bonds(A, [[bond.vector]; auxtensors_A], Val(ainds))
    Bb = contract_bonds(B, [[bond.vector]; auxtensors_B], Val(binds))
    N = ndims(A) + ndims(B) - 1

    cinds = ncon_indices(
        size.([A, B]),
        [((1, bond.Aind), (2, bond.Bind))],
        [
            (1, sort([ind(b, bond.A) for b in auxbonds_A])),
            (2, sort([ind(b, bond.B) for b in auxbonds_B])),
            (1, (ndims(A),)),
            (2, (ndims(B),)),
        ],
    )
    AB = ncon([Ab, Bb], cinds)

    AB_contractions = [((1, i), (2, i)) for i = 1:(ndims(AB)-2)]
    AB_open =
        [(1, (ndims(AB) - 1,)), (1, (ndims(AB),)), (2, (ndims(AB) - 1,)), (2, (ndims(AB),))]
    ABABinds = ncon_indices(size.([AB, AB]), AB_contractions, AB_open)
    ABAB = ncon([AB, AB], ABABinds, [false, true])
    return ABAB
end

function calc_1site_ev(u::UnitCell, op, sitenum)
    AA = calc_1site_BraKet(u, sitenum)
    @tensor ev[] := AA[1, 2] * op[1, 2]
    @tensor AA_norm[] := AA[1, 1]
    return ev ./ AA_norm
end

function calc_2site_ev(u::UnitCell, op, alongbond)
    ABAB = calc_2site_BraKet(u, alongbond)
    @tensor ev[] := ABAB[1, 2, 3, 4] * op.tensor[1, 2, 3, 4]
    @tensor ABAB_norm[] := ABAB[1, 2, 1, 2]
    return ev ./ ABAB_norm
end

const LogStep = @NamedTuple begin
    diff::Float64
    svs::Vector{Vector{Float64}}
end
const PEPS_SU_LogStep = LogStep

"""
`simple_update(u::UnitCell, ops, max_bond_dim, convergence, maxit, logger)`
Iterated simple update of unit cell with one operator per bond
"""
function simple_update(
    u::UnitCell,
    ops;
    τ₀=1.0,
    max_bond_rank=10,
    min_τ=1e-5,
    convergence=1e-8,
    sv_cutoff=1e-8,
    maxit=-1,
    logger=Logger{LogStep}(printit=50),
)
    bondinfo = [simple_update_information(u, i) for (i, _) in enumerate(ops)]
    it = 0
    τ = τ₀
    while τ >= min_τ
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
        it += length(logger.log)
        τ /= 10
    end
    return logger
end

function simple_update(
    u,
    ops,
    bondinfo,
    τ;
    max_bond_rank=10,
    convergence=1e-8,
    sv_cutoff=1e-8,
    maxit=-1,
    logger=nothing,
)
    eops = [exp(-τ * op) for op in ops]
    for it = 1:maxit
        diff = 0.0
        for (op, info, bond) in zip(eops, bondinfo, u.bonds)
            d = simple_update_step!(
                u.sites[bond.A],
                u.sites[bond.B],
                op,
                bond,
                info,
                max_bond_rank,
                sv_cutoff,
            )
            diff += d
        end
        !isnothing(logger) && record!(logger, (diff, [b.vector for b in u.bonds]))
        if diff < convergence
            return logger
        end
    end
    return logger
end

function normalized_1site_ops(op, u::UnitCell)
    [
        (op ⊗ I(size(u.sites[bond.B].tensor)[end])) / nbonds(u, bond.A) .+
        (I(size(u.sites[bond.A].tensor)[end]) ⊗ op) / nbonds(u, bond.B) for bond in u.bonds
    ]
end

function unitcell_from_structurematrix(M, bonddims, pdims=fill(2, size(M)[1]), initf=rand)
    bonds = Bond[]
    sitedims = [
        let bondinds = findall(siterow .!= 0)
            [
                bonddims[bondinds][sortperm(siterow[bondinds])]
                [pdims[i]]
            ]
        end for (i, siterow) in enumerate(eachrow(M))
    ]
    sites = [Site(initf(ComplexF64, sdims...)) for sdims in sitedims]
    bonds = [
        let bdim = bonddims[i]
            sitenums = findall(bondcol .!= 0)
            siteinds = bondcol[sitenums]
            Bond(initf(bdim), sitenums..., siteinds...)
        end for (i, bondcol) in enumerate(eachcol(M))
    ]

    return UnitCell(sites, bonds)
end
unitcell_from_structurematrix(M, bonddims, pdim::Int, initf=rand) =
    unitcell_from_structurematrix(M, bonddims, fill(pdim, size(M)[1]), initf)
# gPEPS:2 ends here

# [[file:../SimpleUpdate.org::*gPEPS][gPEPS:3]]
end
# gPEPS:3 ends here
